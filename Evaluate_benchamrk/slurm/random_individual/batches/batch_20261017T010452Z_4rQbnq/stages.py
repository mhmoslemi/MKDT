"""One bounded SSL stage or all probes for one seed; never retrain for a target."""
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import sys

batch = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(batch / 'code/Evaluate_benchamrk'))
import torch
from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
from data import RawDataset, select_subset
from evaluate import linear_probe
from run import aggregate, clean_temporary_checkpoints, expected_evaluations, evaluation_targets, size_key
from train_ssl import train


def main(task_key):
    with (batch / 'tasks.tsv').open() as stream:
        rows = [row for row in csv.DictReader(stream, delimiter='\t') if row['key'] == task_key]
    if len(rows) != 1:
        raise ValueError('Unknown or duplicate task key')
    task = rows[0]
    if task['account'] != os.environ['SLURM_JOB_ACCOUNT']:
        raise ValueError('Account mismatch')
    validation = json.loads((batch / 'validation.json').read_text())
    if not validation.get('complete') or len(validation['cases']) != 18:
        raise ValueError('Validation has not passed')
    protocol = json.loads((batch / 'code/Evaluate_benchamrk/protocol.json').read_text())
    if digest_json(protocol) != validation['production_protocol_sha256']:
        raise ValueError('Validation used a different production recipe')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is unavailable')
    source, model, method = task['source'], task['model'], task['ssl']
    percentage, seed = float(task['subset']), int(task['seed'])
    targets = evaluation_targets(source, model)
    requested = expected_evaluations(source, model, False, protocol)
    root = batch / 'results'
    group = root / source / model / 'random' / f'pt{size_key(percentage)}' / method
    folder = group / f'seed_{seed:03d}'
    with exclusive_run(folder):
        source_set = RawDataset('/scratch/mmoslem3/data', source, True)
        if source_set.fingerprint != validation['datasets'][source][0]:
            raise ValueError('Source dataset changed since validation')
        selected, selection = select_subset(source_set, 'random', percentage, seed, protocol,
                                             root / 'selections', 0, 'cuda')
        signature = {'protocol': protocol, 'code_hashes': code_hashes(), 'source': source,
                     'source_fingerprint': source_set.fingerprint,
                     'target_fingerprints': {name: validation['datasets'][name] for name in targets},
                     'model': model, 'selection_method': 'random', 'ssl': method,
                     'subset': percentage, 'synthetic': None}
        signature_hash = digest_json(signature)
        identity = {'signature': signature_hash, 'seed': seed, 'selection': selection,
                    'selection_method': 'random'}
        config = {'identity': identity, 'signature': signature}
        config_path = folder / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Existing seed has a different configuration')
        atomic_json(config_path, config)
        atomic_json(folder / f'environment_{task_key}.json', {
            'job': os.environ['SLURM_JOB_ID'], 'account': os.environ['SLURM_JOB_ACCOUNT'],
            'phase': task['phase'], 'torch': str(torch.__version__),
            'gpu': torch.cuda.get_device_name(), 'workers': 0, 'threads': 1})
        if (folder / 'complete.json').exists():
            if task['phase'] == 'ssl':
                print('SKIP already completed seed', flush=True)
                return
            # No silent success after result corruption or an incompatible seed.
            from run import validate_completed
            datasets = {name: (RawDataset('/scratch/mmoslem3/data', name, True),
                               RawDataset('/scratch/mmoslem3/data', name, False)) for name in targets}
            validate_completed(folder, identity, datasets, requested)
        elif task['phase'] == 'ssl':
            checkpoint, metadata = train(selected, protocol, source, model, method, seed,
                                          folder, identity, 0, 'cuda',
                                          stop_after_epoch=int(task['end_epoch']))
            atomic_json(folder / f'stage_{task["end_epoch"]}.json', {
                'complete': True, 'end_epoch': int(task['end_epoch']),
                'ssl_complete': checkpoint is not None, 'steps': metadata['steps'],
                'identity': identity, 'job': os.environ['SLURM_JOB_ID']})
            print('SSL STAGE COMPLETE', task, flush=True)
            return
        else:
            checkpoint = folder / 'encoder.pt'
            encoder, _, metadata = load_encoder(checkpoint, identity, 'cpu')
            del encoder
            if metadata['epochs'] != protocol['ssl']['subset_epochs']:
                raise ValueError('Cannot probe an unfinished encoder')
            checkpoint_hash = file_hash(checkpoint)
            hashes = {}
            # Load one downstream dataset at a time, not all seven into RAM.
            for target in targets:
                train_set = source_set if target == source else RawDataset('/scratch/mmoslem3/data', target, True)
                test_set = RawDataset('/scratch/mmoslem3/data', target, False)
                if [train_set.fingerprint, test_set.fingerprint] != validation['datasets'][target]:
                    raise ValueError('Target dataset changed since validation: ' + target)
                for label in protocol['evaluation']['label_percentages']:
                    output = folder / 'evaluations' / f'{target}_lbl{size_key(label)}.json'
                    result = linear_probe(checkpoint, identity, protocol, source, target,
                                          train_set, test_set, label, seed, output, 0, 'cuda')
                    if result['encoder_sha256'] != metadata['encoder_sha256']:
                        raise RuntimeError('Probes did not use the same encoder')
                    hashes[output.name] = file_hash(output)
                del train_set, test_set
                gc.collect()
            atomic_json(folder / 'complete.json', {'complete': True, 'identity': identity,
                'encoder_sha256': metadata['encoder_sha256'], 'checkpoint_file_sha256': checkpoint_hash,
                'checkpoint_retained': False, 'evaluations': requested, 'result_hashes': hashes,
                'ssl': {key: metadata[key] for key in ('epochs', 'steps', 'effective_batch')}})
            clean_temporary_checkpoints(folder, False)
    with (group / '.aggregation_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        aggregate(group, protocol['subset_seeds'], requested, signature_hash)
    print('EVALUATION COMPLETE', task, flush=True)


if __name__ == '__main__':
    main(sys.argv[2])
