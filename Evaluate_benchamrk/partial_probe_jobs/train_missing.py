"""Train or resume only the missing encoders approved by remaining_plan.js."""
import argparse
import functools
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
spec = importlib.util.spec_from_file_location('remaining_main_base', BENCH / 'main_finish_jobs/worker.py')
main_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_base)
base = main_base.base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('synthetic_root', type=Path)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or os.environ.get('SLURM_JOB_ACCOUNT') not in ('aip-boyuwang', 'aip-yiweilu')
            or not base.torch.cuda.is_available() or base.torch.cuda.device_count() != 1):
        parser.error('Use this Vulcan Wang/Lu GPU launcher')
    if not args.manifest.resolve().is_relative_to(HERE / 'batches'):
        parser.error('Unexpected remaining-table manifest')
    job = base.read(args.manifest)
    definitions = base.read(HERE / 'cells.json') + base.read(HERE / 'remaining_cells.json')
    groups = [g for g in definitions if g['id'] == job['group_id']]
    if len(groups) != 1:
        raise ValueError('Unapproved encoder group')
    group = groups[0]
    cifar02 = (group['source'] in ('CIFAR10', 'CIFAR100') and group['kind'] == 'mkdt'
               and group['subset'] == 0.2 and group['epochs'] == (2000 if group['ssl'] == 'simclr' else 10000))
    generated_ours = (group['id'] == 'ours_TinyImageNet_p2_simclr'
                     and group.get('generation_job_name') in job['dependencies'])
    if (job['stage'] != 'ssl' or not job['remaining'] or not (group['source'] == 'TinyImageNet' or cifar02)
            or group['scheme'] not in ('one_ssl', 'two_ssl')
            or job['ssl_seed'] not in ([0] if group['scheme'] == 'one_ssl' else [0, 1])
            or job['account'] != os.environ.get('SLURM_JOB_ACCOUNT') or job['code_hashes'] != base.code_hashes()
            or (job['source_file_sha256'] is None and not (cifar02 or generated_ours))
            or (job['source_file_sha256'] is not None
                and base.file_hash(job['source_file']) != job['source_file_sha256'])):
        raise ValueError('Encoder scope, synthetic inputs, account or code changed')
    args.data_root, args.synthetic_root = args.data_root.resolve(), args.synthetic_root.resolve()
    base.torch.set_num_threads(1)
    base.torch.set_num_interop_threads(1)
    base.torch.backends.cuda.matmul.allow_tf32 = False
    base.torch.backends.cudnn.allow_tf32 = False

    @functools.lru_cache(maxsize=1)
    def datasets(source):
        return base.RawDataset(args.data_root, source, True), base.RawDataset(args.data_root, source, False)

    task = {**group, 'model': 'ResNet18', 'seed': job['ssl_seed'], 'parent_dir': None}
    folder = base.inside_bench(group['checkpoint_group']) / f'ssl_seed_{job["ssl_seed"]:03d}'
    with base.group_reader(folder.parent), base.exclusive_run(folder):
        synthetic, expected = main_base.synthetic_config(task, args, datasets)
        config_path = folder / 'config.json'
        if config_path.exists():
            config = base.read(config_path)
            # Keep the original driver identity when resuming an older, identical recipe.
            fields = ('protocol', 'source', 'model', 'subset', 'ssl', 'selection_method',
                      'source_fingerprint', 'synthetic', 'ssl_seeds', 'probe_seeds', 'code_hashes')
            if (any(config['signature'][k] != expected['signature'][k] for k in fields)
                    or config['identity'] != {'signature': base.digest_json(config['signature']),
                                             'seed': task['seed'], 'selection_method': task['kind']}):
                raise ValueError('Saved SSL recipe or synthetic inputs changed; refusing to restart')
        else:
            config = expected
            base.atomic_json(config_path, config)
        print('RESUME SSL:', group['id'], task['seed'], 'saved epoch estimate', job['resume_epoch_estimate'],
              '->', group['epochs'], flush=True)
        checkpoint, saved = base.train(synthetic, config['signature']['protocol'], group['source'],
            'ResNet18', group['ssl'], task['seed'], folder, config['identity'], workers=0, device='cuda')
        marker = {'complete': True, 'identity': config['identity'], 'epochs': saved['epochs'],
                  'encoder_sha256': saved['encoder_sha256'], 'checkpoint_file_sha256': base.file_hash(checkpoint)}
        if not saved['complete'] or saved['epochs'] != group['epochs']:
            raise ValueError('SSL did not finish its original epoch budget')
        marker_path = folder / 'ssl_complete.json'
        if marker_path.exists():
            old = base.read(marker_path)
            if (old['identity'] != marker['identity'] or old['epochs'] != marker['epochs']
                    or old['encoder_sha256'] != marker['encoder_sha256']):
                raise ValueError('Recovered encoder differs from old probes; do not combine them')
        base.atomic_json(marker_path, marker)
    print('SSL COMPLETE:', job['name'], flush=True)


if __name__ == '__main__':
    main()
