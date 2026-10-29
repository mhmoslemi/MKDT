"""Probe the explicitly accepted 1%/2% TinyImageNet snapshots, without more SSL."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))
import torch
from common import atomic_json, atomic_save, build_encoder, digest_json, exclusive_run, file_hash, state_hash
from data import RawDataset
from evaluate import linear_probe


def read(path):
    return json.loads(Path(path).read_text())


def main():
    batch, data_root = (Path(value).resolve() for value in sys.argv[1:])
    if (not batch.is_relative_to(HERE / 'snapshot_probes') or not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-boyuwang'
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the supplied single-GPU Wang snapshot-probe job')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with exclusive_run(batch):
        train_set = RawDataset(data_root, 'TinyImageNet', True)
        test_set = RawDataset(data_root, 'TinyImageNet', False)
        cells = []
        for subset in (1, 2):
            output = batch / ('pt%d' % subset)
            config = read(output / 'source_config.json')
            signature = config['signature']
            if (digest_json(signature) != config['identity']['signature'] or signature['source'] != 'TinyImageNet'
                    or signature['model'] != 'ResNet18' or signature['subset'] != subset
                    or signature['ssl'] != 'simclr' or signature['source_fingerprint'] != train_set.fingerprint):
                raise ValueError('Wrong original TinyImageNet SSL configuration')
            for relative, expected in signature['code_hashes'].items():
                if file_hash(ROOT / relative) != expected:
                    raise ValueError('Encoder/probe implementation changed: ' + relative)
            protocol = signature['protocol']
            assert protocol['evaluation']['epochs'] == 100 and protocol['ssl']['subset_epochs'] == 2000
            snapshot = output / 'training_snapshot.pt'
            saved = torch.load(snapshot, map_location='cpu', weights_only=True)
            if (saved['identity'] != config['identity'] or not 0 < saved['epoch'] <= 2000
                    or saved['steps'] != saved['epoch'] * saved['steps_per_epoch']):
                raise ValueError('Snapshot identity or saved epoch is invalid')
            encoder, _ = build_encoder('ResNet18', 64)
            weights = {key.removeprefix('encoder.'): value for key, value in saved['model'].items() if key.startswith('encoder.')}
            encoder.load_state_dict(weights, strict=True)
            assert all(torch.isfinite(value).all() for value in weights.values())
            encoder.eval().requires_grad_(False)
            snapshot_info = dict(file_sha256=file_hash(snapshot), ssl_epochs=saved['epoch'], steps=saved['steps'],
                                 requested_ssl_epochs=2000, original_identity=config['identity'])
            identity = dict(signature=digest_json(dict(experiment='mipa_tiny_snapshot_1probe',
                snapshot=snapshot_info, protocol=protocol, subset=subset, probe_seed=0,
                worker_sha256=file_hash(__file__))), seed=0, selection_method='ours')
            checkpoint = output / 'encoder.pt'
            # Complete encoder export, not a claim that the original 2k SSL run finished.
            metadata = dict(kind='ssl_reference_encoder', complete=True, identity=identity,
                model='ResNet18', input_size=64, source='TinyImageNet', method='simclr', seed=0,
                epochs=saved['epoch'], steps=saved['steps'], encoder_sha256=state_hash(encoder),
                normalization=protocol['normalization']['TinyImageNet'],
                ssl_training_complete=saved['epoch'] == 2000, accepted_partial_ssl=saved['epoch'] < 2000,
                requested_ssl_epochs=2000, snapshot=snapshot_info)
            atomic_save(checkpoint, dict(**metadata, encoder=encoder.state_dict()))
            atomic_json(output / 'snapshot.json', metadata)
            print('SNAPSHOT: TinyImageNet %d%%, actual saved SSL epoch %d/2000' % (subset, saved['epoch']), flush=True)
            del encoder, weights, saved
            for labels in (1, 5):
                result = linear_probe(checkpoint, identity, protocol, 'TinyImageNet', 'TinyImageNet',
                    train_set, test_set, labels, 0, output / ('probe_lbl%d_seed0.json' % labels), workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == metadata['encoder_sha256']
                cell = dict(subset=subset, labels=labels, accuracy_percent=result['accuracy_percent'],
                    ssl_epochs=metadata['epochs'], requested_ssl_epochs=2000, linear_epochs=100,
                    probe_seed=0, completed_probes=1, expected_probes=1, snapshot=snapshot_info)
                atomic_json(output / ('summary_lbl%d.json' % labels), cell)
                cells.append(cell)
        atomic_json(batch / 'summary.json', dict(complete=True, cells=cells))
        print('FOUR PROBES COMPLETE:', json.dumps(cells), flush=True)


if __name__ == '__main__':
    main()
