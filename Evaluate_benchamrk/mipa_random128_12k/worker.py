"""Fork the completed 2% MIPA state, continue to 12k, then reuse its evaluation."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
spec = importlib.util.spec_from_file_location('mipa_6k_pipeline', BENCH / 'mipa_random128/worker.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.HERE = HERE
base.SYNTHETICS = ROOT / 'OT-SSL-DD/ours_syntehtic_random128_12k'
base.ITERATIONS[2] = 12000


def fork(job, common):
    parent = ROOT / job['parent_checkpoint']
    assert common.file_hash(parent) == job['parent_sha256'], 'Original 6k checkpoint changed'
    original = base.read(parent.parent / 'generation_state.json')
    assert original['complete'] and original['iteration'] == 6000
    assert original['contract']['generation_args'] == dict(job['generation_args'], Iteration=6000)
    for field in ('selection', 'teacher'):
        assert original['contract'][field] == job[field]
    destination = base.synthetic_path(job['source'], 2)
    origin = dict(parent_checkpoint=str(parent), parent_sha256=job['parent_sha256'],
        from_iteration=6000, target_iteration=12000,
        preserved='pixels, initial pixels, Adam, histories, sampling order, all RNG states',
        scheduler='constant lr 0.001; extend T_max only')
    with common.exclusive_run(destination.parent):
        record = destination.parent / 'extension_origin.json'
        if record.exists():
            assert base.read(record) == origin, 'Different extension origin'
        else:
            assert not destination.exists(), 'Refusing to overwrite an unrecognized checkpoint'
            common.atomic_json(record, origin)
        if destination.exists():
            return
        saved = base.torch.load(parent, map_location='cpu', weights_only=True)
        assert saved['complete'] and saved['iteration'] == 6000
        assert saved['run_args'] == dict(job['generation_args'], Iteration=6000)
        assert saved['initialization'] == job['selection']
        assert saved['data'].shape == saved['initial_data'].shape == (1000, 3, 32, 32)
        assert len(saved['loss_history']) == 6000
        assert saved['scheduler']['T_max'] == 6000
        assert saved['scheduler']['eta_min'] == 0.001
        assert all(lr == 0.001 for lr in saved['scheduler']['base_lrs'])
        assert all(group['lr'] == 0.001 for group in saved['optimizer']['param_groups'])
        for field in ('rng', 'numpy_rng', 'objective_rng', 'data_rng', 'real_order', 'real_position'):
            assert field in saved
        extended = dict(saved, complete=False, run_args=job['generation_args'],
                        scheduler=dict(saved['scheduler'], T_max=12000))
        # Everything except the declared budget/completion metadata remains the same object.
        assert all(extended[key] is value for key, value in saved.items()
                   if key not in ('complete', 'run_args', 'scheduler'))
        first_png = 'vis_IIC_%s_ResNet18BN_2percent_iter1.png' % job['source']
        shutil.copy2(parent.parent / first_png, destination.parent / first_png)
        shutil.copy2(parent.parent / 'initialization.json', destination.parent / 'initialization.json')
        common.atomic_save(destination, extended)
        common.atomic_json(destination.parent / 'generation_state.json', dict(
            complete=False, iteration=6000, contract=base.contract(job)))
        print('FORKED complete state at 6000 -> 12000; original retained:', parent, flush=True)


def main():
    manifest, data_root = (Path(p).resolve() for p in sys.argv[1:])
    job = json.loads(manifest.read_text())
    assert job['source'] in ('CIFAR10', 'CIFAR100') and job['subset'] == 2
    assert job['generation_args'] == base.generation_args(job['source'], 2)
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
    assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    assert base.torch.cuda.is_available() and base.torch.cuda.device_count() == 1
    base.torch.set_num_threads(1)
    base.torch.set_num_interop_threads(1)
    reference = base.reference_eval.reference_modules()
    common = reference['common']
    for file, sha in job['code_hashes'].items():
        assert common.file_hash(ROOT / file) == sha, 'Implementation changed: ' + file
    for field, file in [('file_sha256', ROOT / job['teacher']['path']),
                        ('marker_sha256', (ROOT / job['teacher']['path']).parent / 'ssl_complete.json')]:
        assert common.file_hash(file) == job['teacher'][field]
    job['data_root'] = str(data_root)
    fork(job, common)
    base.generation(job, data_root, common)
    base.torch.backends.cuda.matmul.allow_tf32 = False
    base.torch.backends.cudnn.allow_tf32 = False
    base.ssl_and_main(job, reference)
    assert common.file_hash(ROOT / job['parent_checkpoint']) == job['parent_sha256']
    print('DONE: 12k MIPA + 2k SimCLR + five probes per label budget; original 6k unchanged.', flush=True)


if __name__ == '__main__':
    main()
