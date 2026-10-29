"""One temperature: random-image MIPA, 2k SimCLR, then five frozen probes."""
import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path.insert(0, str(ROOT))
from Evaluate_benchamrk.mipa_init_ablation import worker as initialization

TEMPERATURES = (0.05, 0.1, 0.2, 0.4, 0.8)
BASE_ARGS = initialization.generation_args()
INIT = dict(method='random', seed=0, kmeans_iterations=0)


def generation_args(temperature):
    assert temperature in TEMPERATURES
    return dict(BASE_ARGS, batch_real=256, lr_img=0.001, temperature=temperature)


def variant_for(temperature):
    assert temperature in TEMPERATURES
    return 'tau_' + str(temperature).replace('.', 'p') + '_lr0p001'


def synthetic_path(variant):
    assert variant in {variant_for(t) for t in TEMPERATURES}
    return (ROOT / 'OT-SSL-DD/ours_syntehtic_temperature_ablation/CIFAR10/ResNet18BN/pt0.2'
            / variant / 'res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt')


def configure(temperature):
    # Reuse the checked initialization/evaluation loops without changing their files.
    variant = variant_for(temperature)
    initialization.HERE = HERE
    initialization.VARIANTS = {variant: dict(INIT)}
    initialization.generation_args = lambda: generation_args(temperature)
    initialization.synthetic_path = synthetic_path
    return variant


def main():
    manifest, data_root = map(lambda p: Path(p).resolve(), sys.argv[1:])
    if not manifest.is_relative_to(HERE / 'batches'):
        raise ValueError('Unexpected temperature-ablation manifest')
    job = json.loads(manifest.read_text())
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != job['account']
            or job['account'] not in ('aip-boyuwang', 'aip-yiweilu')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the supplied one-GPU Vulcan launcher')
    variant = configure(job['temperature'])
    assert job['stage'] == 'pipeline' and job['variants'] == [variant]
    assert job['generation_args'] == generation_args(job['temperature'])
    assert job['initialization'] == INIT
    assert job['evaluation'] == dict(method='simclr', ssl_epochs=2000, ssl_seeds=[0],
        probe_epochs=100, probe_seeds=list(range(5)), labels=[1], target='CIFAR10')
    reference = initialization.reference_eval.reference_modules()
    common = reference['common']
    for relative, expected in job['code_hashes'].items():
        if common.file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed after submission: ' + relative)
    teacher = ROOT / job['teacher']['path']
    assert common.file_hash(teacher) == job['teacher']['file_sha256']
    assert common.file_hash(teacher.parent / 'ssl_complete.json') == job['teacher']['marker_sha256']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with common.exclusive_run(HERE / 'sessions' / variant):
        initialization.generate(job, variant, data_root, common)
        initialization.evaluate(job, variant, data_root, reference)


if __name__ == '__main__':
    main()
