"""Shared run identities; no changes to previous benchmark recipes or outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
UPSTREAM = HERE / 'upstream'
COMMIT = '37e1372446ab35c03fcf77302539bd4a5bee47eb'
sys.path.insert(0, str(BENCH))
from common import atomic_json, atomic_save, digest_json, exclusive_run, file_hash, state_hash


def read(path):
    return json.loads(Path(path).read_text())


def load_unit(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE / 'units'):
        raise ValueError('Unexpected unit file')
    unit = read(path)
    if (unit['source'] not in ('CIFAR10', 'CIFAR100') or unit['kind'] not in ('full', 'random', 'mkdt', 'ours')
            or unit['subset'] not in ((100,) if unit['kind'] == 'full' else (0.2, 1))
            or unit['epochs'] != (1000 if unit['kind'] == 'full' else 10000)):
        raise ValueError('Not a requested ResNet50 A2MIM unit')
    if digest_json(unit['signature']) != unit['identity']['signature']:
        raise ValueError('Unit identity is inconsistent')
    if any(unit[key] != unit['signature'][key] for key in ('source', 'kind', 'subset', 'epochs', 'input', 'code_hashes')):
        raise ValueError('Unit does not match its signed recipe')
    for relative, expected in unit['code_hashes'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed after submission: ' + relative)
    if subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Wrong upstream commit')
    subprocess.run(['git', '-C', str(UPSTREAM), 'diff', '--quiet', 'HEAD', '--'], check=True)
    return unit


def folder(unit):
    return HERE / 'checkpoints' / unit['source'] / 'ResNet50ImageNet' / unit['kind'] / ('pt%g' % unit['subset']) / 'a2mim/ssl_seed_000'


def allocated():
    import torch
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') not in ('aip-boyuwang', 'aip-yiweilu')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Run using the supplied single-GPU Vulcan launcher')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
