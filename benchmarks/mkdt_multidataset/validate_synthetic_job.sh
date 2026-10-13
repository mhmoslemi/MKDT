#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

set -euo pipefail
[[ $# -eq 3 ]] || { echo "Usage: $0 <CIFAR10|CIFAR100> <ResNet18BN|VGG11> <1|2|5>" >&2; exit 2; }
source "$PROJECT_ROOT/benchmarks/mkdt_multidataset/environment.sh"

dataset=$1
generator=$2
size=$3
case "$generator" in
    ResNet18BN) model=ResNet18 ;;
    VGG11) model=VGG11 ;;
    *) echo "Invalid generator: $generator" >&2; exit 2 ;;
esac
folder="$MKDT_SYNTH_ROOT/generated/$generator/$dataset/${dataset}_${size}pct"

srun --unbuffered python3 - "$folder" "$dataset" "$model" "$size" <<'PY'
import ast
import sys
from pathlib import Path

import torch

folder = Path(sys.argv[1])
dataset, model, size = sys.argv[2], sys.argv[3], int(sys.argv[4])
expected_images = 50000 * size // 100
for name, shape in (
    ('images_5000.pt', (expected_images, 3, 32, 32)),
    ('labels_5000.pt', (expected_images, 512)),
    ('synlr_5000.pt', ()),
):
    path = folder / name
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f'{path}: expected tensor shape {shape}, got {getattr(value, "shape", None)}')
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise ValueError(f'{path}: expected a finite floating-point tensor')
hparams = ast.literal_eval((folder / 'hparams.txt').read_text())
expected = {
    'dataset': dataset,
    'model': model,
    'iters': 5000,
    'syn_steps': 40,
    'min_start_epoch': 1,
}
for key, value in expected.items():
    if hparams.get(key) != value:
        raise ValueError(f'{folder}/hparams.txt: expected {key}={value!r}, got {hparams.get(key)!r}')
print(f'VALID {dataset} {model} {size}%: {folder}', flush=True)
PY
