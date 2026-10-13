#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:05:00

set -euo pipefail
: "${MKDT_CODE_ROOT:?}"
cd "$MKDT_CODE_ROOT"
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
python3 - <<'PY'
import torch
from utils import get_network

for side in (32, 64):
    network = get_network('ResNet18', 3, 512, (side, side), fix_net=True)
    output = network(torch.zeros(2, 3, side, side))
    assert output.shape == (2, 512), (side, output.shape)
    features = network.features(torch.zeros(2, 3, side, side))
    assert features.shape == (2, 512, 1, 1), (side, features.shape)
print('TINY_RESNET_VALIDATION_OK')
PY
