#!/bin/bash
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/artifacts/sas_proxy_reuse_check/cache
python - <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
root = Path('/scratch/mmoslem3/MKDT')
sys.path[:0] = [str(root/'Evaluate_benchamrk/cross_arch_ablation'), str(root/'Evaluate_benchamrk'), str(root)]
import models
import common
torch.set_num_threads(1)
folder = root/'Evaluate_benchamrk/cross_arch_ablation/results/CIFAR10/ResNet50/full/pt100/simclr/ssl_seed_000'
path = folder/'encoder.pt'
marker = json.loads((folder/'ssl_complete.json').read_text())
with path.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == marker['checkpoint_file_sha256']
state = torch.load(path, map_location='cpu', weights_only=True)
assert state['epochs'] == 1000 and state['source'] == 'CIFAR10' and state['complete']
model, dimension = models.build_encoder('ResNet50',32)
model.load_state_dict(state['encoder'], strict=True)
assert common.state_hash(model) == marker['encoder_sha256']
assert all(torch.isfinite(t).all() for t in model.state_dict().values())
model.eval()
with torch.no_grad():
    features = model(torch.zeros(2,3,32,32))
assert features.shape == (2,2048) and torch.isfinite(features).all()
print(json.dumps({'dataset':state['source'],'epoch':state['epochs'],'checkpoint':str(path),'strict_load':True,'feature_shape':list(features.shape),'file_and_weights_hash_match':True,'projection_head_present':'projector' in state,'normalization':state['normalization']}))
PY
