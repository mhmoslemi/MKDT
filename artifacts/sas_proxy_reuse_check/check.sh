#!/bin/bash
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/artifacts/sas_proxy_reuse_check/cache
python - <<'PY'
import json, sys
from pathlib import Path
import torch
root = Path('/scratch/mmoslem3/MKDT')
sys.path[:0] = [str(root/'Evaluate_benchamrk/cross_arch_ablation'), str(root/'Evaluate_benchamrk'), str(root)]
import models, train_ssl
models.install()
torch.set_num_threads(1)
for dataset in ('CIFAR10', 'CIFAR100'):
    folder = root/'Evaluate_benchamrk/cross_arch_ablation/results'/dataset/'ResNet50/full/pt100/simclr/ssl_seed_000'
    config = json.loads((folder/'config.json').read_text())
    state = torch.load(folder/'training.pt', map_location='cpu', weights_only=True)
    assert state['identity'] == config['identity']
    model = train_ssl.SSLModel('ResNet50', 32, config['signature']['protocol']['methods']['simclr'])
    model.load_state_dict(state['model'], strict=True)
    assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    model.eval()
    with torch.no_grad():
        features = model.projector(model.encoder(torch.zeros(2,3,32,32)))
    assert features.shape == (2,256) and torch.isfinite(features).all()
    print(json.dumps({'dataset':dataset,'epoch':state['epoch'],'checkpoint':str(folder/'training.pt'),'strict_load':True,'projection_shape':list(features.shape),'all_weights_finite':True}), flush=True)
    del model, state
PY
