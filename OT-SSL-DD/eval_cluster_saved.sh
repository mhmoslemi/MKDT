#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

run_dir="$PWD/result_CRMMD2_fresh_20261022T024152Z_dsa_"

python -u - "$run_dir" <<'PY' 2>&1 | tee -a "$run_dir/eval_saved.log"
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

from eval import ROOT, evaluate_synset_SSL, reference_modules

run_dir = Path(sys.argv[1])
checkpoint = run_dir / 'res_CRMMD_CIFAR10_ResNet18BN_100images_full_exp0.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
args = argparse.Namespace(
    dataset=saved['dataset'], model=saved['model'], percentage=saved['percentage'],
    label_percentage=1, ssl_method='simclr', epoch_eval_train=2000,
    epoch_linear_train=100, ssl_log_every=100, lr_linear=0.1,
    batch_linear=256, runs=1, seed=saved['seed'], device='cuda',
    data_path=str(ROOT.parent / 'data'),
)

stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
output = run_dir / 'evaluation' / 'full' / ('exp_000_%s' % stamp)
output.mkdir(parents=True, exist_ok=False)
common = reference_modules()['common']
config = dict(
    vars(args), experiment='full', generation_experiment=0, std_ddof=1,
    synthetic_checkpoint=str(checkpoint.resolve()), save_path=str(output.resolve()),
    Iteration=saved['iteration'], original_requested_iterations=8000,
    stopped_early=saved['iteration'] < 8000,
)
common.atomic_json(output / 'config.json', config)
print('Evaluating saved iteration %d: %s' % (saved['iteration'], checkpoint), flush=True)

_, _, accuracy = evaluate_synset_SSL(
    0, args.model, saved['data'], args, output / ('seed_%03d' % args.seed),
    saved['normalization'], seed=args.seed,
)
summary = dict(
    config=config, completed_runs=1, expected_runs=1, complete=True,
    accuracies_percent=[accuracy * 100], mean_percent=accuracy * 100, std_percent=None,
)
common.atomic_json(output / 'summary.json', summary)
comparison = run_dir / ('summary_CRMMD_%s_%s_%dimages.json' % (
    args.dataset, args.model, saved['num_syn'],
))
common.atomic_json(comparison, dict(results=[summary], complete=True))
print('full seed=%d accuracy=%.2f%%' % (args.seed, accuracy * 100), flush=True)
print('Saved evaluation: %s' % output, flush=True)
print('Saved summary: %s' % comparison, flush=True)
PY
