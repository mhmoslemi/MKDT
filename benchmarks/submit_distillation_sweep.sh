#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SWEEP_ROOT="$PROJECT_ROOT/synthetid_ours"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/data}"
export PROJECT_ROOT SWEEP_ROOT DATA_ROOT
[[ ! -e "$SWEEP_ROOT" ]] || { echo "Refusing to replace $SWEEP_ROOT" >&2; exit 1; }
[[ -f "$DATA_ROOT/cifar-10-batches-py/data_batch_1" ]] || { echo "Missing CIFAR-10 under $DATA_ROOT" >&2; exit 1; }
[[ -f "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}" ]] || { echo "Missing Python environment" >&2; exit 1; }
cd "$PROJECT_ROOT"
srun --account=aip-yiweilu --time=0-00:10:00 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G bash -lc 'module load StdEnv/2023 python/3.11.5; python3 OT-SSL-DD/distillation_sweep.py make-jobs --root "$SWEEP_ROOT" --project-root "$PROJECT_ROOT"'
SWEEP_CODE_ROOT="$SWEEP_ROOT/code"
export SWEEP_CODE_ROOT
mkdir -p "$SWEEP_CODE_ROOT/OT-SSL-DD" "$SWEEP_CODE_ROOT/benchmarks"
cp OT-SSL-DD/mina_IF_distill_sweep.py OT-SSL-DD/distillation_sweep.py OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json "$SWEEP_CODE_ROOT/OT-SSL-DD/"
cp benchmarks/benchmark_cifar10.py "$SWEEP_CODE_ROOT/benchmarks/"
while IFS=$'\t' read -r config lr iterations networks encoder_data steps percentage eval_runs eval_ssl label_percentage ssl_epochs probe_epochs estimate limit member; do
    [[ $config != config_id ]] || continue
    member=${member%$'\r'}
    id=$(tar -xOf "$SWEEP_ROOT/jobs.tar" "$member" | sbatch --parsable --account=aip-yiweilu --export=ALL --chdir="$PROJECT_ROOT" --output="$SWEEP_ROOT/logs/%x-%j.out")
    id=${id%%;*}
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$id" "$config" >> "$SWEEP_ROOT/submitted.tsv"
    printf 'Submitted %s %s\n' "$id" "$config"
done < "$SWEEP_ROOT/configs.tsv"
printf 'Submitted 288 jobs. Results: %s\n' "$SWEEP_ROOT"
