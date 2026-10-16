#!/bin/bash
#SBATCH --job-name=vsr_smoke
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:l40s:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
cd "$PROJECT_ROOT"
module --force purge
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache" XDG_CACHE_HOME="$SCRATCH/cache"

cache="$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP/targets/CIFAR10.pt"
root="$SLURM_TMPDIR/subset_refresh_smoke"
selection="$root/selections"

srun --unbuffered python3 benchmarks/benchmark_subset_refresh.py prepare \
    --source CIFAR10 --source-cache "$cache" --output "$selection" \
    --seed-start 0 --runs 1

for method in random kmeans; do
    encoder="$root/$method/encoders"
    srun --unbuffered python3 benchmarks/benchmark_subset_refresh.py pretrain \
        --source CIFAR10 --source-cache "$cache" --output "$encoder" \
        --selection-dir "$selection" --method "$method" --model ConvNet \
        --ssl-method simclr --subset-percentage 0.2 --ssl-epochs 1 \
        --seed-start 0 --runs 1
done

srun --unbuffered python3 benchmarks/benchmark_subset_refresh.py evaluate \
    --source CIFAR10 --source-cache "$cache" \
    --encoder-dir "$root/random/encoders" --target CIFAR10 \
    --target-cache "$cache" --output "$root/random/evaluation" \
    --method random --model ConvNet --ssl-method simclr \
    --subset-percentage 0.2 --ssl-epochs 1 \
    --label-percentage 1 --probe-epochs 1 --runs 1

test -s "$root/random/encoders/seed_000.pt"
test -s "$root/kmeans/encoders/seed_000.pt"
test -s "$root/random/evaluation/summary.json"
echo "SUBSET REFRESH SMOKE PASSED"
