#!/bin/bash
# One job: 1% and 2% synthetic sizes, one 100-epoch head per label budget.
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 5 ]] || exit 2
    bench=$1 batch=$2 venv=$3 data=$4 expected=$5
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    script="$bench/mipa_b6000_i5000_lr0p0015/probe_tiny_snapshots.py"
    [[ $(sha256sum "$script" | awk '{print $1}') == "$expected" ]] || exit 2
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/mipa_b6000_i5000_lr0p0015/cache/xdg" TORCH_HOME="$bench/mipa_b6000_i5000_lr0p0015/cache/torch"
    export TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u "$script" "$batch" "$data"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/mipa_b6000_i5000_lr0p0015"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -r $data/tinyimagenet.pt ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
module spider python/3.11.5 >/dev/null 2>&1
mkdir -p "$run/snapshot_probes" "$run/logs"
batch=$(mktemp -d "$run/snapshot_probes/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
for subset in 1 2; do
    source="$run/stopped_results/TinyImageNet/ResNet18BN/pt$subset/simclr/ssl_seed_000"
    mkdir "$batch/pt$subset"
    # Training replaces checkpoints atomically; this link pins the current inode.
    ln "$source/training.pt" "$batch/pt$subset/training_snapshot.pt"
    cp "$source/config.json" "$batch/pt$subset/source_config.json"
done
seconds=$(node - "$run" <<'JS'
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const dir=path.join(process.argv[2],'stopped_results/TinyImageNet/ResNet18BN/pt0.2/simclr/ssl_seed_000/evaluations');
let seconds=90;
for(const labels of [1,5]){
    const times=[0,1,2,3,4,5].map(seed=>{
        const r=JSON.parse(fs.readFileSync(path.join(dir,`TinyImageNet_lbl${labels}_probe${seed}.json`)));
        assert.ok(r.complete&&r.encoder_unchanged&&r.expected.evaluation.epochs===100&&r.seconds>0);
        return r.seconds;
    });
    seconds+=2*Math.max(...times);
}
assert.ok(seconds<1800);console.log(Math.ceil(seconds));
JS
)
printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
hash=$(sha256sum "$run/probe_tiny_snapshots.py" | awk '{print $1}')
job=$(sbatch --parsable --job-name=mipa_tiny_snapshot_4probes --account=aip-boyuwang --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
    --output="$batch/probes-%j.out" --error="$batch/probes-%j.err" \
    "$bench/probe_mipa_tiny_snapshots_vulcan.sh" --worker "$bench" "$batch" "$venv" "$data" "$hash")
printf 'Job: %s\nWalltime: %s\nResults: %s\n' "$job" "$wall" "$batch"
