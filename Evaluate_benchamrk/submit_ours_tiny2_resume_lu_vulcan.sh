#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    bench=$2 venv=$3 data=$4
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/partial_probe_jobs/cache/hf" XDG_CACHE_HOME="$bench/partial_probe_jobs/cache/xdg"
    export TORCH_HOME="$bench/partial_probe_jobs/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    exec 9>"$bench/../OT-SSL-DD/ours_syntehtic/TinyImageNet/ResNet18BN/pt2/.generation.lock"
    flock -n 9 || { echo 'Generation is already running.' >&2; exit 2; }
    cd "$bench"
    python -u partial_probe_jobs/resume_ours_tiny2.py "$data"
    exit
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -r $data/tinyimagenet.pt ]] || exit 2
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || exit 2
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
rg -qx 'aip-yiweilu\|?' <<< "$accounts"
module spider python/3.11.5 >/dev/null 2>&1
module spider nodejs/20.16.0 >/dev/null 2>&1
if ! command -v node >/dev/null; then module load StdEnv/2023 nodejs/20.16.0; fi
run="$bench/partial_probe_jobs"
mkdir -p "$run/logs"
exec 8>"$run/.ours_tiny2_submit.lock"
flock -n 8 || exit 2
name=oursgen_TinyImageNet_ResNet18BN_p2
job=$(squeue -h -u "$(id -un)" -t PD,R,CF,CG -o '%200j|%i' | awk -F '|' -v name="$name" '{gsub(/^[ \t]+|[ \t]+$/, "", $1); if($1==name)print $2}')
if [[ -z $job ]]; then
    seconds=$(node - "$bench/../OT-SSL-DD" <<'JS'
const fs=require('fs'),cp=require('child_process'),p=require('path'),assert=require('assert/strict');
const root=process.argv[2],file=p.join(root,'ours_syntehtic/TinyImageNet/ResNet18BN/pt2/res_IIC_TinyImageNet_ResNet18BN_2percent_final.pt');
const b=cp.execFileSync('unzip',['-p',file,'*/data.pkl'],{maxBuffer:8*1024*1024});
let i=b.indexOf(Buffer.from('iteration'))+9;assert.ok(i>9);if(b[i]===0x71)i+=2;else if(b[i]===0x72)i+=5;else if(b[i]===0x94)i++;
const iteration=b[i]===0x4d?b.readUInt16LE(i+1):b[i]===0x4a?b.readInt32LE(i+1):b[i]===0x4b?b[i+1]:NaN;
assert.ok(iteration>0&&iteration<=6000);
const rows=[...fs.readFileSync(p.join(root,'ours_syntehtic/logs/TinyImageNet_ResNet18BN_pt2.log'),'utf8').matchAll(/\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] iter = (\d+)/g)].map(m=>({t:Date.parse(m[1].replace(' ','T')+'Z'),i:Number(m[2])}));
const rates=rows.slice(1).flatMap((r,j)=>{const prev=rows[j],dt=(r.t-prev.t)/1000,di=r.i-prev.i;return di===25&&dt>0&&dt<600?[dt/di]:[];}).slice(-200).sort((a,b)=>a-b);
assert.ok(rates.length);console.log(Math.ceil((6000-iteration)*rates[Math.ceil(rates.length*.95)-1]+60));
JS
    )
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=$(sbatch --parsable --job-name="$name" --account=aip-yiweilu --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$walltime" --chdir="$bench" \
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$bench/submit_ours_tiny2_resume_lu_vulcan.sh" --worker "$bench" "$venv" "$data")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || exit 2
    echo "Submitted generation resume: $job ($walltime; Lu, 1 L40S, 1 CPU, 6G)"
else
    echo "Generation already queued/running: $job"
fi
export TABLE_ACCOUNT=aip-yiweilu TABLE_GROUP=ours_TinyImageNet_p2_simclr
exec bash "$bench/submit_remaining_tables_half_vulcan.sh"
