#!/bin/bash
# A separate run; keep the original method and all existing experiments unchanged.
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 4 ]] || exit 2
    bench=$1 manifest=$2 venv=$3 data=$4
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/mipa_cifar10_p2_b6000_i2000_lr0p0025_disp1/cache"
    export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u mipa_cifar10_p2_b6000_i2000_lr0p0025_disp1/worker.py "$manifest" "$data"
fi
dry=0
if [[ ${1:-} == --dry-run ]]; then dry=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/mipa_cifar10_p2_b6000_i2000_lr0p0025_disp1"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -r $data/cifar-10-batches-py/data_batch_1 && -r $data/cifar-10-batches-py/test_batch ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
module spider nodejs/20.16.0 >/dev/null 2>&1
if ! command -v node >/dev/null; then module load StdEnv/2023 nodejs/20.16.0; fi
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
name=mipa_d1_CIFAR10_p2
existing=$(squeue -h -u "$(id -un)" -o '%200j|%i' | awk -F '|' -v name="$name" '{gsub(/ /,"",$1); if($1==name)print $2}')
if [[ -n $existing ]]; then echo "Already queued/running: $existing"; exit 0; fi
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
node - "$bench" "$batch" <<'JS'
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const bench=process.argv[2],batch=process.argv[3],root=path.dirname(bench),run=path.dirname(path.dirname(batch));
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const teacher=path.join(bench,'full_ssl_checkpoints_1000ep_3seeds/CIFAR10/ResNet18/full/pt100/simclr/seed_000/encoder.pt');
assert.ok(read(path.join(path.dirname(teacher),'ssl_complete.json')).complete);
const files=fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f).concat(['networks.py','benchmarks/downstream_data.py',
    'Evaluate_benchamrk/protocol.json','OT-SSL-DD/method1.py','OT-SSL-DD/eval.py','OT-SSL-DD/dm_dsa.py','OT-SSL-DD/dd_teachers.py','OT-SSL-DD/utils.py','OT-SSL-DD/networks.py',
    'Evaluate_benchamrk/mipa_cifar10_p2_b6000_i2000_lr0p0025_disp1/worker.py','Evaluate_benchamrk/submit_mipa_cifar10_p2_lr0p0025_disp1_vulcan.sh']);
const generation_args={dataset:'CIFAR10',percentage:2,model:'ResNet18BN',seed:0,Iteration:2000,lr_img:.0025,batch_real:6000,
    temperature:.2,disp_weight:1,pair_weight:0,num_random_networks:1,random_models:'ResNet18BN',dsa:true,
    distill_aug_strategy:'color_crop_cutout_flip_scale_rotate',distill_aug_mode:'M',teacher_source:'pretrained',teacher_seed:[0],teacher_ssl:'simclr',
    teacher_path:path.join(bench,'full_ssl_checkpoints_1000ep_3seeds'),feature_batch:128};
assert.equal(50000*generation_args.percentage/100,1000);assert.equal(generation_args.Iteration,2000);
assert.equal(generation_args.lr_img,.0025);assert.equal(generation_args.disp_weight,1);assert.equal(generation_args.pair_weight,0);
const timing=read(path.join(bench,'mipa_b6000_i5000_lr0p0015/batches/batch_20261028T003926Z_OjF5mJ/coverage.json')).timings.find(t=>t.source==='CIFAR10');
const logs=path.join(bench,'mipa_b6000_i5000_lr0p0015/logs');let rates=[];
for(const file of ['mipa6k_CIFAR10_p2_g1000-1980004.out','mipa6k_CIFAR10_p2_g2000-1980005.out']){
    let previous;for(const line of fs.readFileSync(path.join(logs,file),'utf8').split('\n')){
        const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;
        const current={time:Date.parse(m[1]+'T'+m[2]+'Z')/1000,it:+m[3]};
        if(previous&&current.it>previous.it)rates.push((current.time-previous.time)/(current.it-previous.it));previous=current;
    }
}
assert.ok(rates.length>0&&rates.every(v=>Number.isFinite(v)&&v>0));rates.sort((a,b)=>a-b);
const generation_rate=rates[Math.ceil(.95*rates.length)-1];
const recentSSL=path.join(bench,'mipa_b6000_i5000_lr0p0015/results/CIFAR100/ResNet18BN/pt2/simclr/ssl_seed_000/ssl.jsonl');
const sslRates=fs.readFileSync(recentSSL,'utf8').trim().split('\n').slice(-200).map(l=>JSON.parse(l).seconds).sort((a,b)=>a-b);
assert.ok(sslRates.length===200&&sslRates.every(v=>Number.isFinite(v)&&v>0));const ssl_rate=Math.max(timing.ssl_epoch_seconds_p95,sslRates[Math.ceil(.95*sslRates.length)-1]);
const seconds=Math.ceil(21+generation_rate*2000+ssl_rate*2000+6*(timing.probe_seconds_p95[1]+timing.probe_seconds_p95[5])+60);
assert.ok(seconds>3600&&seconds<2*3600);
const synthetic_directory='OT-SSL-DD/ours_syntehtic_b6000_i2000_lr0p0025_disp1/CIFAR10/ResNet18BN/pt2';
assert.ok(synthetic_directory!== 'OT-SSL-DD/ours_syntehtic/CIFAR10/ResNet18BN/pt2');
const config={account:'aip-boyuwang',generation_args,synthetic_directory,
    teacher:{path:path.relative(root,teacher),file_sha256:hash(teacher),marker_sha256:hash(path.join(path.dirname(teacher),'ssl_complete.json'))},
    code_hashes:Object.fromEntries(files.map(f=>[f,hash(path.join(root,f))])),
    evaluation:{ssl:'simclr',epochs:2000,ssl_seeds:[0],probe_epochs:100,probe_seeds:[0,1,2,3,4,5],labels:[1,5]},
    timing:{seconds,generation_iteration_seconds_p95:generation_rate,ssl_epoch_seconds_p95:ssl_rate,buffer_seconds:60}};
fs.writeFileSync(path.join(batch,'job.json'),JSON.stringify(config,null,2)+'\n',{flag:'wx'});
fs.writeFileSync(path.join(batch,'seconds'),String(seconds)+'\n',{flag:'wx'});
console.log(JSON.stringify({account:config.account,generation_args,synthetic_directory,evaluation:config.evaluation,timing:config.timing},null,2));
console.log('PASS: separate 1000-image synthetic set; one 2k SimCLR encoder; 12 frozen probes; original experiments untouched.');
JS
seconds=$(<"$batch/seconds")
printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
job=DRY_RUN
if (( ! dry )); then
    job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$bench/submit_mipa_cifar10_p2_lr0p0025_disp1_vulcan.sh" --worker "$bench" "$batch/job.json" "$venv" "$data")
    job=${job%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 2
fi
printf 'job\taccount\twalltime\n%s\taip-boyuwang\t%s\n' "$job" "$wall" > "$batch/submitted.tsv"
printf 'Job: %s; account: Wang; walltime: %s\nReceipt: %s\n' "$job" "$wall" "$batch/submitted.tsv"
