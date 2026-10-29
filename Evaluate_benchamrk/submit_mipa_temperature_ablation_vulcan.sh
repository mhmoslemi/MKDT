#!/bin/bash
# Five separate jobs; each resumes distillation, SSL and its five saved probes.
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
    export XDG_CACHE_HOME="$bench/mipa_temperature_ablation/cache"
    export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u mipa_temperature_ablation/worker.py "$manifest" "$data"
fi
dry=0
if [[ ${1:-} == --dry-run ]]; then dry=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/mipa_temperature_ablation"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -r $data/cifar-10-batches-py/data_batch_1 && -r $data/cifar-10-batches-py/test_batch ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
available=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    awk -F '|' -v account="$account" '$1==account {found=1} END {exit !found}' <<< "$available" || exit 2
done
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -o '%200j|%i' > "$batch/queue.tsv"
node - "$bench" "$batch" <<'JS'
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const bench=process.argv[2],batch=process.argv[3],root=path.dirname(bench),here=path.join(bench,'mipa_temperature_ablation');
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=v=>{assert.ok(v.length&&v.every(x=>Number.isFinite(x)&&x>0));return v.sort((a,b)=>a-b)[Math.ceil(.95*v.length)-1];};
const reference=read(path.join(bench,'mipa_init_ablation/batches/split_random_20261028T190323Z_OlWMJ6/mipai_random_seed_001_gen.json'));
const files=Object.keys(reference.code_hashes).concat(['Evaluate_benchamrk/mipa_temperature_ablation/worker.py','Evaluate_benchamrk/submit_mipa_temperature_ablation_vulcan.sh']);
const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(root,f))]));
const teacher=reference.teacher;
assert.equal(hash(path.join(root,teacher.path)),teacher.file_sha256);
assert.equal(hash(path.join(root,path.dirname(teacher.path),'ssl_complete.json')),teacher.marker_sha256);
assert.ok(read(path.join(root,path.dirname(teacher.path),'ssl_complete.json')).complete);
const queue=new Set(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|')[0].trim()));
function rate(file){
    let previous;const rates=[];
    for(const line of fs.readFileSync(file,'utf8').split('\n')){
        const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;
        const current={time:Date.parse(m[1]+'T'+m[2]+'Z')/1000,iteration:+m[3]};
        if(previous&&current.time>previous.time&&current.iteration>previous.iteration)rates.push((current.time-previous.time)/(current.iteration-previous.iteration));
        previous=current;
    }
    return p95(rates);
}
// Approximate the new 100-image/batch-256 iteration from three measured configurations.
const rate500B128=rate(path.join(bench,'mipa_batch_ablation/logs/mipab_CIFAR10_p1_b128-1993924.out'));
const rate500B256=rate(path.join(bench,'mipa_batch_ablation/logs/mipab_CIFAR10_p1_b256-1993925.out'));
const rate100B1024=rate(path.join(bench,'mipa_init_ablation/logs/mipai_random_seed_001_gen-1995144.out'));
const realCost=(rate500B256-rate500B128)/128;
const generationRate=rate100B1024-realCost*(1024-256);
assert.ok(generationRate>0&&generationRate<=rate100B1024);
const measured=path.join(bench,'mipa_init_ablation/results/random_seed_001/simclr/ssl_seed_000');
const sslRate=p95(fs.readFileSync(path.join(measured,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(l=>JSON.parse(l).seconds));
const probeRate=p95(Array.from({length:5},(_,i)=>read(path.join(measured,`evaluations/CIFAR10_lbl1_probe${i}.json`)).seconds));
let rows='';const jobs=[];
for(const [index,temperature] of [0.05,0.1,0.2,0.4,0.8].entries()){
    const variant='tau_'+String(temperature).replace('.','p')+'_lr0p001',name='mipat_'+variant;
    if(queue.has(name)){console.log('SKIP active '+name);continue;}
    const generation_args={...reference.generation_args,batch_real:256,lr_img:.001,temperature};
    const initialization={method:'random',seed:0,kmeans_iterations:0};
    const gen=path.join(root,'OT-SSL-DD/ours_syntehtic_temperature_ablation/CIFAR10/ResNet18BN/pt0.2',variant);
    const folder=path.join(here,'results',variant,'simclr/ssl_seed_000');
    const contract={generation_args,initialization,teacher,code_hashes};
    let iteration=0,epoch=0,missing=5;
    const state=path.join(gen,'generation_state.json');
    if(fs.existsSync(state)){const s=read(state);assert.deepEqual(s.contract,contract);assert.ok(s.iteration>=0&&s.iteration<=10000);iteration=s.iteration;}
    const marker=path.join(folder,'ssl_complete.json');
    if(fs.existsSync(marker)){const m=read(marker);assert.ok(m.complete&&m.epochs===2000&&fs.existsSync(path.join(folder,'encoder.pt')));epoch=2000;}
    else if(fs.existsSync(path.join(folder,'training.pt'))&&fs.existsSync(path.join(folder,'ssl.jsonl'))){
        const lines=fs.readFileSync(path.join(folder,'ssl.jsonl'),'utf8').trim().split('\n').filter(Boolean);
        if(lines.length)epoch=Math.floor(JSON.parse(lines.at(-1)).epoch/500)*500;
    }
    for(let seed=0;seed<5;seed++){const f=path.join(folder,'evaluations',`CIFAR10_lbl1_probe${seed}.json`);if(fs.existsSync(f)){const p=read(f);assert.ok(epoch===2000&&p.complete&&p.encoder_unchanged);missing--;}}
    const summary=path.join(folder,'summary_lbl1.json');
    if(fs.existsSync(summary)){const s=read(summary);assert.ok(s.complete&&s.completed_probes===5&&missing===0&&s.ssl_epochs===2000);assert.deepEqual(s.generation_args,generation_args);console.log('SKIP complete '+name);continue;}
    const seconds=Math.ceil((10000-iteration)*generationRate+(2000-epoch)*sslRate+missing*probeRate+20+60);
    const config={name,stage:'pipeline',variants:[variant],temperature,generation_args,initialization,teacher,code_hashes,
        account:index%2?'aip-yiweilu':'aip-boyuwang',seconds,buffer_seconds:60,
        evaluation:{method:'simclr',ssl_epochs:2000,ssl_seeds:[0],probe_epochs:100,probe_seeds:[0,1,2,3,4],labels:[1],target:'CIFAR10'},
        timing:{estimated_generation_iteration_seconds:generationRate,ssl_epoch_seconds_p95:sslRate,probe_seconds_p95:probeRate,approximate:true}};
    const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify(config,null,2)+'\n',{flag:'wx'});
    rows+=[name,config.account,seconds,manifest].join('\t')+'\n';jobs.push(config);
}
fs.writeFileSync(path.join(batch,'jobs.tsv'),rows,{flag:'wx'});
fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify({jobs:jobs.length,temperatures:[.05,.1,.2,.4,.8],initialization_seed:0,generation_batch:256,generation_lr:.001,generation_iterations:10000,ssl_epochs:2000,probes_per_encoder:5,buffer_seconds:60},null,2)+'\n',{flag:'wx'});
console.log('Planned '+jobs.length+' separate complete-pipeline jobs; estimated '+Math.ceil((10000*generationRate+2000*sslRate+5*probeRate+80)/60)+' minutes each.');
JS
printf 'temperature\tjob\taccount\twalltime\n' > "$batch/submitted.tsv"
while IFS=$'\t' read -r name account seconds manifest; do
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=DRY_RUN
    if (( ! dry )); then
        job=$(sbatch --parsable --job-name="$name" --account="$account" --nodes=1 --ntasks=1 \
            --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
            --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
            "$bench/submit_mipa_temperature_ablation_vulcan.sh" --worker "$bench" "$manifest" "$venv" "$data")
        job=${job%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 2
    fi
    printf '%s\t%s\t%s\t%s\n' "$name" "$job" "$account" "$wall" >> "$batch/submitted.tsv"
    printf '%s: %s, %s, %s\n' "$name" "$job" "$account" "$wall"
done < "$batch/jobs.tsv"
printf 'Receipt: %s\n' "$batch/submitted.tsv"
