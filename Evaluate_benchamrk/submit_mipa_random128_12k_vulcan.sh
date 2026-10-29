#!/bin/bash
# Two independent jobs: continue each existing 2% set, then SSL and main-table probes.
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
    export XDG_CACHE_HOME="$bench/mipa_random128_12k/cache/xdg" HF_HOME="$bench/mipa_random128_12k/cache/hf"
    export TORCH_HOME="$bench/mipa_random128_12k/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    python -c 'import ast,pathlib,sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())' "$bench/mipa_random128_12k/worker.py"
    exec python -u "$bench/mipa_random128_12k/worker.py" "$manifest" "$data"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/mipa_random128_12k"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
[[ $(sinfo -h -o '%G' --Node | sort -u) == *gpu:l40s:* ]] || exit 2
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F'|' '$1=="aip-boyuwang" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -o '%200j|%i' > "$batch/queue.tsv"
node - "$bench" "$batch" <<'NODE'
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [bench,batch]=process.argv.slice(2),root=path.dirname(bench),read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
const p95=a=>{assert.ok(a.length&&a.every(v=>Number.isFinite(v)&&v>0));a.sort((a,b)=>a-b);return a[Math.ceil(.95*a.length)-1];};
const jobs=[];
for(const source of ['CIFAR10','CIFAR100']){
 const name=`r12812k_${source}_p2`,out=path.join(bench,'mipa_random128_12k/results',source,'ResNet18BN/pt2/simclr/ssl_seed_000');
 if(queue.has(name)){console.log('SKIP active '+name+' '+queue.get(name));continue;}
 if(fs.existsSync(path.join(out,'complete.json'))){assert.ok(read(path.join(out,'complete.json')).complete);console.log('SKIP complete '+name);continue;}
 const parent=path.join(root,'OT-SSL-DD/ours_syntehtic_random128',source,'ResNet18BN/pt2'),state=read(path.join(parent,'generation_state.json'));
 assert.ok(state.complete&&state.iteration===6000);const contract=state.contract;
 const baseline=path.join(bench,'mipa_random128/results',source,'ResNet18BN/pt2/simclr/ssl_seed_000'),config=read(path.join(baseline,'config.json'));
 const saved=config.signature.synthetic;assert.equal(saved.iteration,6000);
 const parentFile=path.join(parent,`res_IIC_${source}_ResNet18BN_2percent_final.pt`);assert.equal(saved.path,parentFile);
 assert.deepEqual(config.signature.generation_args,contract.generation_args);
 const code_hashes={...contract.code_hashes};for(const [f,sha]of Object.entries(code_hashes))assert.equal(hash(path.join(root,f)),sha,'Original code changed: '+f);
 for(const f of ['Evaluate_benchamrk/mipa_random128_12k/worker.py','Evaluate_benchamrk/submit_mipa_random128_12k_vulcan.sh'])code_hashes[f]=hash(path.join(root,f));
 const logDir=path.join(bench,'mipa_random128/logs'),logs=fs.readdirSync(logDir).filter(f=>f.startsWith(`r128_${source}_p2_gen-`)&&f.endsWith('.out'));assert.equal(logs.length,1);
 const progress=[...fs.readFileSync(path.join(logDir,logs[0]),'utf8').matchAll(/^\[([^\]]+)\] iter = (\d+)/gm)].map(m=>({t:Date.parse(m[1].replace(' ','T')+'Z')/1000,i:+m[2]}));
 const rates=progress.slice(1).map((r,i)=>(r.t-progress[i].t)/(r.i-progress[i].i));
 const ssl=fs.readFileSync(path.join(baseline,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(JSON.parse);
 let probes=0;for(const labels of [1,5])for(let seed=0;seed<5;seed++)probes+=read(path.join(baseline,'evaluations',`${source}_lbl${labels}_probe${seed}.json`)).seconds;
 const seconds=Math.ceil(6000*p95(rates)+2000*p95(ssl.map(r=>r.seconds))+probes+80);
 const job={...contract,generation_args:{...contract.generation_args,Iteration:12000},code_hashes,source,subset:2,
  name,stage:'continue_and_evaluate',account:'aip-boyuwang',parent_checkpoint:path.relative(root,parentFile),parent_sha256:saved.file_sha256,
  seconds,buffer_seconds:60,startup_seconds:20,ssl_epochs:2000,linear_epochs:100,probe_seeds:[0,1,2,3,4],labels:[1,5]};
 const file=path.join(batch,name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});jobs.push([name,seconds,file]);
}
fs.writeFileSync(path.join(batch,'jobs.tsv'),jobs.map(j=>j.join('\t')).join('\n')+(jobs.length?'\n':''),{flag:'wx'});
console.log('New continuation/evaluation jobs: '+jobs.length);
NODE
printf 'name\tjob\taccount\twalltime\tmanifest\n' > "$batch/submitted.tsv"
while IFS=$'\t' read -r name seconds manifest; do
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$bench/submit_mipa_random128_12k_vulcan.sh" --worker "$bench" "$manifest" "$venv" "$data")
    job=${job%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 2
    printf '%s\t%s\taip-boyuwang\t%s\t%s\n' "$name" "$job" "$wall" "$manifest" >> "$batch/submitted.tsv"
    printf '%s: %s, Wang, %s\n' "$name" "$job" "$wall"
done < "$batch/jobs.tsv"
printf 'Receipt: %s\n' "$batch/submitted.tsv"
