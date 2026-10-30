#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 2 ]] || exit 2
    bench=$1 output=$2
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source /home/mmoslem3/ENV/bin/activate
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache"
    export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf" TMPDIR="$SLURM_TMPDIR"
    cd "$bench"
    exec python -u iic_init/resume_cifar150_lu_killarney.py --output "$output"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/iic_init"
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == killarney ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F '|' '$1=="aip-yiweilu" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
mkdir -p "$run/runs" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another IIC submitter is running.' >&2; exit 2; }
for dataset in CIFAR10 CIFAR100; do
    name="iic_${dataset}_R18_resume_e150_lu"
    active=$(squeue -h -u "$(id -un)" --name="$name" -o '%i')
    if [[ -n $active ]]; then printf 'Already queued: %s (%s)\n' "$dataset" "$active"; continue; fi
    output="$run/runs/${dataset}_ResNet18_e150_lu_killarney"
    mkdir -p "$output"
    seconds=$(node - "$bench" "$output" "$dataset" <<'JS'
const fs=require('fs'),p=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [bench,output,dataset]=process.argv.slice(2),root=p.dirname(bench),runs=p.join(bench,'iic_init/runs');
const read=f=>JSON.parse(fs.readFileSync(f)),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert.ok(['CIFAR10','CIFAR100'].includes(dataset));
assert.ok(fs.realpathSync(output).startsWith(runs+p.sep));
const files=['networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/common.py','Evaluate_benchamrk/data.py',
 'Evaluate_benchamrk/iic_init/resume_cifar150_lu_killarney.py','Evaluate_benchamrk/submit_iic_cifar_resume150_lu_killarney.sh'];
const code_hashes=Object.fromEntries(files.map(f=>[f,hash(p.join(root,f))]));
const file=p.join(output,'config.json');
let config;
if(fs.existsSync(file)){
 config=read(file);
 assert.deepEqual(config.code_hashes,code_hashes,'Code changed since submission; saved checkpoint preserved');
}else{
 const candidates=[];
 for(const entry of fs.readdirSync(runs,{withFileTypes:true})){
  if(!entry.isDirectory()||!entry.name.startsWith(dataset+'_'))continue;
  const source=p.join(runs,entry.name),cfg=p.join(source,'config.json');
  if(source===output||!fs.existsSync(cfg)||!fs.existsSync(p.join(source,'training.pt'))||!fs.existsSync(p.join(source,'training.jsonl')))continue;
  const previous=read(cfg);
  if(previous.kind!=='frozen_teacher_iic'||previous.dataset!==dataset||![100,150].includes(previous.epochs))continue;
  const times=fs.readFileSync(p.join(source,'training.jsonl'),'utf8').trim().split('\n').filter(Boolean).map(l=>JSON.parse(l));
  if(!times.length)continue;
  const epoch=Math.max(...times.map(row=>row.epoch));
  assert.ok(Number.isInteger(epoch)&&epoch>0&&epoch<=previous.epochs,'Invalid source epoch: '+source);
  candidates.push({source,previous,times,epoch});
 }
 candidates.sort((a,b)=>b.epoch-a.epoch||a.source.localeCompare(b.source));
 assert.ok(candidates.length,`No saved ${dataset} IIC training.pt found under ${runs}. Sync its saved clustering checkpoint and config first; this script never starts CIFAR clustering from scratch.`);
 const {source,previous,times,epoch}=candidates[0];
 for(const f of files.slice(0,4))assert.equal(code_hashes[f],previous.code_hashes[f],'Inference/augmentation code changed: '+f);
 assert.equal(hash(previous.teacher.path),previous.teacher.file_sha256,'Original teacher changed');
 assert.equal(hash(p.join(p.dirname(previous.teacher.path),'ssl_complete.json')),previous.teacher.marker_sha256);
 const rates=times.map(row=>row.seconds).sort((a,b)=>a-b);
 assert.ok(rates.length&&rates.every(n=>n>0&&Number.isFinite(n)));
 const rate=rates[Math.ceil(rates.length*.95)-1];
 // The last logged epoch can precede its checkpoint write; allow one repeated epoch.
 const seconds=Math.ceil((rate*(150-Math.max(0,epoch-1)+3)+60)/60)*60;
 const snapshot=(name,destination)=>{
  const original=p.join(source,name),sha=hash(original),target=p.join(output,destination);
  if(!fs.existsSync(target))fs.copyFileSync(original,target,fs.constants.COPYFILE_EXCL);
  assert.equal(hash(target),sha,'Source changed while copying; saved files preserved: '+original);
  return {path:target,sha};
 };
 const originConfig=snapshot('config.json','origin_config.json'),originCheckpoint=snapshot('training.pt','origin_training.pt');
 const resume_from={source_run:source,config:originConfig.path,config_file_sha256:originConfig.sha,
  checkpoint:originCheckpoint.path,checkpoint_sha256:originCheckpoint.sha,logged_epoch:epoch};
 config={...previous,account:'aip-yiweilu',epochs:150,code_hashes,seconds,buffer_seconds:60,
  data_root:fs.realpathSync(process.env.BENCH_DATA_ROOT||previous.data_root),
  timing_basis:'Remaining epochs from observed IIC p95 timing, one checkpoint-write allowance, three final-pass allowances and 60s buffer',resume_from};
 if(!fs.existsSync(p.join(output,'training.jsonl')))fs.copyFileSync(p.join(source,'training.jsonl'),p.join(output,'training.jsonl'),fs.constants.COPYFILE_EXCL);
 process.stderr.write(`Resume ${dataset} from ${source} (last logged epoch ${epoch}); heads and Adam state will be restored.\n`);
}
assert.equal(config.account,'aip-yiweilu');assert.equal(config.dataset,dataset);assert.equal(config.epochs,150);
assert.equal(config.kind,'frozen_teacher_iic');assert.equal(config.seed,0);
assert.equal(config.batch_size,256);assert.equal(config.lr,.0001);assert.equal(config.subheads,5);
assert.equal(config.encoder_frozen,true);assert.equal(config.batchnorm_frozen,true);
assert.equal(config.feature_normalization,'none');assert.equal(config.augmentations,'reference_simclr_two_views');
assert.deepEqual(config.cluster_counts,dataset==='CIFAR10'?{main:10,overcluster:70}:{main:100,overcluster:700});
assert.equal(hash(config.teacher.path),config.teacher.file_sha256,'Teacher weights changed');
assert.equal(hash(p.join(p.dirname(config.teacher.path),'ssl_complete.json')),config.teacher.marker_sha256);
assert.equal(hash(config.resume_from.config),config.resume_from.config_file_sha256,'Origin config changed');
assert.equal(hash(config.resume_from.checkpoint),config.resume_from.checkpoint_sha256,'Origin checkpoint changed');
const required=dataset==='CIFAR10'?
 ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','batches.meta'].map(f=>p.join(config.data_root,'cifar-10-batches-py',f)):
 ['train','meta'].map(f=>p.join(config.data_root,'cifar-100-python',f));
for(const f of required)assert.ok(fs.existsSync(f)&&fs.statSync(f).isFile(),'Training data missing: '+f);
assert.ok(Number.isInteger(config.seconds)&&config.seconds>0);
if(!fs.existsSync(file))fs.writeFileSync(file,JSON.stringify(config,null,2)+'\n',{flag:'wx'});
console.log(config.seconds);
JS
)
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=$(sbatch --parsable --job-name="$name" --account=aip-yiweilu --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$bench/submit_iic_cifar_resume150_lu_killarney.sh" --worker "$bench" "$output")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || exit 2
    printf 'job\tdataset\taccount\twalltime\toutput\n%s\t%s\taip-yiweilu\t%s\t%s\n' "$job" "$dataset" "$wall" "$output" > "$output/submitted_${job}.tsv"
    printf 'Submitted %s: %s, resume to 150, Lu/L40S, %s\nOutput: %s\nLog: %s/logs/%s-%s.out\n' "$job" "$dataset" "$wall" "$output" "$run" "$name" "$job"
done
echo 'Rerun this script to continue the saved CIFAR clustering heads and Adam state. Original runs are unchanged.'
