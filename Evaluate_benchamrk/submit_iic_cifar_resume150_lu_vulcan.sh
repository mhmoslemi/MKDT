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
    exec python -u iic_init/resume_cifar150.py --output "$output"
fi
account=aip-yiweilu
if [[ ${1:-} == --wang ]]; then account=aip-boyuwang; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/iic_init"
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F '|' -v account="$account" '$1==account {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
mkdir -p "$run/runs" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
for dataset in CIFAR10 CIFAR100; do
    name="iic_${dataset}_R18_resume_e150"
    active=$(squeue -h -u "$(id -un)" --name="$name" -o '%i')
    if [[ -n $active ]]; then printf 'Already queued: %s (%s)\n' "$dataset" "$active"; continue; fi
    output="$run/runs/${dataset}_ResNet18_e150_from80_lu_vulcan"
    mkdir -p "$output"
    seconds=$(node - "$bench" "$output" "$dataset" "$account" <<'JS'
const fs=require('fs'),p=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [bench,output,dataset,account]=process.argv.slice(2),root=p.dirname(bench);
assert.ok(['aip-yiweilu','aip-boyuwang'].includes(account));
const read=f=>JSON.parse(fs.readFileSync(f)),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert.ok(fs.realpathSync(output).startsWith(p.join(bench,'iic_init/runs')+p.sep));
const folder=dataset==='CIFAR10'?'CIFAR10_ResNet18_e100_20261030T173733Z_ueNMxo_lu_l40s':'CIFAR100_ResNet18_e100_20261030T174140Z_OcY02E_lu_l40s';
const preview=dataset==='CIFAR10'?'CIFAR10_epoch080_pt0.2_mksx5ygw':'CIFAR100_epoch080_pt0.2_xgmlvv8b';
const source=p.join(bench,'iic_init/runs',folder),previous=read(p.join(source,'config.json'));
const metadata=read(p.join(bench,'iic_init/previews',preview,'indices.json'));
assert.equal(previous.dataset,dataset);assert.equal(previous.epochs,100);
assert.equal(metadata.dataset,dataset);assert.equal(metadata.epoch,80);
const checkpoint=p.join(source,'training.pt'),checkpoint_sha256=hash(checkpoint);
assert.equal(checkpoint_sha256,metadata.checkpoint_sha256,'Original checkpoint changed since epoch-80 preview');
assert.equal(hash(previous.teacher.path),previous.teacher.file_sha256);
assert.equal(hash(p.join(p.dirname(previous.teacher.path),'ssl_complete.json')),previous.teacher.marker_sha256);
const files=['networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/common.py','Evaluate_benchamrk/data.py',
 'Evaluate_benchamrk/iic_init/resume_cifar150.py','Evaluate_benchamrk/submit_iic_cifar_resume150_lu_vulcan.sh'];
const code_hashes=Object.fromEntries(files.map(f=>[f,hash(p.join(root,f))]));
for(const f of files.slice(0,4))assert.equal(code_hashes[f],previous.code_hashes[f],'Inference/augmentation code changed');
const times=fs.readFileSync(p.join(source,'training.jsonl'),'utf8').trim().split('\n').map(l=>JSON.parse(l));
assert.equal(times.at(-1).epoch,80);
const rates=times.map(row=>row.seconds).sort((a,b)=>a-b);
assert.ok(rates.length&&rates.every(n=>n>0&&Number.isFinite(n)));
const rate=rates[Math.ceil(rates.length*.95)-1],seconds=Math.ceil((rate*(150-80+3)+60)/60)*60;
const resume_from={config:p.join(source,'config.json'),config_sha256:metadata.config_sha256,
 checkpoint,checkpoint_sha256,epoch:80};
const config={...previous,account,epochs:150,code_hashes,seconds,buffer_seconds:60,
 timing_basis:'Continue saved epoch 80 to 150 using observed IIC p95 epoch times; three final-pass allowances and 60s buffer',resume_from};
const file=p.join(output,'config.json');
if(fs.existsSync(file)){
 const existing=read(file);
 try{assert.deepEqual(existing,config);}catch(error){
  assert.ok(!fs.existsSync(p.join(output,'training.pt'))&&!fs.existsSync(p.join(output,'complete.json')),
   'A continuation checkpoint already exists; refusing to change its configuration');
  const scientific=c=>{const {account,code_hashes,...rest}=c;return rest;};
  assert.deepEqual(scientific(existing),scientific(config),'Clustering/resume settings changed');
  const backup=p.join(output,'config_before_account_change.json');
  if(!fs.existsSync(backup))fs.copyFileSync(file,backup,fs.constants.COPYFILE_EXCL);
  fs.writeFileSync(file,JSON.stringify(config,null,2)+'\n');
 }
}else fs.writeFileSync(file,JSON.stringify(config,null,2)+'\n',{flag:'wx'});
if(!fs.existsSync(p.join(output,'training.jsonl')))fs.copyFileSync(p.join(source,'training.jsonl'),p.join(output,'training.jsonl'),fs.constants.COPYFILE_EXCL);
console.log(seconds);
JS
)
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=$(sbatch --parsable --job-name="$name" --account="$account" --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$bench/submit_iic_cifar_resume150_lu_vulcan.sh" --worker "$bench" "$output")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || exit 2
    printf 'job\tdataset\taccount\twalltime\toutput\n%s\t%s\t%s\t%s\t%s\n' "$job" "$dataset" "$account" "$wall" "$output" > "$output/submitted_${job}.tsv"
    printf 'Submitted %s: %s, resume 80 -> 150, %s/L40S, %s\nOutput: %s\nLog: %s/logs/%s-%s.out\n' "$job" "$dataset" "$account" "$wall" "$output" "$run" "$name" "$job"
done
