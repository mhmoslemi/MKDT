#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 4 ]] || exit 2
    bench=$1 output=$2 worker=$3 venv=$4
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module spider python/3.11.5 >/dev/null 2>&1
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/iic_init/cache" TORCH_HOME="$bench/iic_init/cache/torch"
    export HF_HOME="$bench/iic_init/cache/hf" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u "$worker" --output "$output"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/iic_init"
venv=${BENCH_VENV:-$HOME/ENV}
[[ -r $venv/bin/activate ]] || { echo "Missing environment: $venv" >&2; exit 2; }
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == killarney ]] || {
    echo 'Run this submitter on Killarney.' >&2; exit 2;
}
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory =~ gpu:h100: ]] || { echo 'Full H100 GRES not found on Killarney.' >&2; exit 2; }
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
mkdir -p "$run/runs" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another IIC submitter is running.' >&2; exit 2; }
for dataset in CIFAR10 CIFAR100; do
    if [[ $dataset == CIFAR10 ]]; then
        folder=CIFAR10_ResNet18_e100_20261030T173733Z_ueNMxo
        worker=iic_init/train.py
    else
        folder=CIFAR100_ResNet18_e100_20261030T174140Z_OcY02E
        worker=iic_init/train_cifar100.py
    fi
    name="iic_${dataset}_R18_frozen_e100"
    active=$(squeue -h -u "$(id -un)" --name="$name" -o '%i')
    if [[ -n $active ]]; then printf 'Already queued: %s (%s)\n' "$dataset" "$active"; continue; fi
    output="$run/runs/$folder"
    mkdir -p "$output"
    seconds=$(node - "$bench" "$output" "$dataset" "$worker" <<'JS'
const fs=require('fs'),p=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [bench,output,dataset,worker]=process.argv.slice(2),root=p.dirname(bench);
const read=f=>JSON.parse(fs.readFileSync(f)),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert.ok(fs.realpathSync(output).startsWith(p.join(bench,'iic_init/runs')+p.sep));
const path=p.join(bench,'full_ssl_checkpoints_1000ep_3seeds',dataset,'ResNet18/full/pt100/simclr/seed_000/encoder.pt');
const markerFile=p.join(p.dirname(path),'ssl_complete.json'),marker=read(markerFile),teacherConfig=read(p.join(p.dirname(path),'config.json'));
assert.equal(marker.complete,true);assert.equal(marker.source,dataset);assert.equal(marker.model,'ResNet18');assert.equal(marker.method,'simclr');assert.equal(marker.seed,0);
assert.equal(hash(path),marker.checkpoint_file_sha256);
const data_root=p.resolve(root,'../data');
const dataFile=dataset==='CIFAR10'?'cifar-10-batches-py/data_batch_1':'cifar-100-python/train';
assert.ok(fs.existsSync(p.join(data_root,dataFile)),'Training dataset is not staged');
const files=['networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/common.py','Evaluate_benchamrk/data.py',
 'Evaluate_benchamrk/'+worker,'Evaluate_benchamrk/submit_iic_both_wang_h100_killarney.sh'];
const code_hashes=Object.fromEntries(files.map(f=>[f,hash(p.join(root,f))]));
const times=fs.readFileSync(p.join(p.dirname(path),'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(l=>JSON.parse(l).seconds).sort((a,b)=>a-b);
assert.ok(times.length&&times.every(n=>n>0&&Number.isFinite(n)));
const seconds=Math.ceil(times[Math.ceil(times.length*.95)-1]*103+60);
const config={account:'aip-boyuwang',kind:'frozen_teacher_iic',dataset,epochs:100,batch_size:256,lr:.0001,seed:0,subheads:5,
 cluster_counts:dataset==='CIFAR10'?{main:10,overcluster:70}:{main:100,overcluster:700},
 encoder_frozen:true,batchnorm_frozen:true,feature_normalization:'none',augmentations:'reference_simclr_two_views',schedule:'both_independent_head_families_per_full_training_pass',
 teacher:{path,file_sha256:hash(path),marker_sha256:hash(markerFile),encoder_sha256:marker.encoder_sha256},
 source_fingerprint:teacherConfig.signature.source_fingerprint,data_root,protocol:teacherConfig.signature.protocol,code_hashes,
 seconds,buffer_seconds:60,timing_basis:'100 full-training p95 epochs as conservative baseline, plus three final-pass allowances and 60s buffer',
 reference_loss:'https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py',adaptation:'Frozen pretrained encoder and RGB SimCLR views; not an original from-scratch IIC reproduction'};
const file=p.join(output,'config.json');
if(fs.existsSync(file)){
 const previous=read(file);
 try{assert.deepEqual(previous,config);}catch(error){
  assert.ok(!fs.existsSync(p.join(output,'training.pt'))&&!fs.existsSync(p.join(output,'complete.json')),
   'Existing trained run configuration changed; refusing to invalidate its checkpoint');
  const scientific=c=>{const {account,code_hashes,...rest}=c;return rest;};
  assert.deepEqual(scientific(previous),scientific(config),'Scientific recipe changed');
  const backup=p.join(output,'config_before_killarney.json');
  if(!fs.existsSync(backup))fs.copyFileSync(file,backup,fs.constants.COPYFILE_EXCL);
  fs.writeFileSync(file,JSON.stringify(config,null,2)+'\n');
 }
}else fs.writeFileSync(file,JSON.stringify(config,null,2)+'\n',{flag:'wx'});
console.log(seconds);
JS
)
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:h100:1 --time="$wall" --chdir="$bench" \
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$bench/submit_iic_both_wang_h100_killarney.sh" --worker "$bench" "$output" "$worker" "$venv")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || exit 2
    printf 'job\taccount\tgpu\twalltime\toutput\n%s\taip-boyuwang\th100\t%s\t%s\n' "$job" "$wall" "$output" > "$output/submitted_${job}.tsv"
    printf 'Submitted %s: %s, Wang/H100, %s\nOutput: %s\nLog: %s/logs/%s-%s.out\n' "$job" "$dataset" "$wall" "$output" "$run" "$name" "$job"
done
echo 'Rerun this same script to resume saved epochs and Adam state; teacher weights remain frozen.'
