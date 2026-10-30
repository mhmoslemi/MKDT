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
    export XDG_CACHE_HOME="$bench/iic_init/cache" TORCH_HOME="$bench/iic_init/cache/torch"
    export HF_HOME="$bench/iic_init/cache/hf" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u iic_init/train.py --output "$output"
fi
[[ $# -le 1 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/iic_init"
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
mkdir -p "$run/runs" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
active=$(squeue -h -u "$(id -un)" --name=iic_CIFAR10_R18_frozen_e100 -o '%i')
if [[ -n $active ]]; then printf 'Already queued: %s\n' "$active"; exit 0; fi
output=${1:-}
if [[ -z $output ]]; then output=$(mktemp -d "$run/runs/CIFAR10_ResNet18_e100_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX"); fi
seconds=$(node - "$bench" "$output" <<'JS'
const fs=require('fs'),p=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [bench,destination]=process.argv.slice(2),root=p.dirname(bench),output=fs.realpathSync(destination),read=f=>JSON.parse(fs.readFileSync(f)),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert.ok(output.startsWith(p.join(bench,'iic_init/runs')+p.sep));
const path=p.join(bench,'full_ssl_checkpoints_1000ep_3seeds/CIFAR10/ResNet18/full/pt100/simclr/seed_000/encoder.pt');
const marker=read(p.join(p.dirname(path),'ssl_complete.json')),teacherConfig=read(p.join(p.dirname(path),'config.json'));
assert.equal(marker.complete,true);assert.equal(marker.source,'CIFAR10');assert.equal(marker.model,'ResNet18');assert.equal(marker.method,'simclr');assert.equal(marker.seed,0);
assert.equal(hash(path),marker.checkpoint_file_sha256);
const files=['networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/common.py','Evaluate_benchamrk/data.py','Evaluate_benchamrk/iic_init/train.py','Evaluate_benchamrk/submit_iic_cifar10_frozen_vulcan.sh'];
const code_hashes=Object.fromEntries(files.map(f=>[f,hash(p.join(root,f))]));
const times=fs.readFileSync(p.join(p.dirname(path),'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(l=>JSON.parse(l).seconds).sort((a,b)=>a-b);
assert.ok(times.length&&times.every(n=>n>0&&Number.isFinite(n)));
const fullEpoch=times[Math.ceil(times.length*.95)-1],seconds=Math.ceil(fullEpoch*103+60);
const config={account:'aip-boyuwang',kind:'frozen_teacher_iic',dataset:'CIFAR10',epochs:100,batch_size:256,lr:.0001,seed:0,subheads:5,cluster_counts:{main:10,overcluster:70},
 encoder_frozen:true,batchnorm_frozen:true,feature_normalization:'none',augmentations:'reference_simclr_two_views',schedule:'both_independent_head_families_per_full_training_pass',
 teacher:{path,file_sha256:hash(path),marker_sha256:hash(p.join(p.dirname(path),'ssl_complete.json')),encoder_sha256:marker.encoder_sha256},
 source_fingerprint:teacherConfig.signature.source_fingerprint,data_root:'/scratch/mmoslem3/data',protocol:teacherConfig.signature.protocol,code_hashes,
 seconds,buffer_seconds:60,timing_basis:'100 full-training p95 epochs as conservative baseline, plus three final-pass allowances and 60s buffer',
 reference_loss:'https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py',adaptation:'Frozen pretrained encoder and RGB SimCLR views; not an original from-scratch IIC reproduction'};
const file=p.join(output,'config.json');if(fs.existsSync(file))assert.deepEqual(read(file),config,'Resume configuration changed');else fs.writeFileSync(file,JSON.stringify(config,null,2)+'\n',{flag:'wx'});
console.log(seconds);
JS
)
printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
job=$(sbatch --parsable --job-name=iic_CIFAR10_R18_frozen_e100 --account=aip-boyuwang --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
    --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
    "$bench/submit_iic_cifar10_frozen_vulcan.sh" --worker "$bench" "$output")
job=${job%%;*}
[[ $job =~ ^[0-9]+$ ]] || exit 2
printf 'job\taccount\twalltime\toutput\n%s\taip-boyuwang\t%s\t%s\n' "$job" "$wall" "$output" > "$output/submitted_${job}.tsv"
printf 'Submitted %s on Wang, requested %s\nOutput: %s\nLog: %s/logs/iic_CIFAR10_R18_frozen_e100-%s.out\n' "$job" "$wall" "$output" "$run" "$job"
