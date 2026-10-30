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
    exec python -u iic_init/train_svhn_tiny_wang_killarney.py --output "$output"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/iic_init"
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == killarney ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
mkdir -p "$run/runs" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another IIC submitter is running.' >&2; exit 2; }
for dataset in SVHN TinyImageNet; do
    name="iic_${dataset}_R18_frozen_e200_wang"
    active=$(squeue -h -u "$(id -un)" --name="$name" -o '%i')
    if [[ -n $active ]]; then printf 'Already queued: %s (%s)\n' "$dataset" "$active"; continue; fi
    output="$run/runs/${dataset}_ResNet18_e200_wang_killarney"
    mkdir -p "$output"
    seconds=$(node - "$bench" "$output" "$dataset" <<'JS'
const fs=require('fs'),p=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [bench,output,dataset]=process.argv.slice(2),root=p.dirname(bench);
const read=f=>JSON.parse(fs.readFileSync(f)),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert.ok(fs.realpathSync(output).startsWith(p.join(bench,'iic_init/runs')+p.sep));
const seed=dataset==='SVHN'?0:1,train_count=dataset==='SVHN'?73257:100000;
const teacherFolder=dataset==='SVHN'?'digits_simclr/checkpoints/SVHN/ResNet18/full/pt100/simclr/seed_000':
 'full_ssl_checkpoints_1000ep_3seeds/TinyImageNet/ResNet18/full/pt100/simclr/seed_001';
const path=p.join(bench,teacherFolder,'encoder.pt'),markerFile=p.join(p.dirname(path),'ssl_complete.json');
const marker=read(markerFile),teacherConfig=read(p.join(p.dirname(path),'config.json'));
assert.equal(marker.complete,true);assert.equal(marker.epochs,1000);
assert.equal(teacherConfig.signature.source,dataset);assert.equal(teacherConfig.signature.model,'ResNet18');
assert.equal(teacherConfig.signature.ssl,'simclr');assert.equal(teacherConfig.identity.seed,seed);
assert.deepEqual(marker.identity,teacherConfig.identity);assert.equal(hash(path),marker.checkpoint_file_sha256);
assert.equal(teacherConfig.signature.image_count,train_count);
const requested_root=process.env.BENCH_DATA_ROOT||p.resolve(root,'../data');
assert.ok(fs.existsSync(requested_root),`Data directory missing: ${requested_root}. Set BENCH_DATA_ROOT to its location.`);
const data_root=fs.realpathSync(requested_root);
const required=p.join(data_root,dataset==='SVHN'?'SVHN/train_32x32.mat':'tinyimagenet.pt');
if(dataset==='SVHN'&&!fs.existsSync(required)){
 const flat=p.join(data_root,'train_32x32.mat');
 if(fs.existsSync(flat)&&fs.statSync(flat).isFile()){
  assert.ok(!fs.lstatSync(required,{throwIfNoEntry:false}),`Existing broken SVHN path preserved: ${required}`);
  fs.mkdirSync(p.dirname(required),{recursive:true});
  fs.symlinkSync(flat,required);
  process.stderr.write(`Using existing SVHN archive: ${required} -> ${flat}\n`);
 }
}
assert.ok(fs.existsSync(required)&&fs.statSync(required).isFile(),
 `${dataset} training data missing: ${required}`+(dataset==='SVHN'?` (also checked ${p.join(data_root,'train_32x32.mat')})`:'')+
 '. Set BENCH_DATA_ROOT to the directory containing the dataset.');
const files=['networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/common.py','Evaluate_benchamrk/data.py',
 'Evaluate_benchamrk/digits_simclr/datasets.py','Evaluate_benchamrk/iic_init/train_svhn_tiny_wang_killarney.py',
 'Evaluate_benchamrk/submit_iic_svhn_tiny_wang_killarney.sh'];
const code_hashes=Object.fromEntries(files.map(f=>[f,hash(p.join(root,f))]));
const file=p.join(output,'config.json'),previous=fs.existsSync(file)?read(file):null;
const seconds=dataset==='SVHN'?22920:70680;
const timing={approximate:true,basis:'Same limits as Vulcan jobs 2038429 and 2038430, based on measured IIC/full-SSL epoch timings',buffer_seconds:60};
const config={account:'aip-boyuwang',kind:'frozen_teacher_iic',dataset,epochs:200,batch_size:256,lr:.0001,seed:0,subheads:5,
 cluster_counts:dataset==='SVHN'?{main:10,overcluster:70}:{main:200,overcluster:1400},
 encoder_frozen:true,batchnorm_frozen:true,feature_normalization:'none',augmentations:'reference_simclr_two_views',
 schedule:'both_independent_head_families_per_full_training_pass',
 teacher:{path,seed,file_sha256:hash(path),marker_sha256:hash(markerFile),encoder_sha256:marker.encoder_sha256},
 train_count,source_fingerprint:teacherConfig.signature.source_fingerprint,data_root,protocol:teacherConfig.signature.protocol,
 code_hashes,seconds,buffer_seconds:60,timing,
 reference_loss:'https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py',
 adaptation:'Frozen pretrained encoder and RGB SimCLR views; same head-only adaptation as the CIFAR runs, not an original from-scratch IIC reproduction'};
if(previous)assert.deepEqual(previous,config,'Resume configuration changed; saved checkpoint preserved');
else fs.writeFileSync(file,JSON.stringify(config,null,2)+'\n',{flag:'wx'});
console.log(seconds);
JS
)
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    memory=6G
    [[ $dataset != TinyImageNet ]] || memory=8G
    job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem="$memory" --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$bench/submit_iic_svhn_tiny_wang_killarney.sh" --worker "$bench" "$output")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || exit 2
    printf 'job\tdataset\taccount\twalltime\toutput\n%s\t%s\taip-boyuwang\t%s\t%s\n' "$job" "$dataset" "$wall" "$output" > "$output/submitted_${job}.tsv"
    printf 'Submitted %s: %s, Wang/L40S, %s\nOutput: %s\nLog: %s/logs/%s-%s.out\n' "$job" "$dataset" "$wall" "$output" "$run" "$name" "$job"
done
echo 'Rerun this script to resume epoch checkpoints, including clustering heads and Adam state.'
