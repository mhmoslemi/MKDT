#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 2 ]] || exit 2
    bench=$1 manifest=$2
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source /home/mmoslem3/ENV/bin/activate
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/mipa_multilayer/cache" TORCH_HOME="$bench/mipa_multilayer/cache/torch"
    export HF_HOME="$bench/mipa_multilayer/cache/hf" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u mipa_multilayer/extra_svhn_probes.py "$manifest"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F '|' '$1=="aip-yiweilu" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
folder="$bench/mipa_multilayer/batches/SVHN_cos8_pt0.2_20261030T163103Z_jBEyg6/results/SVHN/ResNet18BN/pt0.2/simclr/ssl_seed_000"
mkdir -p "$folder/extra_probes10"
exec 9>"$folder/extra_probes10/.submit.lock"
flock -n 9 || exit 2
active=$(squeue -h -u "$(id -un)" --name=svhn02_cos8_plus5probe -o '%i')
if [[ -n $active ]]; then printf 'Already queued: %s\n' "$active"; exit 0; fi
seconds=$(node - "$bench" "$folder" <<'JS'
const fs=require('fs'),p=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [bench,folder]=process.argv.slice(2),root=p.dirname(bench),read=f=>JSON.parse(fs.readFileSync(f)),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const config=read(p.join(folder,'config.json')),sig=config.signature,marker=read(p.join(folder,'ssl_complete.json'));
assert.ok(fs.existsSync(p.join(folder,'complete.json')));assert.equal(sig.source,'SVHN');assert.equal(sig.subset,.2);assert.equal(sig.generation_args.cosine_views,8);assert.equal(marker.epochs,2000);assert.equal(marker.complete,true);
assert.equal(hash(p.join(folder,'encoder.pt')),marker.checkpoint_file_sha256);
const durations=[];for(const labels of [1,5]){const seconds=[];for(let seed=0;seed<5;seed++){const result=read(p.join(folder,'evaluations',`SVHN_lbl${labels}_probe${seed}.json`));assert.equal(result.complete,true);assert.equal(result.encoder_unchanged,true);assert.equal(result.encoder_sha256,marker.encoder_sha256);assert.deepEqual(result.expected.identity,config.identity);assert.equal(result.expected.seed,seed);assert.equal(result.expected.label_percentage,labels);assert.equal(result.expected.evaluation.epochs,100);assert.ok(result.seconds>0&&Number.isFinite(result.seconds));seconds.push(result.seconds);}durations.push(Math.max(...seconds)*5);}
const seconds=Math.ceil(durations.reduce((a,b)=>a+b,0)+20+60);
const files=[...Object.keys(sig.code_hashes),'OT-SSL-DD/eval.py','Evaluate_benchamrk/digits_simclr/datasets.py','Evaluate_benchamrk/fashionmnist/dataset.py','Evaluate_benchamrk/mipa_multilayer/extra_svhn_probes.py','Evaluate_benchamrk/submit_svhn_cos8_02_extra_probes_vulcan.sh'];
const job={account:'aip-yiweilu',folder:p.relative(root,folder),data_root:'/scratch/mmoslem3/data',labels:[1,5],added_probe_seeds:[5,6,7,8,9],
 config_sha256:hash(p.join(folder,'config.json')),encoder_file_sha256:marker.checkpoint_file_sha256,code_hashes:Object.fromEntries(files.map(f=>[f,hash(p.join(root,f))])),seconds,buffer_seconds:60};
const file=p.join(folder,'extra_probes10/job.json');if(fs.existsSync(file))assert.deepEqual(read(file),job,'Probe resume inputs changed');else fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});console.log(seconds);
JS
)
printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
job=$(sbatch --parsable --job-name=svhn02_cos8_plus5probe --account=aip-yiweilu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
    --output="$bench/mipa_multilayer/logs/%x-%j.out" --error="$bench/mipa_multilayer/logs/%x-%j.err" \
    "$bench/submit_svhn_cos8_02_extra_probes_vulcan.sh" --worker "$bench" "$folder/extra_probes10/job.json")
job=${job%%;*}
[[ $job =~ ^[0-9]+$ ]] || exit 2
printf 'job\taccount\twalltime\n%s\taip-yiweilu\t%s\n' "$job" "$wall" > "$folder/extra_probes10/submitted_${job}.tsv"
printf 'Submitted %s on Lu, requested %s; five additional probes each at 1%% and 5%% labels.\n' "$job" "$wall"
