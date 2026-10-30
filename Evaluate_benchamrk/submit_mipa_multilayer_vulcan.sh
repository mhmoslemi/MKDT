#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 3 ]] || exit 2
    bench=$1 manifest=$2 data=$3
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source /home/mmoslem3/ENV/bin/activate
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/mipa_multilayer/cache"
    export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u mipa_multilayer/worker.py "$manifest" "$data"
fi
d3s=0
if [[ ${1:-} == --d3s ]]; then d3s=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/mipa_multilayer"
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
sacctmgr -nP show assoc "user=$(id -un)" format=Account | awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
for source in CIFAR10 FashionMNIST; do
    name="mipa_all_${source}_5k_dsa"
    if (( d3s )); then name="mipa_d3s_${source}_5k_lr0p002_reg1e4"; fi
    active=$(squeue -h -u "$(id -un)" --name="$name" -o '%i')
    if [[ -n $active ]]; then printf 'Already queued: %s %s\n' "$source" "$active"; continue; fi
    batch=$(mktemp -d "$run/batches/${source}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    seconds=$(node - "$bench" "$batch" "$source" "$d3s" <<'JS'
const fs=require('fs'),p=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [bench,batch,source,mode]=process.argv.slice(2),d3s=mode==='1',root=p.dirname(bench),read=f=>JSON.parse(fs.readFileSync(f)),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const oldFolder=p.join(bench,`mipa_similarity_ablation/results/${source}/ResNet18BN/pt0.2/mmd/simclr/ssl_seed_000`);
const old=read(p.join(oldFolder,'config.json')).signature;
const args={...old.generation_args,Iteration:5000,feature_layers:'all'};
if(d3s)Object.assign(args,{lr_img:.002,d3s_weight:1,d3s_reg:.0001});
if(source==='FashionMNIST')args.single_channel=true;
assert.equal(args.lr_img,.002);assert.equal(args.batch_real,128);assert.deepEqual(args.teacher_seed,[0]);assert.equal(args.disp_weight,0);assert.equal(args.pair_weight,0);assert.equal(args.dsa,true);
assert.equal(hash(p.join(root,old.selection.config)),old.selection.config_sha256);
assert.equal(hash(p.join(root,old.teacher.path)),old.teacher.file_sha256);
assert.equal(hash(p.join(root,p.dirname(old.teacher.path),'ssl_complete.json')),old.teacher.marker_sha256);
fs.copyFileSync(p.join(root,'OT-SSL-DD/method1.py'),p.join(batch,'method1.py'),fs.constants.COPYFILE_EXCL);
const files=[...fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),'networks.py','benchmarks/downstream_data.py',
 ...['method1.py','eval.py','utils.py','networks.py','dd_teachers.py','dm_dsa.py'].map(f=>'OT-SSL-DD/'+f),
 'Evaluate_benchamrk/protocol.json','Evaluate_benchamrk/fashionmnist/full/prepared.json','Evaluate_benchamrk/fashionmnist/dataset.py',
 'Evaluate_benchamrk/digits_simclr/datasets.py','Evaluate_benchamrk/mipa_multilayer/worker.py','Evaluate_benchamrk/submit_mipa_multilayer_vulcan.sh'];
if(d3s)files.push('Evaluate_benchamrk/mipa_multilayer/check_d3s.py');
const code_hashes=Object.fromEntries(files.map(f=>[f,hash(p.join(root,f))]));
const p95=a=>{assert.ok(a.length&&a.every(n=>n>0&&Number.isFinite(n)));return a.sort((a,b)=>a-b)[Math.ceil(a.length*.95)-1];};
const rates=[],logs=p.join(bench,'mipa_similarity_ablation/logs');
for(const file of fs.readdirSync(logs).filter(f=>f.startsWith(`mips_${source}_p0p2_mmd-`)&&f.endsWith('.out'))){let last;for(const line of fs.readFileSync(p.join(logs,file),'utf8').split('\n')){const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;const cur={t:Date.parse(m[1]+'T'+m[2]+'Z')/1000,i:+m[3]};if(last&&cur.i>last.i&&cur.t>last.t)rates.push((cur.t-last.t)/(cur.i-last.i));last=cur;}}
const sslRate=p95(fs.readFileSync(p.join(oldFolder,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(l=>JSON.parse(l).seconds));
const probes=Object.fromEntries([1,5].map(lbl=>[lbl,p95([0,1,2,3,4].map(seed=>read(p.join(oldFolder,'evaluations',`${source}_lbl${lbl}_probe${seed}.json`)).seconds))]));
const calibration=d3s?read(p.join(bench,'mipa_multilayer/d3s_timing.json')):null;
if(calibration){assert.equal(calibration.method_sha256,hash(p.join(root,'OT-SSL-DD/method1.py')));assert.equal(calibration.check_sha256,hash(p.join(bench,'mipa_multilayer/check_d3s.py')));assert.equal(calibration.complete,true);assert.equal(calibration.datasets[source].teacher_sha256,old.teacher.file_sha256);}
const generationRate=d3s?calibration.datasets[source].iteration_seconds_p95:p95(rates)*1.5;
const startup=d3s?calibration.datasets[source].pre_loop_seconds+20:20;
const seconds=Math.ceil(5000*generationRate+2000*sslRate+2*probes[1]+2*probes[5]+startup+60);
const experiment=p.basename(batch);
const job={account:'aip-boyuwang',similarity:d3s?'d3s':'cosine',generation_args:args,selection:old.selection,teacher:old.teacher,code_hashes,seconds,buffer_seconds:60,
 synthetic_path:`OT-SSL-DD/${d3s?'ours_syntehtic_d3s_5k_lr0p002_reg1e4_dsa':'ours_syntehtic_multilayer_5k_dsa'}/${experiment}/${source}/ResNet18BN/pt0.2/res_IIC_${source}_ResNet18BN_0.2percent_final.pt`,
 evaluation:{ssl_epochs:2000,ssl_seed:0,probe_epochs:100,probe_seeds:[0,1],labels:[1,5]},
 timing:{approximate:true,generation_seconds_per_iteration:generationRate,multilayer_overhead_factor:d3s?null:1.5,ssl_seconds_per_epoch:sslRate,probe_seconds:probes,startup_seconds:startup,gpu_calibration:calibration}};
fs.writeFileSync(p.join(batch,'job.json'),JSON.stringify(job,null,2)+'\n',{flag:'wx'});console.log(seconds);
JS
)
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$bench/submit_mipa_multilayer_vulcan.sh" --worker "$bench" "$batch/job.json" /scratch/mmoslem3/data)
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || exit 2
    printf 'job\tdataset\taccount\twalltime\tmanifest\n%s\t%s\taip-boyuwang\t%s\t%s\n' "$job" "$source" "$wall" "$batch/job.json" > "$batch/submitted.tsv"
    printf 'Submitted %s %s on Wang, requested %s\nReceipt: %s\n' "$job" "$source" "$wall" "$batch/submitted.tsv"
done
