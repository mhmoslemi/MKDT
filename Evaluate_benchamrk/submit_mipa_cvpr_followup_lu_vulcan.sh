#!/bin/bash
set -euo pipefail
bench=/scratch/mmoslem3/MKDT/Evaluate_benchamrk
run="$bench/mipa_cvpr_followup"
if [[ ${1:-} == --worker ]]; then
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source /home/mmoslem3/ENV/bin/activate
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$run/cache/xdg" TORCH_HOME="$run/cache/torch" HF_HOME="$run/cache/hf"
    export TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u "$run/worker.py" "$2"
fi
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || exit 2
dry=0
[[ ${1:-} != --dry-run ]] || dry=1
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
[[ $(sinfo -h -o '%G' --Node | sort -u) == *gpu:l40s:* ]] || exit 2
accounts=$(sacctmgr -nP show assoc user=mmoslem3 format=Account)
awk -F '|' '$1=="aip-yiweilu" {found=1} END {exit !found}' <<< "$accounts"
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
mkdir -p "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
plan=$(node - "$run" <<'JS'
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const run=process.argv[2],bench=path.dirname(run),root=path.dirname(bench);
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const {epochHeader}=require(path.join(bench,'cross_arch_ablation/plan'));
const plan=read(path.join(run,'manifest.json'));
assert.equal(plan.jobs.length,36);
assert.equal(plan.shared.account,'aip-yiweilu');assert.equal(plan.shared.gpu,'l40s');
for(const [file,sha]of Object.entries(plan.shared.code_hashes))assert.equal(hash(path.join(root,file)),sha,'Code changed: '+file);
for(const set of Object.values(plan.sets)){
    assert.equal(hash(set.synthetic_file),set.synthetic_sha256);
    assert.equal(hash(set.main_config),set.main_config_sha256);
    assert.equal(hash(path.join(set.encoder_dir,'encoder.pt')),set.encoder_sha256);
}
for(const task of plan.jobs){
    const recipe=plan.recipes[task.recipe]||{},job={...plan.shared,...plan.sets[task.set],...recipe,...task};
    const marker=path.join(job.output_dir,job.stage==='downstream'?'complete_'+job.name+'.json':'complete.json');
    if(fs.existsSync(marker)&&read(marker).complete)continue;
    let epoch=0;
    if(job.stage!=='downstream'){
        if(fs.existsSync(path.join(job.output_dir,'ssl_complete.json')))epoch=job.ssl_epochs;
        else if(fs.existsSync(path.join(job.output_dir,'training.pt')))epoch=epochHeader(path.join(job.output_dir,'training.pt'),read(path.join(job.output_dir,'config.json')).identity);
    }
    assert(epoch>=0&&epoch<=job.ssl_epochs);
    const seconds=Math.ceil(((job.stage==='downstream'?0:job.ssl_epochs-epoch)*(job.epoch_seconds||0)+job.probe_seconds)*1.25+60);
    process.stdout.write([job.name,job.account,job.memory,seconds].join('\t')+'\n');
}
JS
)
declare -A active=()
while IFS='|' read -r name id; do [[ -z $name ]] || active["$name"]=$id; done < <(squeue -h -u mmoslem3 -o '%j|%i')
while IFS=$'\t' read -r name account memory seconds; do
    [[ -n $name ]] || continue
    if [[ -n ${active[$name]:-} ]]; then echo "ACTIVE $name: ${active[$name]}"; continue; fi
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=DRY_RUN
    if (( ! dry )); then
        job=$(sbatch --parsable --job-name="$name" --account="$account" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem="$memory" --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" "$bench/submit_mipa_cvpr_followup_lu_vulcan.sh" --worker "$name")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || exit 2
        printf '%s\t%s\t%s\t%s\n' "$name" "$job" "$account" "$wall" >> "$run/submitted.tsv"
    fi
    active[$name]=$job
    echo "$name: $job, $account, $wall"
done <<< "$plan"
echo "Receipt: $run/submitted.tsv"
