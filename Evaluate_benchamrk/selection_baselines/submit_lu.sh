#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    run=$(dirname -- "$2")
else
    run=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
fi
bench=$(dirname -- "$run")
root=$(dirname -- "$bench")
if [[ ${1:-} == --worker ]]; then
    shift
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source /home/mmoslem3/ENV/bin/activate
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$run/cache" TORCH_HOME="$run/cache/torch" HF_HOME="$run/cache/hf"
    export TMPDIR="${SLURM_TMPDIR:?}"
    cd "$root"
    exec python -u "$run/worker.py" "$@"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]]
[[ $(sinfo -h -o '%G' --Node | sort -u) == *gpu:l40s:* ]]
sacctmgr -nP show assoc user=mmoslem3 format=Account | awk -F '|' '$1=="aip-yiweilu" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
for file in /home/mmoslem3/ENV/bin/activate /scratch/mmoslem3/data/cifar-10-batches-py/data_batch_1 /scratch/mmoslem3/data/cifar-100-python/train /scratch/mmoslem3/data/tinyimagenet.pt; do
    [[ -s $file ]] || { echo "Missing input: $file" >&2; exit 2; }
done
mkdir -p "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9
node - "$run" "$root" <<'JS'
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const [run,root]=process.argv.slice(2),bench=path.dirname(run);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const files=[...fs.readdirSync(bench).filter(x=>x.endsWith('.py')).map(x=>path.join(bench,x)),
  path.join(bench,'protocol.json'),path.join(root,'networks.py'),path.join(root,'benchmarks/downstream_data.py'),
  ...['worker.py','selection.py','test_selection.py','submit_lu.sh','upstream/sas/submodular_maximization.py','upstream/sas/subset_dataset.py'].map(x=>path.join(run,x))];
const code_hashes=Object.fromEntries(files.map(f=>[path.relative(root,f),hash(f)]));
const datasets={};
for(const ds of ['CIFAR10','CIFAR100','TinyImageNet']){
 const teacher_seed=ds==='TinyImageNet'?1:0;
 const folder=path.join(bench,'full_ssl_checkpoints_1000ep_3seeds',ds,'ResNet18/full/pt100/simclr',`seed_${String(teacher_seed).padStart(3,'0')}`);
 const marker=read(path.join(folder,'ssl_complete.json'));
 assert.ok(marker.complete);
 const teacher=path.relative(root,path.join(folder,'encoder.pt'));
 datasets[ds]={teacher,teacher_seed,teacher_sha256:hash(path.join(root,teacher))};
}
const manifest={experiment:'sas_probcover_main_table',account:'aip-yiweilu',data_root:'/scratch/mmoslem3/data',
 datasets,code_hashes,percentages:[0.2,1,2],methods:['sas','probcover'],ssl_epochs:2000,ssl_seeds:[0],probe_seeds:[0,1,2,3,4],labels:[1,5]};
const target=path.join(run,'manifest.json');
if(fs.existsSync(target))assert.deepEqual(read(target),manifest,'Existing experiment configuration changed');
else fs.writeFileSync(target,JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
JS
declare -A active=()
while IFS='|' read -r name job; do [[ -z $name ]] || active["$name"]=$job; done < <(squeue -h -u mmoslem3 -o '%j|%i')
receipt="$run/submitted.tsv"
[[ -s $receipt ]] || printf 'stage\tsource\tmethod\tpercentage\tjob\taccount\twalltime\tdependency\n' > "$receipt"
submit() {
    local name=$1 minutes=$2 memory=$3 dependency=$4
    shift 4
    if [[ -n ${active[$name]:-} ]]; then REPLY=${active[$name]}; echo "ACTIVE $name $REPLY"; return; fi
    local wall job
    printf -v wall '%02d:%02d:00' "$((minutes/60))" "$((minutes%60))"
    local dependencies=()
    [[ $dependency == - ]] || dependencies=("--dependency=afterok:$dependency" --kill-on-invalid-dep=yes)
    job=$(sbatch --parsable --account=aip-yiweilu --job-name="$name" --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem="$memory" --gres=gpu:l40s:1 --time="$wall" \
        --chdir="$root" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "${dependencies[@]}" "$run/submit_lu.sh" --worker "$run/manifest.json" "$@")
    REPLY=${job%%;*}
    [[ $REPLY =~ ^[0-9]+$ ]]
    active["$name"]=$REPLY
    printf '%s\t%s\t%s\t%s\t%s\taip-yiweilu\t%s\t%s\n' "$1" "$2" "${3:--}" "${4:--}" "$REPLY" "$wall" "$dependency" >> "$receipt"
    echo "$name $REPLY ($wall, Lu)"
}
for source in CIFAR10 CIFAR100 TinyImageNet; do
    selection_dependency=-
    if [[ ! -s $run/selections/$source/complete.json ]]; then
        selection_minutes=20
        [[ $source != TinyImageNet ]] || selection_minutes=40
        submit "sbsel_$source" "$selection_minutes" 16G "${SAS_PROBCOVER_CHECK_JOB:--}" select "$source"
        selection_dependency=$REPLY
    fi
    for method in sas probcover; do
        for percentage in 0.2 1 2; do
            folder="$run/results/$source/$method/pt$percentage/simclr/ssl_seed_000"
            [[ ! -s $folder/complete.json ]] || { echo "DONE $source $method $percentage"; continue; }
            case "$source:$percentage" in
                TinyImageNet:0.2) minutes=60;;
                TinyImageNet:1) minutes=120;;
                TinyImageNet:2) minutes=240;;
                *:0.2) minutes=18;;
                *:1) minutes=28;;
                *:2) minutes=50;;
            esac
            submit "sbev_${source}_${method}_p${percentage/./p}" "$minutes" 6G "$selection_dependency" evaluate "$source" "$method" "$percentage"
        done
    done
done
echo "Receipt: $receipt"
