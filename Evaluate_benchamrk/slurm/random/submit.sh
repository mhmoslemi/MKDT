#!/bin/bash
# Prepare a frozen bundle, validate it, then explicitly submit production.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(readlink -f "$script_dir/../../..")
mode=${1:?Usage: submit.sh prepare | validate BATCH | production BATCH}
if [[ $mode == prepare ]]; then
    mkdir -p "$script_dir/batches"
    batch=$(mktemp -d "$script_dir/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    mkdir -p "$batch/code/Evaluate_benchamrk/scripts" "$batch/code/benchmarks" "$batch/code/CVPR_tables" "$batch/logs"
    cp "$project_root/networks.py" "$batch/code/"
    cp "$project_root/benchmarks/downstream_data.py" "$batch/code/benchmarks/"
    cp "$project_root"/CVPR_tables/*.tex "$batch/code/CVPR_tables/"
    cp "$project_root"/Evaluate_benchamrk/*.py "$project_root"/Evaluate_benchamrk/*.sh \
        "$project_root"/Evaluate_benchamrk/*.json "$project_root"/Evaluate_benchamrk/*.tsv \
        "$batch/code/Evaluate_benchamrk/"
    cp "$project_root"/Evaluate_benchamrk/scripts/*.sh "$batch/code/Evaluate_benchamrk/scripts/"
    cp "$script_dir"/{environment.sh,worker.sh,run_seed.py,validate.sh,validate.py} "$batch/"
    node - "$batch" <<'JS'
const fs = require('fs'), path = require('path');
const batch = process.argv[2], bench = path.join(batch, 'code/Evaluate_benchamrk');
function readTSV(file) {
  const [header,...rows] = fs.readFileSync(file,'utf8').trim().split('\n').map(l=>l.split('\t'));
  return rows.map(row=>Object.fromEntries(header.map((key,i)=>[key,row[i]])));
}
const runners = readTSV(path.join(bench,'runners.tsv')).filter(r=>r.method==='random');
const cells = readTSV(path.join(bench,'table_coverage.tsv')).filter(r=>r.method==='random');
if(runners.length!==72 || cells.length!==432) throw Error('Unexpected random coverage');
const names = new Set(runners.map(r=>r.script));
for(const cell of cells) if(!names.has(cell.scripts)) throw Error('Uncovered cell '+JSON.stringify(cell));
const tasks = {'aip-boyuwang': [], 'aip-yiweilu': []};
// Interleave configurations rather than launch all seeds of one configuration together.
for(let seed=0;seed<10;seed++) for(const row of runners) {
  if(row.seeds!=='0,1,2,3,4,5,6,7,8,9') throw Error('Unexpected seeds');
  const account = seed===9 ? 'aip-yiweilu' : 'aip-boyuwang';
  tasks[account].push([tasks[account].length,row.source,row.model,row.subset,row.ssl,seed,row.script]);
}
for(const [account,rows] of Object.entries(tasks)) fs.writeFileSync(path.join(batch,`tasks_${account}.tsv`),
  'task_id\tsource\tmodel\tsubset\tssl\tseed\trunner\n'+rows.map(r=>r.join('\t')).join('\n')+'\n');
fs.writeFileSync(path.join(batch,'manifest.json'),JSON.stringify({selection:'random',cluster:'vulcan',
  configurations:72,seeds_per_configuration:10,tasks:720,accounts:{'aip-boyuwang':648,'aip-yiweilu':72},
  logical_cells:432,active_cells:cells.filter(c=>c.disabled==='False').length,
  seed_evaluations:4320,time_limit:'04:00:00',gpus_per_task:1,gpu_type:'l40s',cpus_per_task:8,
  memory:'16G',workers:4,threads:4,data_root:'/scratch/mmoslem3/data',
  output_root:path.join(batch,'results'),concurrency:{'aip-boyuwang':36,'aip-yiweilu':4}},null,2)+'\n');
JS
    (cd "$batch/code" && rg --files -0 | sort -z | xargs -0 sha256sum) > "$batch/code.sha256"
    printf 'submitted_utc\tjob_id\taccount\tphase\ttasks\tarray\n' > "$batch/submitted.tsv"
    printf '%s\n' "$batch"
    exit 0
fi
batch=$(readlink -f -- "${2:?batch directory}")
[[ $batch == "$script_dir"/batches/batch_* && -s $batch/manifest.json ]] || { echo 'Invalid batch' >&2; exit 2; }
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
sinfo -h -o '%G' --Node | sort -u | grep -q 'gpu:l40s:'
cd "$batch/code"
sha256sum --check --quiet "$batch/code.sha256"
if [[ $mode == validate ]]; then
    ! awk -F '\t' 'NR>1 && $4=="validation" {found=1} END {exit !found}' "$batch/submitted.tsv" || { echo 'Validation already submitted' >&2; exit 2; }
    job=$(sbatch --parsable --account=aip-boyuwang --job-name=ebr_rand_check \
        --gres=gpu:l40s:1 --cpus-per-task=8 --mem=16G --time=00:30:00 \
        --chdir="$batch/code" --output="$batch/logs/validation-%j.out" \
        --error="$batch/logs/validation-%j.err" "$batch/validate.sh" "$batch")
    job=${job%%;*}
    printf '%s\t%s\taip-boyuwang\tvalidation\t1\t-\n' "$(date -u +%FT%TZ)" "$job" >> "$batch/submitted.tsv"
    printf 'Validation job: %s\n' "$job"
elif [[ $mode == production ]]; then
    # It is safe to queue before validation finishes: afterok gates the arrays,
    # and each worker independently checks the complete validation marker.
    validation_job=$(awk -F '\t' '$4=="validation"{print $2}' "$batch/submitted.tsv")
    [[ $validation_job =~ ^[0-9]+$ ]] || exit 2
    for account in aip-boyuwang aip-yiweilu; do
        if awk -F '\t' -v account="$account" 'NR>1 && $3==account && $4=="production" {found=1} END {exit !found}' "$batch/submitted.tsv"; then
            printf 'Already submitted: %s\n' "$account"
            continue
        fi
        if [[ $account == aip-boyuwang ]]; then count=648; array='0-647%36'; short=wang
        else count=72; array='0-71%4'; short=lu; fi
        job=$(sbatch --parsable --account="$account" --job-name="ebr_rand_$short" \
            --gres=gpu:l40s:1 --cpus-per-task=8 --mem=16G --time=04:00:00 \
            --array="$array" --dependency="afterok:$validation_job" \
            --chdir="$batch/code" --output="$batch/logs/$short-%A_%a.out" \
            --error="$batch/logs/$short-%A_%a.err" "$batch/worker.sh" "$batch" "$account")
        job=${job%%;*}
        printf '%s\t%s\t%s\tproduction\t%s\t%s\n' "$(date -u +%FT%TZ)" "$job" "$account" "$count" "$array" >> "$batch/submitted.tsv"
        printf 'Submitted %s: %s (%s seed tasks)\n' "$account" "$job" "$count"
    done
else
    echo 'Unknown mode' >&2
    exit 2
fi
