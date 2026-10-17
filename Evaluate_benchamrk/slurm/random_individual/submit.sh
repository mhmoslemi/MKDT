#!/bin/bash
# Independent jobs only: bounded SSL stages -> probes of the saved encoder.
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
    cp "$script_dir"/{environment.sh,worker.sh,stages.py,validate.sh,validate.py} "$batch/"
    node - "$batch" <<'JS'
const fs=require('fs'),path=require('path'),batch=process.argv[2],bench=path.join(batch,'code/Evaluate_benchamrk');
function readTSV(file) {const [header,...rows]=fs.readFileSync(file,'utf8').trim().split('\n').map(l=>l.split('\t'));return rows.map(r=>Object.fromEntries(header.map((k,i)=>[k,r[i]])));}
const runners=readTSV(path.join(bench,'runners.tsv')).filter(r=>r.method==='random');
const cells=readTSV(path.join(bench,'table_coverage.tsv')).filter(r=>r.method==='random');
if(runners.length!==72 || cells.length!==432)throw Error('Unexpected coverage');
const names=new Set(runners.map(r=>r.script));
for(const c of cells)if(!names.has(c.scripts))throw Error('Uncovered random cell');
const tasks=[];
for(let seed=0;seed<10;seed++)for(const r of runners){
  if(r.seeds!=='0,1,2,3,4,5,6,7,8,9')throw Error('Unexpected seeds');
  const account=seed===9?'aip-yiweilu':'aip-boyuwang';
  const size=Number(r.subset);
  const ends=r.source==='TinyImageNet'&&size===5?[250,500,750,1000]:
    (size===5 || (r.source==='TinyImageNet'&&size===2))?[500,1000]:[1000];
  let dependency='validation';
  for(const end of [...ends,0]){
    const key=String(tasks.length),phase=end?'ssl':'evaluate';
    const shortSource={CIFAR10:'c10',CIFAR100:'c100',TinyImageNet:'tiny'}[r.source];
    const shortModel={ConvNet:'cn',VGG11:'vg',ResNet18:'r18'}[r.model];
    const name=`eri_${shortSource}_${shortModel}_${r.ssl==='simclr'?'s':'b'}_${r.subset.replace('.','p')}_s${seed}_${end||'ev'}`;
    tasks.push({key,account,phase,source:r.source,model:r.model,subset:r.subset,ssl:r.ssl,seed:String(seed),end_epoch:String(end),dependency,name,runner:r.script});
    dependency=key;
  }
}
if(tasks.length!==1800 || tasks.filter(t=>t.phase==='ssl').length!==1080)throw Error('Unexpected task count');
const header=Object.keys(tasks[0]);
fs.writeFileSync(path.join(batch,'tasks.tsv'),header.join('\t')+'\n'+tasks.map(t=>header.map(k=>t[k]).join('\t')).join('\n')+'\n');
fs.writeFileSync(path.join(batch,'manifest.json'),JSON.stringify({selection:'random',cluster:'vulcan',arrays:false,
  configurations:72,seeds_per_configuration:10,ssl_stage_jobs:1080,evaluation_jobs:720,production_jobs:1800,
  accounts:{'aip-boyuwang':1620,'aip-yiweilu':180},cpus_per_job:1,memory:'6G',gpu:'l40s:1',
  time_limit:'01:00:00',workers:0,threads:1,total_ssl_epochs_per_seed:1000,
  logical_cells:432,active_cells:cells.filter(c=>c.disabled==='False').length,seed_evaluations:4320,
  output_root:path.join(batch,'results'),old_cancelled_jobs:[1806957,1806965,1806966]},null,2)+'\n');
JS
    (cd "$batch/code" && rg --files -0 | sort -z | xargs -0 sha256sum) > "$batch/code.sha256"
    (cd "$batch" && sha256sum environment.sh worker.sh stages.py validate.sh validate.py tasks.tsv manifest.json code.sha256) > "$batch/bundle.sha256"
    printf 'submitted_utc\tjob_id\taccount\tphase\tkey\tdependency_job\tname\n' > "$batch/submitted.tsv"
    printf '%s\n' "$batch"
    exit 0
fi
batch=$(readlink -f -- "${2:?batch}")
[[ $batch == "$script_dir"/batches/batch_* && -s $batch/manifest.json ]] || exit 2
exec 9>"$batch/.submit.lock"
flock -n 9 || { echo 'Another submitter is active' >&2; exit 2; }
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
sinfo -h -o '%G' --Node | sort -u | grep -q 'gpu:l40s:'
(cd "$batch/code" && sha256sum --check --quiet "$batch/code.sha256")
(cd "$batch" && sha256sum --check --quiet bundle.sha256)
declare -A job_ids
while IFS=$'\t' read -r submitted job account phase key dep name; do
    [[ $submitted == submitted_utc ]] && continue
    job_ids[$key]=$job
done < "$batch/submitted.tsv"
if [[ $mode == validate ]]; then
    [[ -z ${job_ids[validation]:-} ]] || { echo 'Validation already submitted'; exit 2; }
    job=$(sbatch --parsable --account=aip-boyuwang --job-name=eri_check \
        --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time=00:30:00 \
        --chdir="$batch/code" --output="$batch/logs/validation-%j.out" \
        --error="$batch/logs/validation-%j.err" "$batch/validate.sh" "$batch")
    job=${job%%;*}
    printf '%s\t%s\taip-boyuwang\tvalidation\tvalidation\t-\teri_check\n' "$(date -u +%FT%TZ)" "$job" >> "$batch/submitted.tsv"
    printf 'Validation: %s\n' "$job"
elif [[ $mode == production ]]; then
    [[ ${job_ids[validation]:-} =~ ^[0-9]+$ ]] || exit 2
    count=0
    while IFS=$'\t' read -r key account phase source model subset ssl seed end_epoch dependency name runner; do
        [[ $key == key ]] && continue
        [[ -z ${job_ids[$key]:-} ]] || continue
        dependency_job=${job_ids[$dependency]:?Missing upstream job}
        job=$(sbatch --parsable --account="$account" --job-name="$name" \
            --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time=01:00:00 \
            --dependency="afterok:$dependency_job" --chdir="$batch/code" \
            --output="$batch/logs/$name-%j.out" --error="$batch/logs/$name-%j.err" \
            "$batch/worker.sh" "$batch" "$key")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo 'Unexpected sbatch response' >&2; exit 2; }
        job_ids[$key]=$job
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$job" "$account" "$phase" "$key" "$dependency_job" "$name" >> "$batch/submitted.tsv"
        count=$((count+1))
        if ((count % 100 == 0)); then printf 'Submitted %s additional individual jobs; latest=%s\n' "$count" "$job"; fi
    done < "$batch/tasks.tsv"
    printf 'Submission complete: %s additional individual jobs\n' "$count"
else
    echo 'Unknown mode' >&2
    exit 2
fi
