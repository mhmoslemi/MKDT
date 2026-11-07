#!/bin/bash
set -euo pipefail
cd /scratch/mmoslem3/MKDT
run=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_selection_init_lu
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]]
[[ $(sinfo -h -o '%G' --Node | sort -u) == *gpu:l40s:* ]]
sacctmgr -nP show assoc user=mmoslem3 format=Account | awk -F '|' '$1=="aip-yiweilu" {found=1} END {exit !found}'
module spider python/3.11.5 >/dev/null 2>&1
mkdir -p "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9
node - /scratch/mmoslem3/MKDT <<'JS'
const fs=require('fs'),crypto=require('crypto'),assert=require('assert/strict');
const root=process.argv[2],hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const folder=root+'/Evaluate_benchamrk/selection_baselines/selections/CIFAR10',marker=read(folder+'/complete.json');
assert(marker.complete);
for(const variant of ['sas','probcover'])for(const percentage of [0.2,1,2]){
 const name=`${variant}_pt${percentage}.json`,file=folder+'/'+name,s=read(file);
 assert.equal(hash(file),marker.files[name]);assert(s.complete);assert.equal(s.method,variant);assert.equal(s.percentage,percentage);
 assert.equal(s.indices.length,50000*percentage/100);assert.equal(new Set(s.indices).size,s.indices.length);
 assert(s.indices.every(i=>Number.isInteger(i)&&i>=0&&i<50000));
 assert.equal(s.expected.ground_truth_labels_used,false);
 const baseline=read(root+`/Evaluate_benchamrk/mipa_cvpr_color_crop/snapshot/CIFAR10_pt${percentage===0.2?'0p2':percentage}.json`);
 for(const [f,h] of Object.entries(baseline.code_hashes))assert.equal(hash(f),h,'Changed table dependency: '+f);
 assert.equal(hash(baseline.teacher),baseline.teacher_sha256);assert.equal(s.expected.teacher_sha256,baseline.teacher_sha256);
 assert.deepEqual([baseline.generation_args.lr_img,baseline.generation_args.batch_real,baseline.generation_args.Iteration],
                  [0.0015,128,percentage===0.2?10000:percentage===1?8000:6000]);
}
console.log('PASS: all six selections and frozen main-table dependencies verified.');
JS
declare -A active=()
while IFS='|' read -r name job; do [[ -z $name ]] || active["$name"]=$job; done < <(squeue -h -u mmoslem3 -o '%j|%i')
receipt="$run/submitted.tsv"
[[ -s $receipt ]] || printf 'variant\tpercentage\tstage\tjob\taccount\twalltime\tdependency\n' > "$receipt"
for variant in sas probcover; do
    for percentage in 0.2 1 2; do
        tag=${percentage/./p}
        name="mipaselect_lu_${variant}_C10_p${tag}"
        folder="$run/runs/$variant/pt$percentage"
        case "$percentage" in
            0.2) generation_minutes=15; evaluation_minutes=21;;
            1) generation_minutes=31; evaluation_minutes=32;;
            2) generation_minutes=56; evaluation_minutes=69;;
        esac
        dependency=-
        for stage in generation evaluation; do
            suffix=g; minutes=$generation_minutes; marker="$folder/generation_complete.json"
            if [[ $stage == evaluation ]]; then suffix=e; minutes=$evaluation_minutes; marker="$folder/evaluation/complete.json"; fi
            job_name="${name}_${suffix}"
            if [[ -s $marker ]]; then echo "DONE $job_name"; continue; fi
            if [[ -n ${active[$job_name]:-} ]]; then
                job=${active[$job_name]}
                echo "ACTIVE $job_name $job"
            else
                args=()
                [[ $stage != evaluation || $dependency == - ]] || args=("--dependency=afterok:$dependency" --kill-on-invalid-dep=yes)
                printf -v wall '%02d:%02d:00' "$((minutes/60))" "$((minutes%60))"
                job=$(sbatch --parsable --account=aip-yiweilu --job-name="$job_name" \
                    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" \
                    --chdir=/scratch/mmoslem3/MKDT --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
                    "${args[@]}" "$run/job.sh" "$variant" "$percentage" "$stage")
                job=${job%%;*}
                [[ $job =~ ^[0-9]+$ ]]
                printf '%s\t%s\t%s\t%s\taip-yiweilu\t%s\t%s\n' "$variant" "$percentage" "$stage" "$job" "$wall" "$dependency" >> "$receipt"
                echo "$job_name $job ($wall, Lu)"
            fi
            [[ $stage != generation ]] || dependency=$job
        done
    done
done
echo "Receipt: $receipt"
