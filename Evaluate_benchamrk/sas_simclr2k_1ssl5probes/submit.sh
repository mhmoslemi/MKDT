#!/bin/bash
set -euo pipefail
run=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/sas_simclr2k_1ssl5probes
cd "$run"
[[ -f jobs.tsv ]] || exit 2
exec 9>.submit.lock
flock -n 9
queue=$(squeue -h -u "$USER" -o '%j|%i')
while IFS=$'\t' read -r source pct count ssl_seconds probe_seconds; do
    [[ $source == CIFAR10 || $source == CIFAR100 ]] || exit 2
    case "$pct" in 0.2|1|2) ;; *) exit 2;; esac
    ssl_id=
    for stage in ssl probe; do
        name="sas2k_${source}_p${pct/./p}_${stage}"
        folder="$run/results/$source/pt$pct"
        if [[ $stage == ssl && -f $folder/ssl_complete.json ]]; then continue; fi
        if [[ $stage == probe && -f $folder/summary_lbl1.json && -f $folder/summary_lbl5.json ]]; then continue; fi
        existing=$(awk -F '|' -v name="$name" '$1==name {print $2}' <<< "$queue")
        if [[ -n $existing ]]; then
            [[ $stage != ssl ]] || ssl_id=$existing
            echo "Already queued: $name $existing"
            continue
        fi
        dependency=()
        seconds=$ssl_seconds
        if [[ $stage == probe ]]; then
            seconds=$probe_seconds
            if [[ -n $ssl_id ]]; then
                dependency=("--dependency=afterok:$ssl_id" --kill-on-invalid-dep=yes)
            else
                [[ -f $folder/ssl_complete.json ]] || { echo 'Missing SSL predecessor.' >&2; exit 2; }
            fi
        fi
        [[ $seconds =~ ^[0-9]+$ && $seconds -gt 0 ]] || exit 2
        printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
        job=$(sbatch --parsable --account=aip-yiweilu --job-name="$name" \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:l40s:1 --mem=6G --time="$walltime" \
            --output="$run/logs/%x-%j.out" "${dependency[@]}" "$run/job.sh" "$stage" "$source" "$pct")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || exit 2
        [[ $stage != ssl ]] || ssl_id=$job
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$job" "$source" "$pct" "$stage" "$walltime" "${dependency[*]:--}" | tee -a submitted.tsv
    done
done < jobs.tsv
