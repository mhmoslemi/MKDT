#!/bin/bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: bash $0 /absolute/path/to/batch" >&2
    exit 2
fi

batch="$(cd -- "$1" && pwd -P)"
manifest="$batch/configs.tsv"
coverage="$batch/table_cells.tsv"
epochs="$batch/training_epochs.json"
for required in "$manifest" "$coverage" "$epochs"; do
    [[ -f $required ]] || { echo "Missing: $required" >&2; exit 1; }
done
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

config_count=$(( $(wc -l < "$manifest") - 1 ))
cell_count=$(( $(wc -l < "$coverage") - 1 ))
table_count=$(tail -n +2 "$coverage" | cut -f1 | sort -u | wc -l)
unique_results=$(tail -n +2 "$coverage" | cut -f5 | sort -u | wc -l)
[[ $config_count == 21 && $cell_count == 78 && $table_count == 6 && $unique_results == 42 ]] || {
    echo "Coverage manifest is invalid: configs=$config_count cells=$cell_count tables=$table_count results=$unique_results" >&2
    exit 1
}

complete=0
missing=0
invalid=0
printf 'target\tmodel\tlabels\tprobe_epochs\tstatus\tmean\tstd\n'
while IFS=$'\t' read -r task_id config target model; do
    [[ $task_id == task_id ]] || {
        for labels in 1 5; do
            probe=$(jq -er --arg target "$target" --arg labels "$labels" \
                '.evaluation_by_dataset[$target].no_pretraining_linear_probe_epochs_by_label_percentage[$labels]' \
                "$epochs")
            result_id="${target}_${model}_no_pretrain_lbl${labels}"
            result_root="$batch/evaluations/$result_id"
            shopt -s nullglob
            summaries=("$result_root"/job_*/summary.json)
            shopt -u nullglob
            if (( ${#summaries[@]} == 0 )); then
                printf '%s\t%s\t%s\t%s\tMISSING\t-\t-\n' "$target" "$model" "$labels" "$probe"
                missing=$((missing + 1))
                continue
            fi
            if (( ${#summaries[@]} != 1 )); then
                printf '%s\t%s\t%s\t%s\tDUPLICATE(%s)\t-\t-\n' \
                    "$target" "$model" "$labels" "$probe" "${#summaries[@]}"
                invalid=$((invalid + 1))
                continue
            fi
            summary=${summaries[0]}
            if ! jq -e \
                --arg target "$target" --arg model "$model" \
                --argjson labels "$labels" --argjson probe "$probe" '
                .complete == true and
                .completed_runs == 10 and .expected_runs == 10 and
                .seeds == [0,1,2,3,4,5,6,7,8,9] and
                (.mean_percent | numbers) and (.std_percent | numbers) and
                .config.target == $target and .config.model == $model and
                .config.method == "no_pretrain" and .config.ssl_method == "none" and
                .config.subset_percentage == 0 and .config.ssl_epochs == 0 and
                .config.label_percentage == $labels and .config.probe_epochs == $probe and
                .config.runs == 10 and .config.label_policy == "random" and
                .config.no_pretrain_protocol == "frozen_random_encoder" and
                .config.protocol == "cifar10_to_downstream_v3_random_labels_barlow2x"
                ' "$summary" >/dev/null; then
                printf '%s\t%s\t%s\t%s\tINVALID\t-\t-\n' "$target" "$model" "$labels" "$probe"
                invalid=$((invalid + 1))
                continue
            fi
            shopt -s nullglob
            seed_files=("$(dirname "$summary")"/seed_*.json)
            shopt -u nullglob
            if (( ${#seed_files[@]} != 10 )); then
                printf '%s\t%s\t%s\t%s\tINVALID_SEEDS(%s)\t-\t-\n' \
                    "$target" "$model" "$labels" "$probe" "${#seed_files[@]}"
                invalid=$((invalid + 1))
                continue
            fi
            mean=$(jq -r '.mean_percent' "$summary")
            std=$(jq -r '.std_percent' "$summary")
            printf '%s\t%s\t%s\t%s\tCOMPLETE\t%.4f\t%.4f\n' \
                "$target" "$model" "$labels" "$probe" "$mean" "$std"
            complete=$((complete + 1))
        done
    }
done < "$manifest"

printf '\nSummary: complete=%s/42 missing=%s invalid=%s; mapped_cells=%s/78; tables=%s/6\n' \
    "$complete" "$missing" "$invalid" "$cell_count" "$table_count"
(( complete == 42 && missing == 0 && invalid == 0 ))
