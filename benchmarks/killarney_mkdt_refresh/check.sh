#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if (( $# == 1 )); then
    batch="$(cd -- "$1" && pwd -P)"
elif (( $# == 0 )); then
    shopt -s nullglob
    batches=("$PROJECT_ROOT"/benchmarks/killarney_mkdt_refresh/results/batch_*)
    shopt -u nullglob
    (( ${#batches[@]} > 0 )) || { echo 'No MKDT refresh batches found' >&2; exit 2; }
    batch=$(printf '%s\n' "${batches[@]}" | sort | tail -1)
else
    echo "Usage: bash $0 [batch-directory]" >&2
    exit 2
fi

configs="$batch/configs.tsv"
cells="$batch/table_cells.tsv"
[[ -s $configs && -s $cells ]] || { echo "Incomplete batch metadata: $batch" >&2; exit 2; }

complete_encoders=0
missing_encoders=0
invalid_encoders=0
while IFS=$'\t' read -r config source generator model size ssl epochs scope pretrain_job evaluate_job; do
    [[ $config == config ]] && continue
    encoder="$batch/runs/$config/encoders/seed_000.pt"
    shard="$batch/runs/$config/encoders/shard_000_000.json"
    if [[ ! -s $encoder || ! -s $shard ]]; then
        printf 'MISSING encoder: %s\n' "$config"
        missing_encoders=$((missing_encoders + 1))
        continue
    fi
    if jq -e --arg protocol mkdt_10k_probe1000_500_shared_encoder_v1 '
        .complete == true and .completed_runs == 1 and .expected_runs == 1 and
        .seeds == [0] and .config.protocol == $protocol and
        .config.ssl_epochs == 10000 and .config.runs == 1
        ' "$shard" >/dev/null; then
        complete_encoders=$((complete_encoders + 1))
    else
        printf 'INVALID encoder metadata: %s\n' "$config"
        invalid_encoders=$((invalid_encoders + 1))
    fi
done < "$configs"

complete_cells=0
missing_cells=0
invalid_cells=0
while IFS=$'\t' read -r table config source target generator model size ssl labels relative; do
    [[ $table == table ]] && continue
    summary="$batch/$relative"
    if [[ ! -s $summary ]]; then
        printf 'MISSING result: %s %s labels=%s%% size=%s%% %s %s\n' \
            "$source" "$target" "$labels" "$size" "$model" "$ssl"
        missing_cells=$((missing_cells + 1))
        continue
    fi
    probe=1000
    [[ $labels == 5 ]] && probe=500
    if jq -e --arg protocol mkdt_10k_probe1000_500_shared_encoder_v1 \
        --arg source "$source" --arg target "$target" --arg generator "$generator" \
        --arg model "$model" --arg ssl "$ssl" --argjson size "$size" \
        --argjson labels "$labels" --argjson probe "$probe" '
        .complete == true and .completed_runs == 15 and .expected_runs == 15 and
        .seeds == [range(0; 15)] and
        (.mean_percent | numbers) and (.std_percent | numbers) and
        .config.protocol == $protocol and .config.source == $source and
        .config.target == $target and .config.generator == $generator and
        .config.model == $model and .config.ssl_method == $ssl and
        .config.subset_percentage == $size and .config.ssl_epochs == 10000 and
        .config.label_percentage == $labels and .config.probe_epochs == $probe and
        .config.runs == 15 and .config.encoder_reuse_seed == 0
        ' "$summary" >/dev/null; then
        complete_cells=$((complete_cells + 1))
    else
        printf 'INVALID result: %s\n' "$summary"
        invalid_cells=$((invalid_cells + 1))
    fi
done < "$cells"

table_count=$(tail -n +2 "$cells" | cut -f1 | sort -u | wc -l)
printf 'Batch: %s\nEncoders: complete=%d/36 missing=%d invalid=%d\nResults: complete=%d/312 missing=%d invalid=%d\nTables: %d/6\n' \
    "$batch" "$complete_encoders" "$missing_encoders" "$invalid_encoders" \
    "$complete_cells" "$missing_cells" "$invalid_cells" "$table_count"

(( complete_encoders == 36 && missing_encoders == 0 && invalid_encoders == 0 &&
   complete_cells == 312 && missing_cells == 0 && invalid_cells == 0 && table_count == 6 ))
