#!/bin/bash
set -u

if (( $# > 1 )); then
    echo "Usage: bash $0 [mkdt-batch-directory]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
default_batch="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
batch="${1:-$default_batch}"
generated="$batch/synthetic/generated"

if [[ ! -d $generated ]]; then
    echo "MKDT generated-data directory does not exist:" >&2
    echo "  $generated" >&2
    exit 2
fi

printf 'Checking canonical MKDT distilled data\n'
printf 'Server: %s\n' "$(hostname -f 2>/dev/null || hostname)"
printf 'Root:   %s\n\n' "$generated"
printf '%-14s %-8s %-4s %-9s %s\n' DATASET GENERATOR SIZE STATUS FINAL_CHECKPOINT

complete=0
missing=0
for generator in ConvNet VGG11; do
    for dataset in CIFAR10 CIFAR100 TinyImageNet; do
        dataset_token=$dataset
        final_iteration=5000
        if [[ $dataset == TinyImageNet ]]; then
            dataset_token=Tiny
            final_iteration=10000
        fi

        for percentage in 1 2 5; do
            output="$generated/$generator/$dataset/${dataset_token}_${percentage}pct"
            image="$output/images_${final_iteration}.pt"
            labels="$output/labels_${final_iteration}.pt"
            synlr="$output/synlr_${final_iteration}.pt"
            absent=()

            [[ -s $image ]] || absent+=("$image")
            [[ -s $labels ]] || absent+=("$labels")
            [[ -s $synlr ]] || absent+=("$synlr")

            if (( ${#absent[@]} == 0 )); then
                printf '%-14s %-8s %-4s %-9s %s\n' \
                    "$dataset" "$generator" "${percentage}%" COMPLETE "$image"
                complete=$((complete + 1))
            else
                printf '%-14s %-8s %-4s %-9s %s\n' \
                    "$dataset" "$generator" "${percentage}%" MISSING "$image"
                for file in "${absent[@]}"; do
                    printf '  missing: %s\n' "$file"
                done
                missing=$((missing + 1))
            fi
        done
    done
done

printf '\nSummary: complete=%d/18 missing=%d/18\n' "$complete" "$missing"
if (( missing == 0 )); then
    echo 'ALL MKDT DISTILLED DATA IS PRESENT.'
    exit 0
fi

echo 'MKDT DISTILLED DATA IS INCOMPLETE ON THIS SERVER.'
exit 1
