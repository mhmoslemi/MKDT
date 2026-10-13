#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

old_downstream="$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets"
old_cross="$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP/targets"
for target in CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -f "$old_downstream/$target.pt" ]] || { echo "Missing immutable target cache: $target" >&2; exit 1; }
done
[[ -f "$old_cross/CIFAR10.pt" ]] || { echo "Missing immutable target cache: CIFAR10" >&2; exit 1; }

mkdir -p benchmarks/full_data_refresh/results
FULL_BATCH=$(mktemp -d "$PROJECT_ROOT/benchmarks/full_data_refresh/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
FULL_CODE_ROOT="$FULL_BATCH/code"
TARGET_CACHE_ROOT="$FULL_BATCH/targets"
export FULL_BATCH FULL_CODE_ROOT TARGET_CACHE_ROOT
mkdir -p "$FULL_BATCH/logs" "$FULL_BATCH/runs" "$FULL_CODE_ROOT/benchmarks/full_data_refresh" "$FULL_CODE_ROOT/OT-SSL-DD" "$TARGET_CACHE_ROOT"

ln -s "$old_cross/CIFAR10.pt" "$TARGET_CACHE_ROOT/CIFAR10.pt"
for target in CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    ln -s "$old_downstream/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
done

cp benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py benchmarks/benchmark_full_refresh.py benchmarks/downstream_data.py "$FULL_CODE_ROOT/benchmarks/"
cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$FULL_CODE_ROOT/OT-SSL-DD/"
cp benchmarks/full_data_refresh/report.py "$FULL_CODE_ROOT/benchmarks/full_data_refresh/"
tar -czf "$FULL_BATCH/source.tar.gz" \
    benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py benchmarks/benchmark_full_refresh.py benchmarks/downstream_data.py \
    benchmarks/full_data_refresh/environment.sh benchmarks/full_data_refresh/job.sh \
    benchmarks/full_data_refresh/report.py benchmarks/full_data_refresh/report.sh \
    benchmarks/full_data_refresh/submit.sh \
    OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json

printf 'submitted_utc\tjob_id\taccount\tphase\tconfig\n' > "$FULL_BATCH/submitted.tsv"
printf 'config\tsource\tmodel\tssl_method\tssl_epochs\ttargets\n' > "$FULL_BATCH/configs.tsv"
gpu_index=0
LAST_JOB_ID=''
submit_gpu() {
    local phase=$1 name=$2 limit=$3 dependency=$4
    shift 4
    gpu_index=$((gpu_index + 1))
    local account=aip-boyuwang
    # There are 37 GPU submissions; four Lu jobs is the closest integer to 10%.
    (( gpu_index % 10 == 0 || gpu_index == 37 )) && account=aip-yiweilu
    local args=(sbatch --parsable --account="$account" --job-name="$name" \
        --time="$limit" --mem=8G --chdir="$PROJECT_ROOT" \
        --output="$FULL_BATCH/logs/%x-%j.out")
    [[ -n $dependency ]] && args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    local id
    id=$("${args[@]}" "$PROJECT_ROOT/benchmarks/full_data_refresh/job.sh" "$@")
    LAST_JOB_ID=${id%%;*}
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$account" "$phase" "$name" >> "$FULL_BATCH/submitted.tsv"
    printf 'Submitted %s on %s: %s\n' "$name" "$account" "$LAST_JOB_ID" >&2
}

all_targets='CIFAR10,CIFAR100,TinyImageNet,Aircraft,CUB2011,Dogs,Flowers'
evaluation_ids=()
for source in CIFAR10 CIFAR100 TinyImageNet; do
    for model in ConvNet VGG11 ResNet18; do
        for ssl in simclr barlowtwins; do
            if [[ $ssl == simclr ]]; then ssl_epochs=300; else ssl_epochs=600; fi
            config="full_${source}_${model}_${ssl}"
            targets=$source
            if [[ $source == CIFAR10 || $model == ResNet18 ]]; then
                targets=$all_targets
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$config" "$source" "$model" "$ssl" "$ssl_epochs" "$targets" >> "$FULL_BATCH/configs.tsv"

            case "${source}_${model}_${ssl}" in
                CIFAR10_ConvNet_simclr|CIFAR100_ConvNet_simclr) limit=0-03:15:00 ;;
                CIFAR10_ConvNet_barlowtwins|CIFAR100_ConvNet_barlowtwins) limit=0-07:30:00 ;;
                CIFAR10_VGG11_simclr|CIFAR100_VGG11_simclr) limit=0-04:00:00 ;;
                CIFAR10_VGG11_barlowtwins|CIFAR100_VGG11_barlowtwins) limit=0-07:30:00 ;;
                CIFAR10_ResNet18_simclr|CIFAR100_ResNet18_simclr) limit=0-12:00:00 ;;
                CIFAR10_ResNet18_barlowtwins|CIFAR100_ResNet18_barlowtwins) limit=0-23:30:00 ;;
                TinyImageNet_ConvNet_simclr) limit=0-06:00:00 ;;
                TinyImageNet_ConvNet_barlowtwins) limit=0-14:00:00 ;;
                TinyImageNet_VGG11_simclr) limit=0-08:00:00 ;;
                TinyImageNet_VGG11_barlowtwins) limit=0-14:00:00 ;;
                TinyImageNet_ResNet18_simclr) limit=0-23:30:00 ;;
                TinyImageNet_ResNet18_barlowtwins) limit=0-23:30:00 ;;
            esac

            pretrain_ids=()
            if [[ $source == TinyImageNet && $model == ResNet18 && $ssl == barlowtwins ]]; then
                for seed_start in 0 5; do
                    submit_gpu pretrain "fpre_${source}_${model}_${ssl}_${seed_start}" "$limit" '' \
                        pretrain "$source" "$model" "$ssl" "$ssl_epochs" "$config" "$seed_start" 5
                    pretrain_ids+=("$LAST_JOB_ID")
                done
            else
                submit_gpu pretrain "fpre_${source}_${model}_${ssl}" "$limit" '' \
                    pretrain "$source" "$model" "$ssl" "$ssl_epochs" "$config" 0 10
                pretrain_ids+=("$LAST_JOB_ID")
            fi
            dependency=$(IFS=:; echo "${pretrain_ids[*]}")
            eval_limit=0-00:30:00
            [[ $targets == "$all_targets" ]] && eval_limit=0-02:00:00
            submit_gpu evaluate "feval_${source}_${model}_${ssl}" "$eval_limit" "afterok:$dependency" \
                evaluate "$source" "$model" "$ssl" "$ssl_epochs" "$config" "$targets"
            evaluation_ids+=("$LAST_JOB_ID")
        done
    done
done

dependency=$(IFS=:; echo "${evaluation_ids[*]}")
report_id=$(sbatch --parsable --account=aip-boyuwang --job-name=full_refresh_report \
    --chdir="$PROJECT_ROOT" --output="$FULL_BATCH/logs/%x-%j.out" \
    --dependency="afterany:$dependency" \
    "$PROJECT_ROOT/benchmarks/full_data_refresh/report.sh")
report_id=${report_id%%;*}
printf '%s\t%s\t%s\tCPU\t%s\n' "$(date -u +%FT%TZ)" "$report_id" aip-boyuwang full_refresh_report >> "$FULL_BATCH/submitted.tsv"

printf 'Batch: %s\nGPU jobs: %s\nEvaluation chains: %s\nReport: %s\n' \
    "$FULL_BATCH" "$gpu_index" "${#evaluation_ids[@]}" "$report_id"
