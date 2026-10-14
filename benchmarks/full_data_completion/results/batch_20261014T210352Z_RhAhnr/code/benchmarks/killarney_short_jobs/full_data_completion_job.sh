#!/bin/bash
set -euo pipefail

[[ $# -ge 1 ]] || {
    echo "Usage: $0 <cross-main|c10-pretrain|c10-evaluate> [...]" >&2
    exit 2
}

: "${COMPLETION_CODE_ROOT:?}"
source "$COMPLETION_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_completion_environment.sh"

task=$1
shift

check_ten_direct_checkpoints() {
    local root=$1
    local seed checkpoint
    for (( seed=0; seed<10; seed++ )); do
        printf -v checkpoint '%s/seed_%03d.pt' "$root" "$seed"
        [[ -s $checkpoint ]] || {
            echo "Missing checkpoint: $checkpoint" >&2
            return 1
        }
    done
}

evaluate_cross_main() {
    [[ $# -eq 2 ]] || {
        echo "cross-main requires <CIFAR100|TinyImageNet> <simclr|barlowtwins>" >&2
        exit 2
    }
    local source=$1
    local ssl=$2
    local prefix identifier encoder label probe output

    case "$source" in
        CIFAR100) prefix=cifar100 ;;
        TinyImageNet) prefix=tinyimagenet ;;
        *) echo "Unsupported cross-main source: $source" >&2; exit 2 ;;
    esac
    case "$ssl" in
        simclr|barlowtwins) ;;
        *) echo "Unsupported SSL method: $ssl" >&2; exit 2 ;;
    esac

    identifier="${prefix}_ResNet18_${ssl}_full_size100"
    encoder="$CROSS_BATCH/encoders/$identifier"
    check_ten_direct_checkpoints "$encoder"
    [[ -s $TARGET_CACHE_ROOT/$source.pt ]] || {
        echo "Missing target cache: $TARGET_CACHE_ROOT/$source.pt" >&2
        exit 1
    }

    cd "$CROSS_BATCH/code"
    for label in 1 5; do
        if [[ $label == 1 ]]; then probe=500; else probe=300; fi
        output="$COMPLETION_BATCH/evaluations/${identifier}__${source}_lbl${label}/job_$SLURM_JOB_ID"
        mkdir -p "$(dirname "$output")"
        srun --unbuffered python3 benchmarks/benchmark_cross_dataset.py evaluate \
            --output "$output" \
            --source "$source" --target "$source" \
            --target-cache "$TARGET_CACHE_ROOT/$source.pt" \
            --encoder-dir "$encoder" \
            --method full --ssl-method "$ssl" \
            --subset-percentage 100 --ssl-epochs 300 \
            --label-percentage "$label" --probe-epochs "$probe" --runs 10
    done
}

pretrain_c10() {
    [[ $# -eq 2 ]] || {
        echo "c10-pretrain requires <seed-start> <runs>" >&2
        exit 2
    }
    local seed_start=$1
    local runs=$2
    local final_seed
    local encoder_root="$COMPLETION_BATCH/encoders/cifar10_ResNet18_barlowtwins_full_size100"
    local output

    [[ $seed_start =~ ^[0-9]+$ && $runs =~ ^[0-9]+$ ]] \
        && (( runs > 0 && seed_start + runs <= 10 )) || {
        echo "Invalid seed shard: seed_start=$seed_start runs=$runs" >&2
        exit 2
    }
    final_seed=$((seed_start + runs - 1))
    output="$encoder_root/shard_${seed_start}_${final_seed}"
    [[ ! -e $output ]] || {
        echo "Refusing to overwrite existing encoder shard: $output" >&2
        exit 1
    }
    [[ -s $TARGET_CACHE_ROOT/CIFAR10.pt ]] || {
        echo "Missing target cache: $TARGET_CACHE_ROOT/CIFAR10.pt" >&2
        exit 1
    }

    mkdir -p "$encoder_root"
    cd "$COMPLETION_CODE_ROOT"
    srun --unbuffered python3 benchmarks/benchmark_full_refresh.py pretrain \
        --source CIFAR10 --source-cache "$TARGET_CACHE_ROOT/CIFAR10.pt" \
        --output "$output" --model ResNet18 --ssl-method barlowtwins \
        --ssl-epochs 600 --seed-start "$seed_start" --runs "$runs" \
        --ssl-log-every 100
}

evaluate_c10() {
    [[ $# -eq 3 ]] || {
        echo "c10-evaluate requires <encoder-root> <ssl-epochs> <completion|refresh>" >&2
        exit 2
    }
    local encoder=$1
    local ssl_epochs=$2
    local engine=$3
    local program code_root target label probe identifier output

    case "$engine" in
        completion)
            code_root=$COMPLETION_CODE_ROOT
            program=benchmarks/benchmark_full_refresh.py
            ;;
        refresh)
            code_root=$FULL_REFRESH_BATCH/code
            program=benchmarks/benchmark_full_refresh.py
            [[ -s $code_root/$program ]] || {
                echo "Missing immutable full-refresh code: $code_root/$program" >&2
                exit 1
            }
            ;;
        *) echo "Unsupported C10 evaluation engine: $engine" >&2; exit 2 ;;
    esac
    [[ $ssl_epochs =~ ^[0-9]+$ ]] && (( ssl_epochs > 0 )) || {
        echo "Invalid SSL epoch count: $ssl_epochs" >&2
        exit 2
    }
    identifier=cifar10_ResNet18_barlowtwins_full_size100

    cd "$code_root"
    for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
        [[ -s $TARGET_CACHE_ROOT/$target.pt ]] || {
            echo "Missing target cache: $TARGET_CACHE_ROOT/$target.pt" >&2
            exit 1
        }
        for label in 1 5; do
            if [[ $label == 1 ]]; then probe=500; else probe=300; fi
            output="$COMPLETION_BATCH/evaluations/${identifier}__${target}_lbl${label}/job_$SLURM_JOB_ID"
            mkdir -p "$(dirname "$output")"
            srun --unbuffered python3 "$program" evaluate \
                --source CIFAR10 --encoder-dir "$encoder" \
                --target "$target" --target-cache "$TARGET_CACHE_ROOT/$target.pt" \
                --output "$output" --model ResNet18 --ssl-method barlowtwins \
                --ssl-epochs "$ssl_epochs" --label-percentage "$label" \
                --probe-epochs "$probe" --runs 10
        done
    done
}

case "$task" in
    cross-main) evaluate_cross_main "$@" ;;
    c10-pretrain) pretrain_c10 "$@" ;;
    c10-evaluate) evaluate_c10 "$@" ;;
    *) echo "Unsupported task: $task" >&2; exit 2 ;;
esac
