#!/bin/bash
# Run on either server AFTER teacher training and synthetic-data generation finish.
# Permanently deletes teacher files only. Never removes directories or synthetic data.
set -euo pipefail
shopt -s nullglob

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$script_dir/.." && pwd -P)
[[ -f "$repo/buffer.py" && -f "$repo/distill.py" && -d "$repo/Evaluate_benchamrk" ]] || {
    echo 'Keep this script in MKDT/Evaluate_benchamrk.' >&2
    exit 1
}
deleted=0

delete_teacher_file() {
    local file=$1 resolved
    [[ -f "$file" && ! -L "$file" ]] || return 0
    resolved=$(realpath -e -- "$file")
    # Skip symlinked parent directories, including links outside this repository.
    [[ "$resolved" == "$file" && "$resolved" == "$repo/"* ]] || {
        printf 'SKIPPED linked path: %s\n' "$file"
        return 0
    }
    rm -- "$file"
    printf 'DELETED: %s\n' "$file"
    deleted=$((deleted + 1))
}

# Teacher backbones and their resumable training checkpoints, not SSL benchmark encoders.
for file in \
    "$repo"/krrst_teacher_ckpt/teacher_cifar10.pt \
    "$repo"/krrst_teacher_ckpt/barlow_twins_resnet18_{cifar10,cifar100,tinyimagenet}{,_training}.pt \
    "$repo"/artifacts/mkdt_official/teacher_ckpt.zip; do
    delete_teacher_file "$file"
done

# Expert trajectory files in the generation pipeline's buffer locations only.
# In particular, do NOT delete synthetic/, synthesis/, work/, or generated/ directories.
buffer_roots=(
    "$repo"/buffers_barlow_twins
    "$repo"/mkdt_synthetic/work/buffers
    "$repo"/mkdt_synthetic/buffers
    "$repo"/benchmarks/*/results/batch_*/synthetic/work/buffers
    "$repo"/benchmarks/*/results/batch_*/synthesis/buffers
)
for buffer_root in "${buffer_roots[@]}"; do
    [[ -d "$buffer_root" && ! -L "$buffer_root" ]] || continue
    for file in "$buffer_root"/*/*/replay_buffer_*.pt; do
        [[ ${file##*/} =~ ^replay_buffer_[0-9]+\.pt$ ]] || continue
        delete_teacher_file "$file"
    done
done

printf 'DONE: deleted %s teacher/expert files. Synthetic data was not touched.\n' "$deleted"
