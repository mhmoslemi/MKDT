#!/bin/bash
# Run separately on Vulcan and Killarney. Default: preview only.
# STRICTLY lower configured SSL epochs, as explicitly requested by the user.
set -euo pipefail
case ${1:---dry-run} in
    --dry-run|--trash|--delete) mode=${1:---dry-run} ;;
    --help|-h)
        printf '%s\n' \
            'bash Evaluate_benchamrk/cleanup_lower_epoch_runs.sh --dry-run  # preview only' \
            'bash Evaluate_benchamrk/cleanup_lower_epoch_runs.sh --trash    # reversible move' \
            'bash Evaluate_benchamrk/cleanup_lower_epoch_runs.sh --delete   # permanent; asks for DELETE' \
            'Run separately on each server. No SSH, submissions, cancellations, or Git changes.' \
            'Targets only random/kmeans/MKDT run folders with a saved config proving lower SSL epochs.' \
            'Keeps current runs, equal/higher epochs, 5%/unknown budgets, code, synthetic images,' \
            'generation buffers, shared caches, full-data/other methods, and unclassified/shared logs.' \
            'Missing metadata means KEEP. No age/name-only deletion or inferred epoch counts.' \
            'Trash stays on the same filesystem and does not free quota. Delete is irreversible.'
        exit 0 ;;
    *) echo 'Use --dry-run, --trash, or --delete.' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Unexpected arguments.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$bench_dir/.." && pwd -P)
[[ $bench_dir == "$repo/Evaluate_benchamrk" && -f $repo/Evaluate_benchamrk/train_ssl.py && -d $repo/benchmarks ]] || {
    echo 'Run the script from its original Evaluate_benchamrk directory.' >&2; exit 2;
}
case "$repo" in /scratch/*/MKDT|/project/*/MKDT) ;; *) echo "Unexpected repository location: $repo" >&2; exit 2 ;; esac
[[ $(stat -c %u -- "$repo") == "$(id -u)" ]] || { echo 'Repository is not owned by the current user.' >&2; exit 2; }
for tool in rg jq realpath sha256sum; do
    command -v "$tool" >/dev/null || { echo "Required command is unavailable: $tool" >&2; exit 2; }
done
rules="$bench_dir/cleanup_lower_epoch_rules.jq"
[[ -r $rules ]] || { echo "Missing $rules; sync both cleanup files." >&2; exit 2; }
scratch_plan=$(mktemp -d /tmp/mkdt_cleanup_plan.XXXXXX)
files="$scratch_plan/configs.list"
plan="$scratch_plan/plan.tsv"
# Two scoped repository trees; do not follow symlinks or scan home/scratch roots.
rg --files --hidden --no-ignore -g config.json \
    -g '!**/code/**' -g '!**/synthetic/**' -g '!**/synthesis/**' \
    -g '!**/generated/**' -g '!**/buffers/**' -g '!**/work/**' \
    -g '!**/targets/**' -g '!**/selections/**' -g '!**/cache/**' \
    -g '!**/cleanup_trash/**' -g '!**/cleanup_reports/**' \
    -g '!**/full_data*/**' -g '!**/ours/**' -g '!**/synthetic_new/**' \
    "$bench_dir" "$repo/benchmarks" > "$files" || {
        inventory_status=$?
        [[ $inventory_status == 1 && ! -s $files ]] || { echo 'Inventory failed; nothing changed.' >&2; exit 2; }
    }
# If ANY input is malformed or disappears, fail before performing any cleanup.
while IFS= read -r config; do
    [[ ! -L $config ]] || continue
    jq -r --arg root "$repo" -f "$rules" "$config"
done < "$files" > "$plan"

declare -a targets=() config_hashes=() inode_ids=()
declare -A seen=()
safe_target() {
    local relative=$1 absolute="$repo/$1" actual classified
    [[ $relative != *$'\n'* && $relative != *$'\t'* && $relative != *'..'* && $relative != /* ]] || return 1
    [[ -d $absolute && ! -L $absolute && -f $absolute/config.json && ! -L $absolute/config.json ]] || return 1
    actual=$(realpath -e -- "$absolute") || return 1
    [[ $actual == "$absolute" && $(stat -c %u -- "$absolute") == "$(id -u)" ]] || return 1
    # Re-run the independent positive allowlist AND protected-path rules.
    classified=$(jq -r --arg root "$repo" -f "$rules" "$absolute/config.json") || return 1
    [[ $classified == REMOVE_LOWER$'\t'"$relative"$'\t'* ]] || return 1
    case "$relative" in
        Evaluate_benchamrk/*/*|benchmarks/*/results/batch_*/*) ;;
        *) return 1 ;;
    esac
    # No broad repository/filesystem root can pass these checks.
    [[ $actual != "$repo" && $actual != "$bench_dir" && $actual != "$repo/benchmarks" ]] || return 1
}

printf 'REMOVE only strictly lower SSL epoch budgets:\n'
printf 'path\tmethod\tsource\tsubset\told_epochs\tcurrent_epochs\n'
while IFS=$'\t' read -r decision relative method source_name subset old_epochs current_epochs; do
    [[ $decision == REMOVE_LOWER ]] || continue
    [[ -z ${seen[$relative]:-} ]] || continue
    safe_target "$relative" || { echo "Unsafe target rejected: $relative" >&2; exit 2; }
    seen[$relative]=1
    targets+=("$relative")
    config_hashes+=("$(sha256sum -- "$repo/$relative/config.json" | cut -d' ' -f1)")
    inode_ids+=("$(stat -c '%d:%i' -- "$repo/$relative")")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$relative" "$method" "$source_name" "$subset" "$old_epochs" "$current_epochs"
done < "$plan"
printf '\nSelected %s run directories. Full keep/remove report: %s\n' "${#targets[@]}" "$plan"
awk -F '\t' '$1 != "REMOVE_LOWER" {n[$1]++} END {for(k in n)printf "%s: %d configurations\n",k,n[k]}' "$plan"
if [[ $mode == --dry-run || ${#targets[@]} == 0 ]]; then
    echo 'Nothing moved or deleted.'
    exit 0
fi

for tool in scontrol squeue flock; do
    command -v "$tool" >/dev/null || { echo "Cannot check active jobs: $tool unavailable." >&2; exit 2; }
done
cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ $cluster == vulcan || $cluster == killarney ]] || { echo "Unsupported cluster: $cluster" >&2; exit 2; }
exec 9>"$bench_dir/.cleanup_lower_epochs.lock"
flock -n 9 || { echo 'Another cleanup process is active.' >&2; exit 2; }

check_active_jobs() {
    local queue id name working command in_repo allowed protected rel
    queue=$(squeue -u "$(id -un)" -h -o '%i|%j|%Z|%o') || return 1
    while IFS='|' read -r id name working command; do
        [[ -n $id ]] || continue
        in_repo=0; allowed=0
        [[ "$working $command" != *"$repo"* && "$working $command" != *'/MKDT/'* ]] || in_repo=1
        # Explicit references to selected output folders always block cleanup.
        for rel in "${targets[@]}"; do
            if [[ "$working/ $command " == *"/$rel/"* || "$working/ $command " == *"/$rel "* ]]; then
                printf 'BLOCKED: active job %s (%s) references %s\n' "$id" "$name" "$rel" >&2
                return 1
            fi
        done
        (( in_repo )) || continue
        # Current dedicated SSL/evaluation roots are never cleanup candidates.
        for protected in \
            random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k \
            kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds \
            mkdt_ssl_checkpoints_cifar8k6k_tiny5k4k_6seeds \
            random_eval_results_current_10seeds_120jobs \
            kmeans_eval_results_current_6seeds_120jobs; do
            [[ "$command " != *"/$protected/"* && "$command " != *"/$protected "* ]] || allowed=1
        done
        if (( ! allowed )); then
            printf 'BLOCKED: cannot rule out legacy-output use by active MKDT-repository job %s (%s).\n' "$id" "$name" >&2
            echo 'Wait for it to finish, then rerun. This script never cancels jobs.' >&2
            return 1
        fi
    done <<< "$queue"
}
check_active_jobs || { echo 'Nothing changed; active-job safety check failed.' >&2; exit 2; }
if [[ $mode == --delete ]]; then
    printf '\nPERMANENT deletion on %s: %s run directories, including their checkpoint files.\n' "$cluster" "${#targets[@]}"
    printf 'Not recoverable by this script. Type DELETE to proceed: '
    IFS= read -r answer || exit 2
    [[ $answer == DELETE ]] || { echo 'Cancelled; nothing deleted.'; exit 0; }
else
    echo 'Trash mode: recoverable rename; this does not free disk quota.'
fi
check_active_jobs || { echo 'Active jobs changed; nothing removed.' >&2; exit 2; }
stamp="$(date -u +%Y%m%dT%H%M%SZ)_${cluster}_$$"
report_dir="$bench_dir/cleanup_reports/$stamp"
mkdir -p "$report_dir"
cp -- "$plan" "$report_dir/plan.tsv"
receipt="$report_dir/actions.tsv"
printf 'action\tpath\tdestination\n' > "$receipt"
trash_root="$bench_dir/cleanup_trash/$stamp"
for i in "${!targets[@]}"; do
    relative=${targets[$i]}
    absolute="$repo/$relative"
    safe_target "$relative" || { echo "Target changed; stopping: $relative" >&2; exit 2; }
    [[ $(stat -c '%d:%i' -- "$absolute") == "${inode_ids[$i]}" && \
       $(sha256sum -- "$absolute/config.json" | cut -d' ' -f1) == "${config_hashes[$i]}" ]] || {
        echo "Target or config changed since preview; stopping: $relative" >&2; exit 2;
    }
    decision=$(jq -r --arg root "$repo" -f "$rules" "$absolute/config.json")
    [[ $decision == REMOVE_LOWER$'\t'"$relative"$'\t'* ]] || { echo 'Eligibility changed; stopping.' >&2; exit 2; }
    # Respect the benchmark worker's existing per-seed lock, when present.
    lock_fd=''
    if [[ -e $absolute/.lock ]]; then
        [[ ! -L $absolute/.lock && -f $absolute/.lock ]] || { echo 'Unsafe lock file.' >&2; exit 2; }
        exec {lock_fd}>"$absolute/.lock"
        flock -n "$lock_fd" || { echo "Active worker lock: $relative; stopping." >&2; exit 2; }
    fi
    if [[ $mode == --trash ]]; then
        destination="$trash_root/$relative"
        [[ ! -e $destination && ! -L $destination ]] || { echo 'Trash destination exists.' >&2; exit 2; }
        mkdir -p -- "${destination%/*}"
        [[ $(stat -c %d -- "$absolute") == "$(stat -c %d -- "${destination%/*}")" ]] || {
            echo 'Refusing a cross-filesystem trash move.' >&2; exit 2;
        }
        mv -- "$absolute" "$destination"
        printf 'TRASH\t%s\t%s\n' "$relative" "$destination" >> "$receipt"
    else
        # The only recursive removal: one resolved, owned, allowlisted run folder.
        rm -rf --one-file-system -- "$absolute"
        [[ ! -e $absolute ]] || { echo "Removal incomplete: $relative" >&2; exit 2; }
        printf 'DELETE\t%s\t-\n' "$relative" >> "$receipt"
    fi
    [[ -z $lock_fd ]] || exec {lock_fd}>&-
done
printf 'Finished %s for %s run directories. Receipt: %s\n' "$mode" "${#targets[@]}" "$receipt"
[[ $mode != --trash ]] || printf 'Recoverable files: %s\n' "$trash_root"
[[ $mode != --delete ]] || echo 'Deleted checkpoint files cannot be recovered by this script.'
