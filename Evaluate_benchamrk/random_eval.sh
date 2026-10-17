#!/bin/bash
# Like random_ssl.sh: run with bash; directly submit ordinary Slurm jobs.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 6 ]] || { echo 'Invalid worker arguments' >&2; exit 2; }
    : "${SLURM_JOB_ID:?The worker must run inside a Slurm GPU job}"
    bench_dir=$1 task=$2 venv=$3 data_root=$4 checkpoint_root=$5 output_root=$6
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$output_root/cache/hf" XDG_CACHE_HOME="$output_root/cache/xdg"
    export TORCH_HOME="$output_root/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    exec python "$bench_dir/slurm/random_eval/worker.py" \
        "$task" "$data_root" "$checkpoint_root" "$output_root"
fi

case ${1:---submit} in
    --submit) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --help|-h)
        printf '%s\n' \
            'bash Evaluate_benchamrk/random_eval.sh            # submit ready work across 40 jobs' \
            'bash Evaluate_benchamrk/random_eval.sh --dry-run  # print the jobs' \
            'Same direct Bash/sbatch workflow as random_ssl.sh; uses this checkout and $HOME/ENV.' \
            '38 ready jobs + 1 after 6334332/6334333 + 1 after 6334334/6334335.' \
            '540 encoders; 3240 main/downstream probes; ten seeds; 1% and 5% labels.' \
            '1 GPU, 1 CPU, 6G per job. BENCH_EVAL_TIME=06:00:00; BENCH_EVAL_TAIL_TIME=12:00:00.' \
            'BENCH_VENV, BENCH_DATA_ROOT, BENCH_CHECKPOINT_ROOT and GPU_TYPE may override defaults.' \
            'Results/logs/job IDs: Evaluate_benchamrk/random_eval_results.' \
            'Missing seeds are skipped and listed in missing_checkpoints.tsv; ready seeds still run.' \
            'Re-run this command to resume: completed/live jobs are skipped; encoders are retained.'
        exit 0 ;;
    *) echo 'Use --submit (default), --dry-run, or --help' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Unexpected extra arguments' >&2; exit 2; }
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the login node, not sbatch.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/random_eval.sh"
manifest="$bench_dir/slurm/random_eval/tasks.tsv"
output_root="$bench_dir/random_eval_results"
checkpoint_root=${BENCH_CHECKPOINT_ROOT:-$bench_dir/random_ssl_checkpoints}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
venv=${BENCH_VENV:-$HOME/ENV}
wang=${BENCH_ACCOUNT:-aip-boyuwang}
lu=${BENCH_SECONDARY_ACCOUNT:-aip-yiweilu}
for required in "$manifest" "$bench_dir/slurm/random_eval/worker.py" \
    "$bench_dir/slurm/random_eval/report.py" "$bench_dir/table_coverage.tsv" \
    "$bench_dir/evaluate.py" "$bench_dir/protocol.json"; do
    [[ -r $required ]] || { echo "Missing $required; sync the code first." >&2; exit 2; }
done

declare -A counts probes dependencies active ready waiting missing_by_task
while IFS=$'\t' read -r task source_name model subset ssl seed; do
    [[ $task != task ]] || continue
    counts[$task]=$((${counts[$task]:-0} + 1))
    n=2
    [[ $model != ResNet18 ]] || n=14
    probes[$task]=$((${probes[$task]:-0} + n))
done < "$manifest"
[[ ${#counts[@]} == 40 ]] || { echo 'Expected 40 tasks in tasks.tsv' >&2; exit 2; }
if (( dry_run )); then
    gpu_type=${GPU_TYPE:-'<discovered-at-submit>'}
    dependencies[39]='6334332:6334333'
    dependencies[40]='6334334:6334335'
else
    [[ -r $venv/bin/activate && -d $data_root && -d $checkpoint_root ]] || {
        echo 'Set BENCH_VENV, BENCH_DATA_ROOT, BENCH_CHECKPOINT_ROOT to the existing Killarney paths.' >&2; exit 2;
    }
    venv=$(cd -- "$venv" && pwd -P)
    data_root=$(cd -- "$data_root" && pwd -P)
    checkpoint_root=$(cd -- "$checkpoint_root" && pwd -P)
    module spider python/3.11.5 >/dev/null 2>&1
    sinfo -s
    inventory=$(sinfo -h -o '%G' --Node)
    gpu_types=$(printf '%s\n' "$inventory" | tr ',' '\n' | sed -nE 's/^gpu:([^:]+):.*/\1/p' | sort -u)
    gpu_type=${GPU_TYPE:-}
    if [[ -z $gpu_type ]]; then
        if grep -qx l40s <<< "$gpu_types"; then gpu_type=l40s
        elif grep -qx h100 <<< "$gpu_types"; then gpu_type=h100
        else echo "Set GPU_TYPE to a discovered type: $gpu_types" >&2; exit 2; fi
    fi
    grep -Fxq -- "$gpu_type" <<< "$gpu_types" || { echo "GPU type not available: $gpu_type" >&2; exit 2; }
    accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
    for account in "$wang" "$lu"; do
        printf '%s\n' "$accounts" | cut -d'|' -f1 | grep -Fxq -- "$account" || {
            echo "This account is not available here: $account" >&2; exit 2;
        }
    done
    queue=$(squeue -u "$USER" -h -o '%i|%j|%T')
    cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
    while IFS='|' read -r id name state; do
        [[ -n $id ]] || continue
        active[$id]="$name"
        active[$name]="$id"
    done <<< "$queue"
    mkdir -p "$output_root/logs"
    missing_log="$output_root/missing_checkpoints.tsv"
    printf 'task\tsource\tmodel\tsubset\tssl\tseed\tmissing_files\tstatus\tcheckpoint_dir\n' > "$missing_log"
    # Classify each seed independently: missing training output must not prevent
    # evaluation of other complete encoders, including those in the same job.
    missing=0; available=0; awaiting=0
    while IFS=$'\t' read -r task source_name model subset ssl seed; do
        [[ $task != task ]] || continue
        [[ ! -s $output_root/jobs/$task.json && -z ${active[random_eval_$task]:-} ]] || continue
        printf -v seed_name 'seed_%03d' "$seed"
        folder="$checkpoint_root/$source_name/$model/random/pt$subset/$ssl/$seed_name"
        if [[ -s $folder/encoder.pt && -s $folder/config.json && -s $folder/ssl_complete.json ]]; then
            ready[$task]=$((${ready[$task]:-0} + 1))
            available=$((available + 1))
            continue
        fi
        missing_files=''
        for file in encoder.pt config.json ssl_complete.json; do
            [[ -s $folder/$file ]] || missing_files="${missing_files:+$missing_files,}$file"
        done
        status=missing
        parent=''
        case "$task/$ssl" in
            39/simclr) parent=6334332 ;; 39/barlowtwins) parent=6334333 ;;
            40/simclr) parent=6334334 ;; 40/barlowtwins) parent=6334335 ;;
        esac
        if [[ -n $parent && $cluster == killarney && \
              ${active[$parent]:-} == "random_TinyImageNet_ResNet18_${subset}_${ssl}" ]]; then
            waiting[$task]=$((${waiting[$task]:-0} + 1))
            awaiting=$((awaiting + 1))
            status="waiting_for_$parent"
            case ":${dependencies[$task]:-}:" in
                *":$parent:"*) ;;
                *) dependencies[$task]="${dependencies[$task]:+${dependencies[$task]}:}$parent" ;;
            esac
        else
            missing=$((missing + 1))
            missing_by_task[$task]=$((${missing_by_task[$task]:-0} + 1))
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$source_name" "$model" \
            "$subset" "$ssl" "$seed" "$missing_files" "$status" "$folder" >> "$missing_log"
    done < "$manifest"
    printf '%s ready encoders; %s waiting for active training; %s missing seeds will be skipped.\n' "$available" "$awaiting" "$missing"
    printf 'Missing checkpoint details: %s\n' "$missing_log"
    submission_log="$output_root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
    printf 'job_id\taccount\ttask\tencoders\tprobes\twalltime\tdependency\n' > "$submission_log"
fi

printf 'Random evaluation: 40 jobs / 540 encoders / 3240 probes.\n'
printf 'task\taccount\tplanned_encoders\tplanned_probes\twalltime\tdependency\n'
submitted=0
for task in {01..40}; do
    account=$wang
    case $task in 10|20|30|38) account=$lu ;; esac
    walltime=${BENCH_EVAL_TIME:-06:00:00}
    [[ $task != 39 && $task != 40 ]] || walltime=${BENCH_EVAL_TAIL_TIME:-12:00:00}
    dependency=${dependencies[$task]:--}
    # Evaluate the saved seeds even if a parent ends in TIMEOUT/FAILED.
    [[ $dependency == - ]] || dependency="afterany:$dependency"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$account" "${counts[$task]}" "${probes[$task]}" "$walltime" "$dependency"
    if (( ! dry_run )); then
        if [[ -s $output_root/jobs/$task.json || -n ${active[random_eval_$task]:-} ]]; then
            printf 'SKIP task %s: already completed or active.\n' "$task"
            continue
        fi
        if (( ${ready[$task]:-0} + ${waiting[$task]:-0} == 0 )); then
            printf 'SKIP task %s: no ready checkpoint or active training dependency (%s missing seeds).\n' \
                "$task" "${missing_by_task[$task]:-0}"
            continue
        fi
        args=()
        [[ $dependency == - ]] || args+=("--dependency=$dependency" --kill-on-invalid-dep=yes)
        job_id=$(sbatch --parsable --job-name="random_eval_$task" --account="$account" \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G \
            --gres="gpu:$gpu_type:1" --time="$walltime" --chdir="$bench_dir" \
            --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
            "${args[@]}" "$launcher" --worker "$bench_dir" "$task" "$venv" \
            "$data_root" "$checkpoint_root" "$output_root")
        job_id=${job_id%%;*}
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job_id" "$account" "$task" \
            "${counts[$task]}" "${probes[$task]}" "$walltime" "$dependency" >> "$submission_log"
        printf 'Submitted job %s\n' "$job_id"
        submitted=$((submitted + 1))
    fi
done
if (( dry_run )); then
    printf 'Dry run only: no jobs submitted and no files written.\n'
else
    printf 'Submitted %s jobs. IDs: %s\nResults: %s\n' "$submitted" "$submission_log" "$output_root"
fi
