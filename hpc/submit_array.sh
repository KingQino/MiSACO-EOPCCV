#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-$repo_root/logs}"
requested_max_parallel="${MAX_PARALLEL:-30}"
max_parallel_cap="${MAX_PARALLEL_CAP:-100}"
mem_per_cpu="${MEM_PER_CPU:-3G}"
stats_name="${STATS_NAME:-stats}"
sbatch_script="$repo_root/hpc/run_instance_array.sbatch"
preflight_script="$repo_root/hpc/preflight_matlab.sbatch"
collect_script="$repo_root/hpc/collect_results.sbatch"

mkdir -p "$log_dir"

export MISACO_ARRAY_BASE="${MISACO_ARRAY_BASE:-0}"
export MISACO_MAX_TRIALS="${MISACO_MAX_TRIALS:-10}"
export MISACO_MAX_FE="${MISACO_MAX_FE:-600}"
export MISACO_PARENT_N="${MISACO_PARENT_N:-60}"
export MISACO_OFFSPRING_N="${MISACO_OFFSPRING_N:-100}"
export MISACO_SAVE_DETAILED_LOGS="${MISACO_SAVE_DETAILED_LOGS:-0}"
export MISACO_ALGORITHM="${MISACO_ALGORITHM:-MiSACO}"
export STATS_NAME="$stats_name"
job_name="${JOB_NAME:-$MISACO_ALGORITHM}"
existing_stats_root="${MISACO_STATS_ROOT:-}"
if [[ -n "$existing_stats_root" && "${MISACO_USE_CUSTOM_STATS_ROOT:-0}" == "1" ]]; then
    export MISACO_STATS_ROOT="$existing_stats_root"
else
    if [[ -n "$existing_stats_root" ]]; then
        echo "Ignoring inherited MISACO_STATS_ROOT=$existing_stats_root" >&2
        echo "Set MISACO_USE_CUSTOM_STATS_ROOT=1 to submit with a custom stats root." >&2
    fi
    export MISACO_STATS_ROOT="$repo_root/$MISACO_ALGORITHM/$stats_name"
fi
export MISACO_PLATEMO_DIR="${MISACO_PLATEMO_DIR:-$repo_root/../PlatEMO/PlatEMO}"
export MISACO_ARRAY_MODE="${MISACO_ARRAY_MODE:-seed}"
export MATLAB_MODULE="${MATLAB_MODULE:-matlab/2024a}"
export START_DELAY_SECONDS="${START_DELAY_SECONDS:-90}"

if [[ ! -d "$MISACO_PLATEMO_DIR" ]]; then
    echo "PlatEMO directory not found: $MISACO_PLATEMO_DIR" >&2
    echo "Set MISACO_PLATEMO_DIR=/path/to/PlatEMO/PlatEMO and resubmit." >&2
    exit 1
fi

algorithm_file="$repo_root/$MISACO_ALGORITHM/$MISACO_ALGORITHM.m"
if [[ ! -f "$algorithm_file" ]]; then
    echo "Algorithm file not found: $algorithm_file" >&2
    echo "Make sure the $MISACO_ALGORITHM directory has been uploaded before submitting." >&2
    exit 1
fi

mkdir -p "$MISACO_STATS_ROOT"

if [[ "$MISACO_ARRAY_MODE" == "seed" ]]; then
    array_max=$((30 * MISACO_MAX_TRIALS - 1))
else
    array_max=29
fi

if (( requested_max_parallel > max_parallel_cap )); then
    echo "Requested MAX_PARALLEL=$requested_max_parallel exceeds MAX_PARALLEL_CAP=$max_parallel_cap; using $max_parallel_cap." >&2
    max_parallel="$max_parallel_cap"
else
    max_parallel="$requested_max_parallel"
fi

dependency_args=()
if [[ "${SUBMIT_PREFLIGHT:-1}" == "1" ]]; then
    preflight_job_id=$(sbatch \
        --parsable \
        --job-name="${job_name}-preflight" \
        --cpus-per-task=1 \
        --mem-per-cpu="$mem_per_cpu" \
        --output="$log_dir/%x-%j.out" \
        --chdir="$repo_root" \
        --export=ALL \
        "$preflight_script")
    dependency_args=(--dependency="afterok:$preflight_job_id")
    echo "Submitted MATLAB preflight job: $preflight_job_id"
fi

array_job_id=$(sbatch \
    --parsable \
    "${dependency_args[@]}" \
    --array="0-$array_max%$max_parallel" \
    --job-name="$job_name" \
    --cpus-per-task=1 \
    --mem-per-cpu="$mem_per_cpu" \
    --output="$log_dir/%x-%A_%a.out" \
    --chdir="$repo_root" \
    --export=ALL \
    "$sbatch_script")

echo "Submitted $MISACO_ALGORITHM array job: $array_job_id"
echo "  Problems: F1-F30"
echo "  Seeds per problem: $MISACO_MAX_TRIALS"
echo "  Max FE: $MISACO_MAX_FE"
echo "  Algorithm: $MISACO_ALGORITHM"
echo "  Stats name: $stats_name"
echo "  Array mode: $MISACO_ARRAY_MODE"
echo "  Requested max parallel array tasks: $requested_max_parallel"
echo "  Max parallel cap: $max_parallel_cap"
echo "  Effective max parallel array tasks: $max_parallel"
echo "  CPUs per array task: 1"
echo "  Memory per CPU: $mem_per_cpu"
echo "  MATLAB module: $MATLAB_MODULE"
echo "  Startup stagger window: ${START_DELAY_SECONDS}s"
echo "  Detailed per-seed logs: $MISACO_SAVE_DETAILED_LOGS"
echo "  Stats root: $MISACO_STATS_ROOT"

if [[ "${SUBMIT_COLLECT:-1}" == "1" ]]; then
    collect_job_id=$(sbatch \
        --parsable \
        --dependency="afterany:$array_job_id" \
        --job-name="${job_name}-collect" \
        --output="$log_dir/%x-%j.out" \
        --chdir="$repo_root" \
        --export=ALL \
        "$collect_script")
    echo "Submitted collect job: $collect_job_id"
fi
