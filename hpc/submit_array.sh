#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-$repo_root/logs}"
max_parallel="${MAX_PARALLEL:-30}"
mem_per_cpu="${MEM_PER_CPU:-3G}"
stats_name="${STATS_NAME:-stats}"
sbatch_script="$repo_root/hpc/run_instance_array.sbatch"
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
if [[ -z "${MISACO_STATS_ROOT:-}" ]]; then
    export MISACO_STATS_ROOT="$repo_root/$MISACO_ALGORITHM/$stats_name"
else
    export MISACO_STATS_ROOT
fi
export MISACO_PLATEMO_DIR="${MISACO_PLATEMO_DIR:-$repo_root/../PlatEMO/PlatEMO}"
export MISACO_ARRAY_MODE="${MISACO_ARRAY_MODE:-seed}"

if [[ ! -d "$MISACO_PLATEMO_DIR" ]]; then
    echo "PlatEMO directory not found: $MISACO_PLATEMO_DIR" >&2
    echo "Set MISACO_PLATEMO_DIR=/path/to/PlatEMO/PlatEMO and resubmit." >&2
    exit 1
fi

if [[ "$MISACO_ARRAY_MODE" == "seed" ]]; then
    array_max=$((30 * MISACO_MAX_TRIALS - 1))
else
    array_max=29
fi

array_job_id=$(sbatch \
    --parsable \
    --array="0-$array_max%$max_parallel" \
    --job-name="$job_name" \
    --cpus-per-task=1 \
    --mem-per-cpu="$mem_per_cpu" \
    --output="$log_dir/%x-%A_%a.out" \
    --chdir="$repo_root" \
    "$sbatch_script")

echo "Submitted $MISACO_ALGORITHM array job: $array_job_id"
echo "  Problems: F1-F30"
echo "  Seeds per problem: $MISACO_MAX_TRIALS"
echo "  Max FE: $MISACO_MAX_FE"
echo "  Algorithm: $MISACO_ALGORITHM"
echo "  Stats name: $stats_name"
echo "  Array mode: $MISACO_ARRAY_MODE"
echo "  Max parallel array tasks: $max_parallel"
echo "  CPUs per array task: 1"
echo "  Memory per CPU: $mem_per_cpu"
echo "  Detailed per-seed logs: $MISACO_SAVE_DETAILED_LOGS"
echo "  Stats root: $MISACO_STATS_ROOT"

if [[ "${SUBMIT_COLLECT:-1}" == "1" ]]; then
    collect_job_id=$(sbatch \
        --parsable \
        --dependency="afterany:$array_job_id" \
        --job-name="${job_name}-collect" \
        --output="$log_dir/%x-%j.out" \
        --chdir="$repo_root" \
        "$collect_script")
    echo "Submitted collect job: $collect_job_id"
fi
