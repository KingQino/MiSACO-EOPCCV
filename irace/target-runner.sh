#!/bin/bash
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

if [[ $# -lt 4 ]]; then
    echo "1e6"
    echo "target-runner.sh expects: config_id instance_id seed instance [--param value ...]" >&2
    exit 0
fi

config_id="$1"
instance_id="$2"
irace_seed="$3"
instance_spec="$4"
shift 4

Np="${IRACE_DEFAULT_NP:-60}"
No="${IRACE_DEFAULT_NO:-100}"
LocalStartCount="${IRACE_DEFAULT_LOCAL_START_COUNT:-6}"
LocalElitePool="${IRACE_DEFAULT_LOCAL_ELITE_POOL:-30}"
LocalMinSampleFactor="${IRACE_DEFAULT_LOCAL_MIN_SAMPLE_FACTOR:-5}"
LocalNeighborCount="${IRACE_DEFAULT_LOCAL_NEIGHBOR_COUNT:-30}"
LocalNoveltyWeight="${IRACE_DEFAULT_LOCAL_NOVELTY_WEIGHT:-0.01}"
LocalPredImproveTol="${IRACE_DEFAULT_LOCAL_PRED_IMPROVE_TOL:-0.005}"
LocalSwitchMargin="${IRACE_DEFAULT_LOCAL_SWITCH_MARGIN:-0.08}"
LocalSwitchStartTol="${IRACE_DEFAULT_LOCAL_SWITCH_START_TOL:-0.20}"
LocalAdaptiveWindow="${IRACE_DEFAULT_LOCAL_ADAPTIVE_WINDOW:-12}"
LocalGoodRate="${IRACE_DEFAULT_LOCAL_GOOD_RATE:-0.45}"
LocalBadRate="${IRACE_DEFAULT_LOCAL_BAD_RATE:-0.15}"
LocalStagFE="${IRACE_DEFAULT_LOCAL_STAG_FE:-30}"
LocalHardStagFE="${IRACE_DEFAULT_LOCAL_HARD_STAG_FE:-60}"
LocalRescueStartFrac="${IRACE_DEFAULT_LOCAL_RESCUE_START_FRAC:-0.70}"
LocalRescueStagFE="${IRACE_DEFAULT_LOCAL_RESCUE_STAG_FE:-60}"
LocalRescueBestTol="${IRACE_DEFAULT_LOCAL_RESCUE_BEST_TOL:-0.10}"
LocalRescueStartTol="${IRACE_DEFAULT_LOCAL_RESCUE_START_TOL:-0.35}"
LocalRescueSwitchMargin="${IRACE_DEFAULT_LOCAL_RESCUE_SWITCH_MARGIN:-0.05}"
LocalRescuePredImproveTol="${IRACE_DEFAULT_LOCAL_RESCUE_PRED_IMPROVE_TOL:-0.02}"
LocalRescueMaxFailures="${IRACE_DEFAULT_LOCAL_RESCUE_MAX_FAILURES:-3}"
LocalRescueCooldownFE="${IRACE_DEFAULT_LOCAL_RESCUE_COOLDOWN_FE:-45}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --np) Np="$2"; shift 2 ;;
        --no) No="$2"; shift 2 ;;
        --local-start-count) LocalStartCount="$2"; shift 2 ;;
        --local-elite-pool) LocalElitePool="$2"; shift 2 ;;
        --local-min-sample-factor) LocalMinSampleFactor="$2"; shift 2 ;;
        --local-neighbor-count) LocalNeighborCount="$2"; shift 2 ;;
        --local-novelty-weight) LocalNoveltyWeight="$2"; shift 2 ;;
        --local-pred-improve-tol) LocalPredImproveTol="$2"; shift 2 ;;
        --local-switch-margin) LocalSwitchMargin="$2"; shift 2 ;;
        --local-switch-start-tol) LocalSwitchStartTol="$2"; shift 2 ;;
        --local-adaptive-window) LocalAdaptiveWindow="$2"; shift 2 ;;
        --local-good-rate) LocalGoodRate="$2"; shift 2 ;;
        --local-bad-rate) LocalBadRate="$2"; shift 2 ;;
        --local-stag-fe) LocalStagFE="$2"; shift 2 ;;
        --local-hard-stag-fe) LocalHardStagFE="$2"; shift 2 ;;
        --local-rescue-start-frac) LocalRescueStartFrac="$2"; shift 2 ;;
        --local-rescue-stag-fe) LocalRescueStagFE="$2"; shift 2 ;;
        --local-rescue-best-tol) LocalRescueBestTol="$2"; shift 2 ;;
        --local-rescue-start-tol) LocalRescueStartTol="$2"; shift 2 ;;
        --local-rescue-switch-margin) LocalRescueSwitchMargin="$2"; shift 2 ;;
        --local-rescue-pred-improve-tol) LocalRescuePredImproveTol="$2"; shift 2 ;;
        --local-rescue-max-failures) LocalRescueMaxFailures="$2"; shift 2 ;;
        --local-rescue-cooldown-fe) LocalRescueCooldownFE="$2"; shift 2 ;;
        --*=*)
            key="${1%%=*}"
            value="${1#*=}"
            set -- "$key" "$value" "${@:2}"
            ;;
        *)
            echo "Ignoring unknown IRACE argument: $1" >&2
            shift
            ;;
    esac
done

param_vector="${Np},${No},${LocalStartCount},${LocalElitePool},${LocalMinSampleFactor},${LocalNeighborCount},${LocalNoveltyWeight},${LocalPredImproveTol},${LocalSwitchMargin},${LocalSwitchStartTol},${LocalAdaptiveWindow},${LocalGoodRate},${LocalBadRate},${LocalStagFE},${LocalHardStagFE},${LocalRescueStartFrac},${LocalRescueStagFE},${LocalRescueBestTol},${LocalRescueStartTol},${LocalRescueSwitchMargin},${LocalRescuePredImproveTol},${LocalRescueMaxFailures},${LocalRescueCooldownFE}"

log_dir="${MISACO_IRACE_LOG_DIR:-$repo_root/irace/logs}"
cost_dir="${MISACO_IRACE_COST_DIR:-$repo_root/irace/costs}"
mkdir -p "$log_dir" "$cost_dir"

safe_instance="${instance_spec//[^A-Za-z0-9_.:-]/_}"
log_file="$log_dir/config_${config_id}_instance_${instance_id}_${safe_instance}_seed_${irace_seed}.log"
cost_file="$cost_dir/config_${config_id}_instance_${instance_id}_${safe_instance}_seed_${irace_seed}.txt"

export MISACO_PROJECT_ROOT="$repo_root"
export MISACO_IRACE_CONFIG_ID="$config_id"
export MISACO_IRACE_INSTANCE_ID="$instance_id"
export MISACO_IRACE_SEED="$irace_seed"
export MISACO_IRACE_INSTANCE="$instance_spec"
export MISACO_IRACE_PARAMS="$param_vector"
export MISACO_IRACE_COST_FILE="$cost_file"
export MISACO_IRACE_ALGORITHM="${MISACO_IRACE_ALGORITHM:-MiSACOMSLS5}"
export MISACO_IRACE_MAX_FE="${MISACO_IRACE_MAX_FE:-${MISACO_MAX_FE:-600}}"
export MISACO_IRACE_BASELINE_ROOT="${MISACO_IRACE_BASELINE_ROOT:-$repo_root/MiSACOMSLS5/stats_01}"
export MISACO_IRACE_STATS_ROOT="${MISACO_IRACE_STATS_ROOT:-$repo_root/irace/runs}"
export MISACO_IRACE_TAU="${MISACO_IRACE_TAU:-1e-6}"
export MISACO_PLATEMO_DIR="${MISACO_PLATEMO_DIR:-$repo_root/../PlatEMO/PlatEMO}"

matlab_bin="${MATLAB_BIN:-matlab}"
if [[ -z "${MATLAB_BIN:-}" && -x /Applications/MATLAB_R2024b.app/bin/matlab ]]; then
    matlab_bin="/Applications/MATLAB_R2024b.app/bin/matlab"
fi
platform_name="$(uname -s)"
single_thread="${MISACO_MATLAB_SINGLE_COMP_THREAD:-1}"
if [[ "$platform_name" == "Darwin" && -z "${MISACO_MATLAB_SINGLE_COMP_THREAD:-}" ]]; then
    single_thread=0
fi
if [[ "$platform_name" == "Darwin" ]]; then
    matlab_args=()
else
    matlab_args=(-nodisplay -nosplash -nodesktop)
fi
if [[ "$single_thread" == "1" || "$single_thread" == "true" || "$single_thread" == "yes" ]]; then
    matlab_args+=(-singleCompThread)
fi
matlab_args+=(-batch "addpath(genpath(pwd)); RunIRACETrialFromEnv(); exit")

(
    cd "$repo_root" || exit 1
    "$matlab_bin" "${matlab_args[@]}"
) >"$log_file" 2>&1
status=$?

if [[ $status -ne 0 ]]; then
    echo "MATLAB exited with status $status; see $log_file" >&2
    if [[ -s "$log_file" ]]; then
        echo "Last 80 lines from $log_file:" >&2
        tail -80 "$log_file" >&2
    fi
    echo "1e6"
    exit 0
fi

if [[ ! -s "$cost_file" ]]; then
    echo "Missing cost file $cost_file; see $log_file" >&2
    if [[ -s "$log_file" ]]; then
        echo "Last 80 lines from $log_file:" >&2
        tail -80 "$log_file" >&2
    fi
    echo "1e6"
    exit 0
fi

cost_value="$(tail -n 1 "$cost_file")"
if [[ "$cost_value" == "1000000" || "$cost_value" == "1000000.0" || "$cost_value" == "1e6" ]]; then
    echo "Penalty cost returned; see $log_file" >&2
    if [[ -s "$log_file" ]]; then
        echo "Last 80 lines from $log_file:" >&2
        tail -80 "$log_file" >&2
    fi
fi

echo "$cost_value"
