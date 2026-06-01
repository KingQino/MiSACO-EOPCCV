#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=misaco
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G

set -euo pipefail

# Adjust these lines to match your cluster environment.
module purge
module load "${MATLAB_MODULE:-matlab/2024a}"

export MATLAB_PREFDIR="/tmp/$USER/matlab_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MATLAB_PREFDIR"
export XDG_CACHE_HOME="$MATLAB_PREFDIR/xdg-cache"
export XDG_CONFIG_HOME="$MATLAB_PREFDIR/xdg-config"
export XDG_RUNTIME_DIR="$MATLAB_PREFDIR/xdg-runtime"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MATLAB_USE_DISPLAY=0
export MATLAB_DISABLE_SERVICEHOST=1
export MATHWORKS_SERVICEHOST_DISABLE=1

export MISACO_ARRAY_BASE="${MISACO_ARRAY_BASE:-0}"
export MISACO_MAX_TRIALS="${MISACO_MAX_TRIALS:-10}"
export MISACO_MAX_FE="${MISACO_MAX_FE:-600}"
export MISACO_PARENT_N="${MISACO_PARENT_N:-60}"
export MISACO_OFFSPRING_N="${MISACO_OFFSPRING_N:-100}"
export MISACO_SAVE_DETAILED_LOGS="${MISACO_SAVE_DETAILED_LOGS:-0}"
export MISACO_ALGORITHM="${MISACO_ALGORITHM:-MiSACO}"
export STATS_NAME="${STATS_NAME:-stats}"
export MISACO_STATS_ROOT="${MISACO_STATS_ROOT:-${MISACO_ALGORITHM}/${STATS_NAME}}"
export MISACO_PLATEMO_DIR="${MISACO_PLATEMO_DIR:-../PlatEMO/PlatEMO}"
export MISACO_ARRAY_MODE="${MISACO_ARRAY_MODE:-seed}"

echo "Working directory: $PWD"
echo "MISACO_ALGORITHM=$MISACO_ALGORITHM"
echo "STATS_NAME=$STATS_NAME"
echo "MISACO_STATS_ROOT=$MISACO_STATS_ROOT"
echo "MISACO_PLATEMO_DIR=$MISACO_PLATEMO_DIR"

start_delay="${START_DELAY_SECONDS:-90}"
if [[ "$start_delay" =~ ^[0-9]+$ ]] && (( start_delay > 0 )); then
    sleep $(( SLURM_ARRAY_TASK_ID % start_delay ))
fi

matlab -nodisplay -nosplash -nodesktop -singleCompThread -batch "RunEOPCCVHPC()"
