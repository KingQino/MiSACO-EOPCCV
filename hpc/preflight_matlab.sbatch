#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=matlab-preflight
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

set -euo pipefail

module purge
module load "${MATLAB_MODULE:-matlab/2024a}"

export MATLAB_PREFDIR="/tmp/$USER/matlab_${SLURM_JOB_ID}_preflight"
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

matlab -nodisplay -nosplash -nodesktop -singleCompThread -batch "fprintf('MATLAB preflight OK\\n'); exit"
