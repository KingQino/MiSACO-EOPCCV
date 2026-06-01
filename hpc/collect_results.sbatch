#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=misaco-collect
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

# Adjust these lines to match your cluster environment.
module purge
module load matlab/2024a

export MATLAB_PREFDIR="/tmp/$USER/matlab_${SLURM_JOB_ID}_collect"
mkdir -p "$MATLAB_PREFDIR"

export MATLAB_USE_DISPLAY=0
export MATLAB_DISABLE_SERVICEHOST=1
export MISACO_STATS_ROOT="${MISACO_STATS_ROOT:-MiSACO/stats}"

matlab -nodisplay -nosplash -nodesktop -batch "CollectEOPCCVResults(getenv('MISACO_STATS_ROOT'))"
