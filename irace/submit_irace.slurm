#!/bin/bash

# Slurm job options
#SBATCH --partition=compute
#SBATCH --job-name=MiSACOMSLS5-IRACE
#SBATCH --output=/data/home/exx866/Matlab/MiSACO-EOPCCV/irace/irace.log
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=3G

set -euo pipefail

module load gcc openmpi cmake R/4.4.1 bc matlab

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export IRACE_HOME="/data/home/exx866/R/x86_64-pc-linux-gnu-library/4.4/irace"
export PATH="$IRACE_HOME/bin:$PATH"

export MATLAB_BIN="${MATLAB_BIN:-matlab}"
export MISACO_MATLAB_SINGLE_COMP_THREAD="${MISACO_MATLAB_SINGLE_COMP_THREAD:-1}"
export MISACO_PLATEMO_DIR="${MISACO_PLATEMO_DIR:-/data/home/exx866/Matlab/PlatEMO/PlatEMO}"
export MISACO_IRACE_BASELINE_ROOT="${MISACO_IRACE_BASELINE_ROOT:-/data/home/exx866/Matlab/MiSACO-EOPCCV/MiSACOMSLS5/stats_01}"
export MISACO_IRACE_STATS_ROOT="${MISACO_IRACE_STATS_ROOT:-/data/home/exx866/Matlab/MiSACO-EOPCCV/irace/runs}"
export MISACO_IRACE_LOG_DIR="${MISACO_IRACE_LOG_DIR:-/data/home/exx866/Matlab/MiSACO-EOPCCV/irace/logs}"
export MISACO_IRACE_COST_DIR="${MISACO_IRACE_COST_DIR:-/data/home/exx866/Matlab/MiSACO-EOPCCV/irace/costs}"
export MISACO_IRACE_MAX_FE="${MISACO_IRACE_MAX_FE:-600}"
export MISACO_IRACE_TAU="${MISACO_IRACE_TAU:-1e-6}"

cd /data/home/exx866/Matlab/MiSACO-EOPCCV/irace
mkdir -p logs costs runs

Rscript run_irace.R
