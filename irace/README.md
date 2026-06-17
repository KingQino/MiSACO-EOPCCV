# IRACE tuning for MiSACOMSLS5

This directory contains a fixed-instance IRACE setup.  Each IRACE instance is
`F<problem>:<seed>`, for example `F12:3`.  The target runner ignores the IRACE
seed for the optimizer seed and instead uses the seed embedded in the instance
string, so each trial can be paired with the existing MiSACOMSLS5 baseline.

The target value minimized by IRACE is:

```text
log((candidateBestObjective + tau) / (baselineBestObjective + tau))
```

where `baselineBestObjective` is read from `MiSACOMSLS5/stats_01`.
Negative values are better than the baseline, zero is equal, and positive
values are worse than the baseline.

## Files

- `instances_train.txt`: training instances, currently 9 selected problems
  with seeds 1-3.
- `instances_validation.txt`: matching validation instances with seeds 4-10.
- `parameters.txt`: first-stage tuning space for 10 local-search parameters.
- `scenario.txt`: minimal IRACE scenario.
- `target-runner.sh`: IRACE target runner.
- `RunIRACETrialFromEnv.m`: MATLAB wrapper called by the target runner.

## Local smoke test

From the repository root:

```bash
MATLAB_BIN=/Applications/MATLAB_R2024b.app/bin/matlab \
MISACO_IRACE_MAX_FE=120 \
irace/target-runner.sh 1 1 123 F12:1 \
  --local-start-count 6 \
  --local-elite-pool 30 \
  --local-min-sample-factor 5 \
  --local-neighbor-count 30 \
  --local-pred-improve-tol 0.005 \
  --local-switch-margin 0.08 \
  --local-switch-start-tol 0.20 \
  --local-good-rate 0.45 \
  --local-bad-rate 0.15 \
  --local-stag-fe 30
```

The command should print one numeric cost.  MATLAB output is written under
`irace/logs/`; run artifacts are written under `irace/runs/`.

## Run IRACE

From the `irace` directory:

```bash
cd irace
Rscript run_irace.R
```

Useful environment overrides:

```bash
export MATLAB_BIN=matlab
export MISACO_PLATEMO_DIR=/path/to/PlatEMO/PlatEMO
export MISACO_IRACE_BASELINE_ROOT=/path/to/MiSACOMSLS5/stats_01
export MISACO_IRACE_STATS_ROOT=/path/to/irace/runs
export MISACO_IRACE_LOG_DIR=/path/to/irace/logs
export MISACO_IRACE_MAX_FE=600
export MISACO_IRACE_TAU=1e-6
export MISACO_MATLAB_SINGLE_COMP_THREAD=1
```

`target-runner.sh` uses the HPC-style `-nodisplay -nosplash -nodesktop`
arguments on Linux.  On macOS it uses only `-batch`, because some local MATLAB
builds exit silently when `-nodisplay` or `-singleCompThread` is combined with
`-batch`.

Increase `maxExperiments` in `scenario.txt` when running a serious tuning job.
The current scenario uses `parallel = 48`, `loadBalancing = 1`, and
`blockSize = 1`.  Make sure the IRACE job allocation has enough CPU and memory
for 48 simultaneous MATLAB workers.  For a first HPC pass, 300-600 experiments
is a reasonable start.  Final validation should still run the selected
configurations on F1-F30, seeds 1-10.
