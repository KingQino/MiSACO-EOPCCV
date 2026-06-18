# IRACE tuning for MiSACOMSLS5

This directory tunes MiSACOMSLS5 on all 30 EOPCCV problems. Each IRACE
instance is `F<problem>`, for example `F12`. The random seed supplied by IRACE
is also used as the optimizer seed, so racing samples more than a fixed set of
three seeds.

The target value minimized by IRACE is:

```text
log((candidateBestObjective + tau) / (baselineReference + tau))
```

where `baselineReference` is the geometric mean of the ten baseline results
for that problem, stored in the tracked `baseline_references.csv` file
(computed in `objective + tau` space). This keeps HPC tuning independent of
the ignored `MiSACOMSLS5/stats_01` directory. Negative values are better than
the baseline reference, zero is equal, and positive values are worse. Set
`MISACO_IRACE_BASELINE_AGGREGATE` to `mean` or `median` to use another
reference.

For compatibility, a manually supplied instance such as `F12:3` still uses
seed 3 and compares against baseline seed 3.

## Files

- `instances_train.txt`: all problems F1-F30; IRACE supplies the run seed.
- `instances_validation.txt`: all problems F1-F30 for optional later testing.
- `initial-configurations.txt`: the MSLS5 default, c12, c25, and five earlier
  elites used to warm-start the FE=600 race.
- `parameters.txt`: first-stage tuning space for 10 local-search parameters.
- `scenario.txt`: minimal IRACE scenario.
- `target-runner.sh`: IRACE target runner.
- `RunIRACETrialFromEnv.m`: MATLAB wrapper called by the target runner.

The FE=600 all-instance run is saved as
`irace_allF_dynamic_FE600.Rdata`. Its HPC artifacts go to
`runs_allF_dynamic_FE600/`, `logs_allF_dynamic_FE600/`, and
`costs_allF_dynamic_FE600/`, so earlier tuning runs remain untouched.

## Local smoke test

From the repository root:

```bash
MATLAB_BIN=/Applications/MATLAB_R2024b.app/bin/matlab \
MISACO_IRACE_MAX_FE=120 \
irace/target-runner.sh 1 1 123 F12 \
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
export MISACO_IRACE_BASELINE_FILE=/path/to/irace/baseline_references.csv
export MISACO_IRACE_BASELINE_ROOT=/path/to/MiSACOMSLS5/stats_01
export MISACO_IRACE_BASELINE_AGGREGATE=geometric
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
The current scenario uses `parallel = 96`, `loadBalancing = 1`, and
`blockSize = 1`. Make sure the IRACE job allocation has enough CPU and memory
for 96 simultaneous MATLAB workers. The provided `submit_irace.slurm`
requests 96 CPUs and 3 GB per CPU. Final validation should still run the
selected configurations on F1-F30, seeds 1-10, with the full FE=600 budget.
