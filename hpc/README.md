# MiSACO HPC Jobs

Submit all 30 EOPCCV instances as a Slurm array:

```bash
cd /path/to/MiSACO-and-EOPCCV
bash hpc/submit_array.sh
```

By default, each array task runs one `(problem, seed)` pair. For 30
problems and 10 seeds, this submits 300 array tasks. Results are written
under `<Algorithm>/<STATS_NAME>/F*/seed/`; by default this is
`MiSACO/stats/F*/seed/`.

Useful overrides:

```bash
MISACO_PLATEMO_DIR=/path/to/PlatEMO/PlatEMO \
MISACO_MAX_FE=600 \
MISACO_MAX_TRIALS=10 \
STATS_NAME=stats_01 \
MAX_PARALLEL=30 \
MEM_PER_CPU=3G \
bash hpc/submit_array.sh
```

With the defaults, up to 30 seed tasks run at once and each task requests
1 CPU core and 3G memory per CPU. This is seed-level parallelism: one array
task runs one `(problem, seed)` pair. Set `MAX_PARALLEL` higher if the
cluster allocation permits, or set `MEM_PER_CPU=4G`/`5G` if your cluster
reports MATLAB memory pressure.

To run the AGES variant instead of the MiSACO baseline:

```bash
MISACO_ALGORITHM=AGES \
STATS_NAME=stats_01 \
MAX_PARALLEL=30 \
MEM_PER_CPU=3G \
bash hpc/submit_array.sh
```

Use a new `STATS_NAME` for each experiment batch, for example `stats_01`,
`stats_02`, or `stats_rastrigin_ablation`. If `MISACO_STATS_ROOT` is set,
it overrides `<Algorithm>/<STATS_NAME>` completely.

The submit script also starts a dependent collect job that writes:

- `<Algorithm>/<STATS_NAME>/general.csv`
- `<Algorithm>/<STATS_NAME>/general.mat`
- `<Algorithm>/<STATS_NAME>/general.txt`

By default, each seed keeps compact outputs plus `history.*.mat`, which
contains the full `History`, `EvalLog`, `SurLog`, and `RunSummary` structs.
Set `MISACO_SAVE_DETAILED_LOGS=1` to also write `eval_log.*.csv`,
`surrogate_log.*.csv`, and `result.*.mat` files for every seed.
