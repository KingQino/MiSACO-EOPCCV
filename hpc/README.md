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
MAX_PARALLEL=100 \
MEM_PER_CPU=4G \
bash hpc/submit_array.sh
```

With the defaults, up to 100 seed tasks run at once and each task requests
1 CPU core and 4G memory per CPU. This is seed-level parallelism: one array
task runs one `(problem, seed)` pair. Set `MAX_PARALLEL` higher if the
cluster allocation permits, or set `MEM_PER_CPU=5G`/`6G` if your cluster
reports MATLAB memory pressure. The submit script caps requested concurrency
at `MAX_PARALLEL_CAP=100` by default. Worker jobs request
`--ntasks-per-node=32` and a two-hour wall time by default. Override the
per-node task hint with `NTASKS_PER_NODE=...` if your cluster recommends a
different packing level.

To run the AGES variant instead of the MiSACO baseline:

```bash
MISACO_ALGORITHM=AGES \
STATS_NAME=stats_01 \
MAX_PARALLEL=100 \
MEM_PER_CPU=4G \
bash hpc/submit_array.sh
```

Recommended MiSACOBasin command:

```bash
MISACO_ALGORITHM=MiSACOBasin \
STATS_NAME=stats_01 \
MAX_PARALLEL=100 \
MEM_PER_CPU=4G \
bash hpc/submit_array.sh
```

Use a new `STATS_NAME` for each experiment batch, for example `stats_01`,
`stats_02`, or `stats_rastrigin_ablation`. By default, results go to
`<Algorithm>/<STATS_NAME>`. To intentionally use a custom path, set both
`MISACO_STATS_ROOT=/path/to/results` and `MISACO_USE_CUSTOM_STATS_ROOT=1`.

The submit script runs a small MATLAB preflight job first. If MATLAB fails
while installing or starting MathWorks Service Host, the array job will not
start. This is intentional: it catches cluster-side MATLAB startup issues
before hundreds of seed tasks are submitted. Set `SUBMIT_PREFLIGHT=0` only
after the MATLAB module is known to be stable.

Recommended AGES command:

```bash
MISACO_ALGORITHM=AGES \
STATS_NAME=stats_02 \
MAX_PARALLEL=100 \
MEM_PER_CPU=4G \
bash hpc/submit_array.sh
```

Recommended AGESV2 command:

```bash
MISACO_ALGORITHM=AGESV2 \
STATS_NAME=stats_01 \
MAX_PARALLEL=100 \
MEM_PER_CPU=4G \
bash hpc/submit_array.sh
```

Recommended AGESV3 command:

```bash
MISACO_ALGORITHM=AGESV3 \
STATS_NAME=stats_01 \
MAX_PARALLEL=100 \
MEM_PER_CPU=4G \
bash hpc/submit_array.sh
```

Recommended AGESV4 command:

```bash
MISACO_ALGORITHM=AGESV4 \
STATS_NAME=stats_01 \
MAX_PARALLEL=100 \
MEM_PER_CPU=4G \
bash hpc/submit_array.sh
```

Download results without creating an extra nested `stats_01` directory by
using a trailing slash on the source path:

```bash
rsync -av --progress \
  exx866@login.hpc.qmul.ac.uk:/data/home/exx866/Matlab/MiSACO-EOPCCV/AGESV4/stats_01/ \
  ./AGESV4/stats_01/
```

The submit script also starts a dependent collect job that writes:

- `<Algorithm>/<STATS_NAME>/general.csv`
- `<Algorithm>/<STATS_NAME>/general.mat`
- `<Algorithm>/<STATS_NAME>/general.txt`

By default, each seed keeps compact outputs plus `history.*.mat`, which
contains the full `History`, `EvalLog`, `SurLog`, and `RunSummary` structs.
Set `MISACO_SAVE_DETAILED_LOGS=1` to also write `eval_log.*.csv`,
`surrogate_log.*.csv`, and `result.*.mat` files for every seed.
