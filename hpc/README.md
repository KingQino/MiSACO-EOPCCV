# MiSACO HPC Jobs

Submit all 30 EOPCCV instances as a Slurm array:

```bash
cd /path/to/MiSACO-and-EOPCCV
bash hpc/submit_array.sh
```

Each array task runs one problem (`F1` to `F30`) for seeds `1..10`.
Results are written under `MiSACO/stats/F*/seed/`.

Useful overrides:

```bash
MISACO_PLATEMO_DIR=/path/to/PlatEMO/PlatEMO \
MISACO_MAX_FE=600 \
MISACO_MAX_TRIALS=10 \
MAX_PARALLEL=30 \
bash hpc/submit_array.sh
```

The submit script also starts a dependent collect job that writes:

- `MiSACO/stats/general.csv`
- `MiSACO/stats/general.mat`
- `MiSACO/stats/general.txt`

By default, each seed keeps compact outputs plus `history.*.mat`, which
contains the full `History`, `EvalLog`, `SurLog`, and `RunSummary` structs.
Set `MISACO_SAVE_DETAILED_LOGS=1` to also write `eval_log.*.csv`,
`surrogate_log.*.csv`, and `result.*.mat` files for every seed.
