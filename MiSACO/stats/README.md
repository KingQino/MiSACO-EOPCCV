# MiSACO Statistics Output

Expected layout after HPC runs:

```text
MiSACO/stats/
  F1/
    1/
      history.F1.seed1.mat
      evols.F1.csv
      solution.F1.txt
      summary.F1.seed1.txt
    ...
    10/
    stats.F1.csv
    stats.F1.mat
    stats.F1.txt
  ...
  F30/
  general.csv
  general.mat
  general.txt
```

`general.csv` is the compact all-instance summary.

`history.*.mat` contains the full `History`, `EvalLog`, `SurLog`, and
`RunSummary` structs. The duplicate detailed files `eval_log.*.csv`,
`surrogate_log.*.csv`, and `result.*.mat` are written only when
`MISACO_SAVE_DETAILED_LOGS=1`.
