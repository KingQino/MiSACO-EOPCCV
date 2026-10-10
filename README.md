# Expensive mixed-variable optimization problem

We propose **ASMLS-MiSACO**, an adaptive surrogate-guided multi-start local-search enhancement of MiSACO for expensive mixed-variable optimization problems involving continuous and categorical decision variables.

Our algorithm achieved **3rd place in the Traditional Evolutionary Algorithms track** of the [IEEE WCCI 2026 Competition on Expensive Mixed-Variable Optimization](https://liuj-2023.github.io/WCCI2026-Competition/).

## Benchmark

The competition uses the EOPCCV benchmark suite, consisting of 30 expensive mixed-variable optimization problems:

- **EOPCCV-F1–F10:** 8 continuous + 2 categorical variables
- **EOPCCV-F11–F20:** 5 continuous + 5 categorical variables
- **EOPCCV-F21–F30:** 2 continuous + 8 categorical variables

## Experimental Results

ASMLS-MiSACO was evaluated on all 30 EOPCCV benchmark problems.

The complete experimental results are available in [`statistics.csv`](statistics.csv).
