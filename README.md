# Expensive mixed-variable optimization problem

We propose **ASMLS-MiSACO**, an adaptive surrogate-guided multi-start local-search enhancement of MiSACO for expensive mixed-variable optimization problems involving continuous and categorical decision variables.

Our algorithm achieved **3rd place in the Traditional Evolutionary Algorithms track** of the [IEEE WCCI 2026 Competition on Expensive Mixed-Variable Optimization](https://liuj-2023.github.io/WCCI2026-Competition/).

## Benchmark

The competition uses the EOPCCV benchmark suite, consisting of 30 expensive mixed-variable optimization problems:

- **EOPCCV-F1–F10:** 8 continuous + 2 categorical variables
- **EOPCCV-F11–F20:** 5 continuous + 5 categorical variables
- **EOPCCV-F21–F30:** 2 continuous + 8 categorical variables

## Implementation

ASMLS-MiSACO is implemented in MATLAB based on [PlatEMO](https://github.com/BIMK/PlatEMO).

## Reference

J. Liu, Y. Wang, G. Sun, and T. Pang, "Multisurrogate-Assisted Ant Colony Optimization for Expensive Optimization Problems With Continuous and Categorical Variables," *IEEE Transactions on Cybernetics*, vol. 52, no. 11, pp. 11348–11361, 2022.

DOI: [10.1109/TCYB.2021.3064676](https://doi.org/10.1109/TCYB.2021.3064676)
