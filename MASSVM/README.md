# MASSVM

MATLAB implementation of robust preference learning for multi-attribute sorting. The repository contains the FADMM solver with a working-set strategy, a standard ADMM baseline, synthetic and real-data experiment scripts, sensitivity analyses, plotting scripts, helper functions, and the datasets used by the experiments.

## Requirements

- MATLAB
- Statistics and Machine Learning Toolbox (`crossvalind` is used for cross-validation)
- Spreadsheet import support for `figures/gamma_plot.m`

No installation step is required.

## Repository structure

```text
MASSVM/
├── startup.m                 Add all repository folders to the MATLAB path
├── mainfun/                  FADMM and ADMM solvers
├── subfun/                   Data construction, proximal operators, metrics, and utilities
├── Syndata/                  Synthetic-data experiments
├── Realdata/                 Real-data experiments
├── sensitivity/              Sensitivity analyses for gamma and nu
├── figures/                  Loss-function and marginal-value plotting scripts
└── dataset/                  Real datasets used by the experiment scripts
```

## Getting started

Clone or download the repository, open MATLAB, and run:

```matlab
cd('path/to/MASSVM')
run('startup.m')
```

`startup.m` resolves the repository root from its own location, so it can also be called from another working directory.

## Main experiment scripts

| Script | Purpose |
| --- | --- |
| `Syndata/Test_Syndata_FADMM.m` | FADMM experiments on synthetic datasets |
| `Syndata/Robust_Syndata_FADMM.m` | FADMM robustness experiments with label contamination |
| `Realdata/Test_RealData_FADMM.m` | FADMM experiments on a real dataset |
| `Realdata/Test_RealData_ADMM.m` | ADMM baseline experiments on a real dataset |
| `Realdata/Robust_Realdata_FADMM.m` | FADMM robustness experiments on real data |
| `Realdata/Robust_Realdata_ADMM.m` | ADMM robustness experiments on real data |
| `sensitivity/Gamma_Syndata_FADMM.m` | Sensitivity analysis for the number of subintervals |
| `sensitivity/Test_Syndata_FADMM_vary_nu.m` | Sensitivity analysis for the loss threshold `nu` |
| `figures/ccs_plot.m` | Plot the capped concave squared loss |
| `figures/gamma_plot.m` | Plot marginal value functions from `visual_gamma.xlsx` |

The experiment scripts perform parameter selection by 5-fold cross-validation and then repeat the train/test evaluation. Their main settings are defined near the beginning of each script, including sample sizes, attribute counts, class counts, noise ratios, numbers of repetitions, and candidate parameter values.

## Real-data experiments

Each CSV file is expected to contain class labels in the first column and attributes in the remaining columns. The first row is treated as a header by the current experiment scripts.

The default dataset is:

- `dataset/machine_ord.csv` for `Test_RealData_FADMM.m` and `Test_RealData_ADMM.m`
- `dataset/HS.csv` for `Robust_Realdata_FADMM.m` and `Robust_Realdata_ADMM.m`

To use another included dataset, change the `fullfile(project_root, ...)` expression at the beginning of the relevant script.

## Solver interface

The main solver can be called directly with pairwise samples:

```matlab
pars.eta = 1;
pars.rho = 1;
pars.nu = 0.2;
pars.maxit = 2000;
pars.tol = 1e-3;
pars.record_history = true;
pars.force_maxit = false;

out = FADMM(X_pair, y_pair, pars);
```

`X_pair` contains one pairwise sample per row, and `y_pair` contains labels in `{+1,-1}`. Important output fields include:

- `out.w`: learned primal coefficients
- `out.mu`: residual variable
- `out.theta`: dual variable
- `out.iter`: number of iterations
- `out.time`: runtime
- `out.acc`: training accuracy on the pairwise data
- `out.flag`: termination status
- `out.objective_history`: objective values when history recording is enabled
- `out.kkt_residual_history`: maximum normalized KKT residual by iteration
- `out.working_set_size_history`: FADMM working-set size by iteration

`ADMM` uses the same main inputs and parameter structure but does not use the FADMM working set.

## Reproducibility

The experiments use random sampling, random train/test splits, and synthetic data generation. Set the MATLAB random-number seed before running a script when reproducible splits are required:

```matlab
rng(1)
```

Pairwise expansion can create substantially more training samples than the number of original alternatives. Large configurations may therefore require considerable memory and runtime.

## Plotting note

`figures/gamma_plot.m` expects `visual_gamma.xlsx` in the repository root and reads the worksheet `m4_f4`. This workbook is not currently included. Add the workbook before running that script, or change the input path and worksheet name in the script.

## Citation

If you use this code in academic work, please cite the associated paper:

``Xie, Boyi and Wu, Zhongming and Xu, Haiwen and Li, Min, Robust preference learning with multiple potentially non-monotonic attributes for sorting problems. 
Available at SSRN: https://ssrn.com/abstract=6853604 or http://dx.doi.org/10.2139/ssrn.6853604''

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
