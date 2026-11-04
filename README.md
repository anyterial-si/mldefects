# Prescreening Point Defects in Semiconductors With Machine Learning

Code, data and results for

> P. Karlsson, J. Davidsson, R. Armiento, *Prescreening Point Defects in Semiconductors With Machine Learning*, [arXiv:2609.14846](https://arxiv.org/abs/2609.14846) (2026).

Ridge, kernel ridge and multilayer-perceptron (MLP) models predict formation energies (FE) and
zero-phonon lines (ZPL) of single and double point defects in 4H-SiC. The data come from the
ADAQ high-throughput database.

## Repository layout

```
data/                         input datasets (CSV), see "Data"
python/machine_learning/      all scripts (run from any directory)
    common.py                 paths, constants, dataset loading
    input_parameters.py       input-parameter (feature) lists of the descriptors
    saved_models/             models trained on all data (not used for the paper, see below)
results/
    <descriptor>/<defect type>/<FE|ZPL>/   optimization results, learning curves, scatter data
    figures/                  generated figures (supplementary/ for optimization plots)
    tables/                   generated LaTeX tables
    jobs/{kernel,neural}/     per-job outputs of the optimization scripts (see "Run order")
requirements.txt
```

Descriptor names used in file and folder names:

| Name in code / files         | Descriptor in the paper     | Input parameters |
|------------------------------|-----------------------------|------------------|
| `sic_no_wyckoff`             | Atomic                      | 5  |
| `sic_wyckoff_letter`         | Wyckoff Letter              | 15 |
| `sic_wyckoff`                | Wyckoff Coordinate          | 59 |
| `sic_wyckoff_letter_nw_data` | Wyckoff Letter Compare      | 15 |
| `sic_wyckoff_nw_data`        | Wyckoff Coordinate Compare  | 59 |

Defect types: `vacancy-substitutional` (VS, only vacancies and substitutions) and `interstitial`
(I, at least one interstitial). Scalings: `unscaled`, `standard` (standardized) and `normal`
(normalized to [0, 1]). In result file names, standardized has no suffix and the others are
`_unscaled` and `_normal`.

## Installation

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python ≥ 3.12 is required. The figures also need LaTeX with the `newtx` fonts, because matplotlib
uses `text.usetex`.

## Data

`data/<descriptor>_<defect type>_<single|double>_<kind>.csv`, one row per defect (charge and
spin state). The input-parameter columns come first, followed by `FE`, `ZPL`, `AE` and identifiers.

* `*_original.csv`: all calculations extracted from ADAQ, with the defect slots in arbitrary order.
* `*_cleaned.csv`: the datasets used for all results. They are made from `original` as follows:
  1. Within each slot pair (`x`, `x_2`), sort the two values in ascending order.
  2. Of every group of rows with identical input parameters, keep only the row with the lowest
     formation energy.
* `*_duplicates.csv`: the rows of `original` that belong to a group of duplicates.
* The Compare datasets (`*_nw_data_*_cleaned.csv`) are the Wyckoff Letter and Wyckoff Coordinate
  input parameters (slots sorted as above) of exactly the samples (`hashID`) in the Atomic dataset.

The scripts that extracted the data from ADAQ and performed the cleaning are not part of this
repository. The rule above reproduces every `cleaned` file exactly from `original`.

## Which script produces what

| Paper item | Script | Reads | Writes |
|---|---|---|---|
| Table 2 (dataset sizes, mean FE / ZPL) | `read_target_average.py` | `data/*_cleaned.csv` | stdout |
| Table 3 (MAEs, optimal hyperparameters) | `make_tables.py` | `optim_results_linear*.toml`, `optim_results_kernel*.toml`, `optim_results_neural_clean.db` | `results/tables/all_results_table.tex` |
| Table 4 (projection scores) | `make_tables.py` | `results.toml` | `results/tables/projections.tex` |
| Fig. 1 (input-parameter statistics) | `input_parameters_details.py` | `data/*_cleaned.csv` | `results/figures/*_input_parameter_details_*.pdf` |
| Fig. 2 (covariance / Pearson heatmaps) | `correlation_heatmaps.py` | `data/*_cleaned.csv` | `results/figures/*_heatmap_*.pdf` |
| Fig. 3 (learning curves) | `plot.py` | `results.toml` | `results/figures/mae_compare_article_{1-4}[_train].pdf`, `mae_legend*.pdf` |
| Fig. 4 (scatter plots) | `plot.py` | `results_scatter.toml` | `results/figures/scatter_{1-4}.pdf`, `scatter_legend.pdf` |
| Supplementary: ridge / kernel ridge optimization | `plot.py` | `optim_results_{linear,kernel}*.toml` | `results/figures/supplementary/*_optim_{linear,kernel}*.pdf` |
| Supplementary: MLP optimization | `neural_heatmaps.py` | `optim_results_neural.db` | `results/figures/supplementary/*_neural_heatmap_*.pdf` |

Panels (a)–(d) of Figs. 3 and 4, and files `_1` … `_4`, are VS FE, VS ZPL, I FE and I ZPL.
The table and figure scripts only read the files in `results/`, so they can be rerun at once
from the shipped results (about 30 min, mostly `plot.py` and `neural_heatmaps.py`).

### Files in `results/<descriptor>/<defect type>/<target>/`

| File | Content | Written by |
|---|---|---|
| `optim_results_linear{,_unscaled,_normal}.toml` | ridge: MAE for each alpha, `best_params` | `optim_linear.py` |
| `optim_results_kernel{,_unscaled,_normal}.toml` | kernel ridge: MAE for each (gamma, alpha), `best_params` | `optim_kernel.py` + `aggregate_results.py kernel` |
| `optim_results_neural.db` | MLP optimization: per-fold MAE of every configuration tried (SQLite) | `optim_neural.py` + `aggregate_results.py neural` |
| `optim_results_neural_clean.db` | re-evaluation of the selected MLP configurations; used for Table 3 | same, with `--neural_target _clean` |
| `optim_results_neural_normal.db` | MLP runs with normalized inputs (not used in the paper) | |
| `results.toml` | learning curves (`MAE_percent_list_*`), averages from `data_average.py` | `ml.py`, `aggregate_results.py curves`, `data_average.py` |
| `{kernel,neural}_partial_results/` | one learning-curve point per file | `ml.py` |
| `results_scatter.toml` | test-set predictions for the scatter plots | `ml.py` |

Complete kernel and MLP learning curves (1022 fits each) were computed only for the four
cases shown in Fig. 3. In the other `results.toml` files those curves are filled with zeros.

## Run order (full reproduction)

All commands are run from `python/machine_learning/`, though any directory works.
Shared options: `--material <descriptor> --defect_type <type> --target_variable <FE|ZPL> --scaling <unscaled|standard|normal>`.
Each command below is one independent job. On a cluster, submit them as separate jobs, ideally
with `OMP_NUM_THREADS=1`. Running the aggregation step afterwards is enough; no special job
scripts are needed.

1. **Ridge optimization**: 60 jobs (5 descriptors × 2 defect types × 2 targets × 3 scalings),
   each a few seconds to a few minutes.
   ```
   python optim_linear.py --material sic_no_wyckoff --defect_type vacancy-substitutional --target_variable FE --scaling standard
   ```
2. **Kernel ridge optimization**: for every case and scaling, one job per gamma in
   {0, 1/59, 1/15, 1/5, 10^-4 … 10^4} (`common.KERNEL_GAMMAS`) and alpha index 0 … 10
   (alpha = 10^-8 … 10^2). `--kernel_optim_target` must match the scaling (`""` for standard,
   `_unscaled`, `_normal`). Each VS job takes about 1 CPU-min; each interstitial job takes
   tens of CPU-minutes and more than 6 GB of RAM.
   ```
   python optim_kernel.py --material sic_no_wyckoff --defect_type vacancy-substitutional --target_variable ZPL \
       --scaling unscaled --kernel_optim_target _unscaled --gamma 10.0 --kernel_optim_alpha_index 7
   python aggregate_results.py kernel        # -> optim_results_kernel*.toml incl. best_params
   ```
3. **MLP optimization**: one job per configuration and fold, with standardized scaling. The
   search was staged by hand (see the paper): (i) 25/50/75/100 neurons per layer × activation ×
   solver on 4 folds; (ii) refining the neuron count; (iii) all 10 folds over alpha. The selected
   configurations (paper Table 3(c)) were then re-evaluated with `--neural_target _clean`.
   Fixed parameters: `max_iter` and `max_fun` 9999999999999, `tol` 1e-4, `random_state` 42.
   The paper's "NAG" solver is `"solver": "sgd"` with the sklearn defaults (Nesterov momentum
   0.9, learning rate 1e-3).
   ```
   python optim_neural.py --material sic_no_wyckoff --defect_type vacancy-substitutional --target_variable FE \
       --scaling standard --fold 1 --neural_target _clean --neural_params \
       '{"hidden_layer_sizes": [24,24,24,24,24,24,24,24], "activation": "tanh", "solver": "lbfgs", "alpha": 0.01,
         "max_iter": 9999999999999, "tol": 0.0001, "max_fun": 9999999999999, "random_state": 42}'
   python aggregate_results.py neural        # -> optim_results_neural{,_clean}.db
   ```
   An L-BFGS fit of a selected configuration takes 15–25 CPU-min per fold for VS and many hours
   for interstitials.
4. **Learning curves and scatter data (Fig. 3, 4)**, using the best configuration of each model
   (dictionaries `linear_targets`, `kernel_targets`, `neural_targets` and `best_targets` in `plot.py`):
   ```
   python ml.py --model linear --material sic_wyckoff_letter --defect_type vacancy-substitutional --target_variable FE --scaling normal
   python ml.py --model kernel --material sic_no_wyckoff --defect_type vacancy-substitutional --target_variable ZPL --scaling unscaled \
       --train_data_percentage 9 --fold 10      # 1022 jobs: d = 1..9, fold = 1..C(10, d); omit both to run all in one job
   python ml.py --model neural ... --neural_params '<Table 3(c) configuration as above>'
   python aggregate_results.py curves        # partial results -> results.toml
   python data_average.py --material ... --defect_type ... --target_variable ...
   ```
   `ml.py` stores the test predictions of fold `--scatter_fold` (default 10) at 9/10 training
   data for the scatter plot. The published Fig. 4 uses fold 10, except panel (c) (I FE, MLP),
   which uses fold 1.
5. **Tables and figures**:
   ```
   python read_target_average.py
   python make_tables.py
   python input_parameters_details.py
   python correlation_heatmaps.py
   python plot.py
   python neural_heatmaps.py
   ```

To write results somewhere other than `results/`, set `MLDEFECTS_RESULTS_DIR` (the smoke test
does this).

## Smoke test

```
python python/machine_learning/smoke_test.py          # about 5 min with 8 cores
```

The smoke test does the following:

* checks Table 2 against `data/`;
* reruns the 60 ridge cells and the 30 VS kernel ridge cells of Table 3 at their published
  optimal hyperparameters, in a temporary copy of `results/`;
* compares them with the published optimization files;
* regenerates Table 3 with `make_tables.py` and checks all 140 MAE cells against the paper.

`--full` also reruns the 30 interstitial kernel ridge cells, which takes several CPU-hours.
The MLP cells are not retrained (see below).

## Reproducibility notes

* **Ridge and kernel ridge** reproduce to floating-point precision (≤ 1e-6 eV), both the full
  optimization grids and the learning curves.
* **MLP results are sensitive to the numerical environment.** With effectively unlimited
  `max_iter`, the L-BFGS fits run for about 29 000–47 000 iterations. Different SciPy versions,
  BLAS libraries or thread counts give different per-fold MAEs (up to about 0.07 eV for the
  same fold). The original environment was not recorded. Rerunning all 10 folds of the VS FE
  headline model (published 0.437 eV) gave 0.485 eV (scikit-learn 1.6.1, SciPy 1.18.1,
  OpenBLAS, 1 thread). Changing only the MLP `random_state` (0–4) gives 10-fold means between
  0.463 and 0.542 eV. That spread (about 0.03 eV standard deviation) is as large as the
  environment effect. The published value sits at the low end of what reruns produce, as is
  expected when one configuration is chosen from about 3 300 using the same folds that report
  its MAE. The qualitative conclusions are unchanged.
* **Table 4**: the MLP ZPL projections computed from the shipped `results.toml` are 0.169 (VS)
  and 0.233 (I), while the paper shows 0.248 and 0.241. The test-MAE part matches Table 3, so
  the stored MLP training MAEs differ from those the paper's table was made with.
* **`saved_models/` and `train_and_save_model.py`** are not used for any result in the paper.
  The Wyckoff Letter interstitial models differ from Table 3: the FE MLP has 5×48 neurons
  (paper: 5×44) and the ZPL kernel ridge model has alpha = 0.01 (optimum: 1).
* **`optim_results_neural.db`** has no fold column for the published runs. Some configurations
  have more than 10 rows (repeated runs) and some MAE rows have no matching hyperparameter row.
  Runs aggregated with the current `aggregate_results.py` record the fold.
