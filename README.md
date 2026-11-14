# Prescreening Point Defects in Semiconductors With Machine Learning

Code, input data and published output values for

> P. Karlsson, J. Davidsson, R. Armiento, *Prescreening Point Defects in Semiconductors With Machine Learning*, [arXiv:2609.14846](https://arxiv.org/abs/2609.14846) (2026).

Ridge, kernel ridge and multilayer-perceptron (MLP) models predict formation energies (FE) and
zero-phonon lines (ZPL) of single and double point defects in 4H-SiC. The data come from the
ADAQ high-throughput database.

This repository contains everything needed to rerun the calculations, plus the output values
that the paper's tables and figures were made from, and the fitted weights of the four
best models quoted in the abstract (see "The best models"). The fitted weights of every Table 3
model and of every MLP learning-curve fit, and the raw per-job outputs, are in the companion
repository **mldefects_results** (`<URL of mldefects_results>`); see "Saved model weights".

## Repository layout

```
data/                         input datasets (CSV), see "Data"
python/machine_learning/      all scripts (run from any directory)
    common.py                 paths, constants, dataset loading
    input_parameters.py       input-parameter (feature) lists of the descriptors
    cv_models.py              per-fold model weights: train, and evaluate (see "Saved model weights")
results/
    <descriptor>/<defect type>/<FE|ZPL>/   published optimization results, MAEs of the Table 3 models,
                                           learning curves, scatter data
requirements.txt
```

Running the scripts also creates the following in `results/`. The published versions of the
first two are in mldefects_results:

```
results/
    <descriptor>/<defect type>/<FE|ZPL>/
        weights/              fitted weights of every Table 3 model, one file per CV fold;
                              curves/: fitted weights of the MLP learning-curve fits
        {kernel,neural}_partial_results/   one learning-curve point per file
    figures/                  generated figures (supplementary/ for optimization plots)
    tables/                   generated LaTeX tables
    jobs/{kernel,neural}/     per-job outputs of the optimization scripts (see "Run order")
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
| Table 3 (MAEs, optimal hyperparameters) | `make_tables.py` | `optim_results_linear*.toml`, `optim_results_kernel*.toml`, `optim_results_neural_clean.db` (MLP configuration), `cv_scores.toml` (MLP MAEs; all MAEs with `--from-weights`) | `results/tables/all_results_table.tex` |
| Table 4 (projection scores) | `make_tables.py` | `results.toml` (or `cv_scores.toml` with `--from-weights`) | `results/tables/projections.tex` |
| Fig. 1 (input-parameter statistics) | `input_parameters_details.py` | `data/*_cleaned.csv` | `results/figures/*_input_parameter_details_*.pdf` |
| Fig. 2 (covariance / Pearson heatmaps) | `correlation_heatmaps.py` | `data/*_cleaned.csv` | `results/figures/*_heatmap_*.pdf` |
| Fig. 3 (learning curves) | `plot.py` | `results.toml` | `results/figures/mae_compare_article_{1-4}[_train].pdf`, `mae_legend*.pdf` |
| Fig. 4 (scatter plots) | `plot.py` | `results_scatter.toml` (predictions from `weights/` with `--from-weights`) | `results/figures/scatter_{1-4}.pdf`, `scatter_legend.pdf` |
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
| `optim_results_neural.db` | MLP optimization: per-fold MAE of every configuration tried (SQLite); `neural_search_jobs.py` lists them as jobs | `optim_neural.py` + `aggregate_results.py neural` |
| `optim_results_neural_clean.db` | 10-fold evaluation of the candidate MLP configurations in the original search; selects the Table 3 configuration | same, with `--neural_target _clean` |
| `cv_scores.toml` | per-fold train and test MAEs of every Table 3 model, computed from its saved weights; the Table 3 MLP MAEs | `cv_models.py train` + `evaluate` |
| `results.toml` | learning curves (`MAE_percent_list_*`), averages from `data_average.py` | `ml.py`, `aggregate_results.py curves`, `data_average.py` |
| `results_scatter.toml` | test-set predictions for the scatter plots | `ml.py` |
| `{kernel,neural}_partial_results/` (mldefects_results only) | one learning-curve point per file | `ml.py` |
| `weights/{ridge,kernel,neural}{,_unscaled,_normal}_fold{1-10}.npz` (here only for the four best models, see "The best models"; all in mldefects_results) | fitted weights of each Table 3 model for each CV fold | `cv_models.py train` |
| `weights/curves/neural_d{1-9}_f{n}.npz` (mldefects_results only) | fitted weights of each MLP learning-curve fit (Fig. 3) | `ml.py --save_weights` |
| `weights/neural_alternative_fold{1-10}.npz` (mldefects_results only) | the alternative MLP configurations compared in Supplementary Material section I | `cv_models.py train --name neural_alternative` |

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
   candidate configurations were evaluated on all 10 folds with `--neural_target _clean`, and
   the one with the lowest mean MAE is the Table 3 configuration (Supplementary Table S1(c)).
   This search was run in the original, unrecorded, environment; the published MLP MAEs come
   from retraining the selected configurations (step 5).
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
   To re-run the published search, `neural_search_jobs.py` lists one such command for every
   configuration and fold recorded in the published databases (about 19 000 jobs per case for
   the search, 12 000 for the 10-fold candidate evaluation):
   ```
   python neural_search_jobs.py --material sic_no_wyckoff --defect_type vacancy-substitutional --target_variable FE [--neural_target _clean] > jobs.txt
   ```
   An L-BFGS fit of a selected configuration takes 15–25 CPU-min per fold for VS and many hours
   for interstitials.
4. **Learning curves and scatter data (Fig. 3, 4)**, using the best configuration of each model
   (dictionaries `linear_targets`, `kernel_targets`, `neural_targets` and `best_targets` in `plot.py`):
   ```
   python ml.py --model linear --material sic_wyckoff_letter --defect_type vacancy-substitutional --target_variable FE --scaling normal
   python ml.py --model kernel --material sic_no_wyckoff --defect_type vacancy-substitutional --target_variable ZPL --scaling unscaled \
       --train_data_percentage 9 --fold 10      # 1022 jobs: d = 1..9, fold = 1..C(10, d); omit both to run all in one job
   python ml.py --model neural ... --neural_params '<Table S1(c) configuration as above>' [--save_weights]
   python aggregate_results.py curves        # partial results -> results.toml
   python data_average.py --material ... --defect_type ... --target_variable ...
   ```
   `ml.py` stores the test predictions of fold `--scatter_fold` (default 10) at 9/10 training
   data for the scatter plot. The published Fig. 4 uses fold 10, except panel (c) (I FE, MLP),
   which uses fold 1. With `--save_weights`, the weights of every fit are saved to
   `weights/curves/`. In the same environment, the fits at 9/10 training data are identical to
   the Table 3 models of step 5 (this holds for the published curves and weights).
5. **Per-fold model weights and the Table 3 MLP MAEs**: 140 cells × 10 folds, with the
   hyperparameters chosen in steps 1–3. `evaluate` writes `cv_scores.toml`, from which
   `make_tables.py` takes the MLP MAEs (see "Saved model weights").
   ```
   python cv_models.py train --descriptor sic_no_wyckoff --defect_type vacancy-substitutional --target_variable FE \
       --model neural [--scaling standard] [--fold 1 2 ...]        # --model ridge | kernel | neural
   python cv_models.py evaluate              # -> cv_scores.toml
   python cv_models.py train ... --model neural --name neural_alternative --neural_params '<JSON>'   # Supplementary section I
   python cv_models.py evaluate-extra --jobs 16   # checks weights/curves/ (about 37 GB) and named configurations
   ```
6. **Tables and figures**:
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

## The best models

`results/` includes the fitted weights (all 10 cross-validation folds, 4.8 MB in total) of the
four best models of Table 3, quoted in the abstract:

| Task | Model | Descriptor | Weights | MAE (eV) |
|---|---|---|---|---|
| VS FE | MLP, standardized | Atomic | `results/sic_no_wyckoff/vacancy-substitutional/FE/weights/neural_fold{1-10}.npz` | 0.478 |
| VS ZPL | kernel ridge, unscaled | Atomic | `results/sic_no_wyckoff/vacancy-substitutional/ZPL/weights/kernel_unscaled_fold{1-10}.npz` | 0.202 |
| I FE | MLP, standardized | Wyckoff Coordinate | `results/sic_wyckoff/interstitial/FE/weights/neural_fold{1-10}.npz` | 1.123 |
| I ZPL | kernel ridge, unscaled | Atomic | `results/sic_no_wyckoff/interstitial/ZPL/weights/kernel_unscaled_fold{1-10}.npz` | 0.230 |

Each fold's model was trained on the other nine folds plus all single defects. To predict, from
`python/machine_learning/`:

```python
import numpy as np
from cv_models import load_model
from input_parameters import FEATURES

models = [load_model("sic_wyckoff", "interstitial", "FE", "neural", "standard", fold) for fold in range(1, 11)]
X = ...   # rows of input parameters, columns FEATURES["sic_wyckoff"], as in data/*_cleaned.csv
prediction = np.mean([model(X) for model in models], axis=0)   # eV; or use a single fold
```

For the kernel ridge models, `load_model` rebuilds the training inputs from `data/`. With these
files, `python plot.py --from-weights` also recomputes the Fig. 4 predictions from the weights.

## Saved model weights

The repository mldefects_results has the same layout as `results/` here, plus a `weights/`
folder in every case folder. These hold the fitted weights of every model in Table 3 (ridge
and kernel ridge for all three scalings, MLP for standardized inputs), separately for each of
the 10 cross-validation folds. Each `.npz` file contains the test-set row indices, the scaler
parameters, the weights (ridge coefficients, kernel ridge dual coefficients, or the MLP weight
matrices and biases), the hyperparameters, the train and test MAEs, and the software versions
and thread settings the model was trained with. Predictions are computed with NumPy from these
arrays, so the files do not depend on the scikit-learn version and contain no pickled code.

To regenerate Tables 3 and 4 and Fig. 4 from the weights, without any training, point
`MLDEFECTS_RESULTS_DIR` at a checkout of mldefects_results (the tables and figures are then
written there):

```
export MLDEFECTS_RESULTS_DIR=/path/to/mldefects_results
python cv_models.py evaluate          # recomputes every fold's MAEs from the weights -> cv_scores.toml
python make_tables.py --from-weights
python plot.py --from-weights
```

`evaluate` checks that each file's test indices match the cross-validation split of `data/` and
that the MAEs computed from the weights match those recorded at training time (to 1e-6 eV). It
also prints each cell next to the published value. For the Fig. 3 learning curves,
`python cv_models.py evaluate-extra --jobs 16` recomputes the MAEs of all 4088 MLP curve fits
from `weights/curves/` and checks them against `results.toml`, from which `plot.py` draws Fig. 3.

The weights were trained with `cv_models.py train` from the hyperparameters selected in the
published optimization results, in the environment recorded in each file (Python 3.12.3,
scikit-learn 1.6.1, SciPy 1.15.3, NumPy 2.2.6 with its bundled OpenBLAS, one thread; versions as
in `requirements.txt`). The ridge and kernel ridge weights reproduce Table 3 to within 3e-7 eV.
The MLP MAEs in Table 3 and the MLP learning curves of Fig. 3 *are* the MAEs of these weights
(see "Reproducibility notes"). Without `--from-weights`, the tables and figures are made from
the values shipped in `results/`; both routes give the published tables and figures.

## Smoke test

```
python python/machine_learning/smoke_test.py          # about 5 min with 8 cores
```

The smoke test does the following:

* checks Table 2 against `data/`;
* reruns the 60 ridge cells and the 30 VS kernel ridge cells of Table 3 at their published
  optimal hyperparameters, in a temporary copy of `results/`;
* compares them with the published optimization files;
* regenerates Tables 3 and 4 with `make_tables.py` and checks all 140 MAE cells and the 12
  projection scores against the paper.

`--full` also reruns the 30 interstitial kernel ridge cells, which takes several CPU-hours.
The MLP cells are not retrained (see below).

## Reproducibility notes

* **Ridge and kernel ridge** reproduce to floating-point precision (≤ 1e-6 eV), both the full
  optimization grids and the learning curves.
* **MLP results are sensitive to the numerical environment.** With effectively unlimited
  `max_iter`, the L-BFGS fits run for tens of thousands of iterations, and the large Adam
  networks for thousands of epochs. Different SciPy/NumPy versions, BLAS kernels or thread
  counts give different per-fold MAEs (up to about 0.07 eV for the same fold). The MLP
  configurations were selected in an original, unrecorded environment (the heatmaps in the
  Supplementary Material show those scores). The published MLP MAEs, learning curves and Fig. 4
  predictions come from retraining the selected configurations in the recorded environment
  above; this changed the 10-fold means by up to 0.04 eV (e.g. VS FE 0.437 → 0.478 eV).
  Changing only the MLP `random_state` (0–4) moves the VS FE mean between 0.463 and 0.542 eV, so
  the environment effect is within the run-to-run spread. The qualitative conclusions are
  unchanged.
* **Exact MLP reproduction.** In the recorded environment (`requirements.txt`, Python 3.12,
  pip wheels, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`), retraining gives bit-identical weights
  on the machine the weights were made on (Intel Broadwell) and, with
  ```
  export OPENBLAS_CORETYPE=Haswell
  export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
  ```
  also on an AVX-512 machine (Intel Xeon Gold 6130); without these settings the latter gives
  different fits. If you cannot reproduce a fit, use the saved weights in mldefects_results.
* **`optim_results_neural.db`** has no fold column for the published runs. Some configurations
  have more than 10 rows (repeated runs) and some MAE rows have no matching hyperparameter row.
  Runs aggregated with the current `aggregate_results.py` record the fold.
