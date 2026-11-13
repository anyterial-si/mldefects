"""Learning curves (paper Fig. 3) and scatter-plot data (Fig. 4).

The double-defect data are split into 10 folds (KFold, shuffled, random_state 42).
For each training fraction d = 1..9 (tenths) every combination of d folds is used
for training (all single-defect data are always added to the training set) and the
remaining folds for testing, i.e. C(10, d) fits per d, 1022 fits in total.

Output, in results/<descriptor>/<defect_type>/<target>/:
  --model linear : runs all 1022 fits and writes MAE_percent_list_linear_{train,test}
                   (one value per d) to results.toml.
  --model kernel / neural : writes one TOML per (d, fold) to {model}_partial_results/;
                   select a single job with --train_data_percentage and --fold, or omit them
                   to run all. aggregate_results.py collects the partial results into results.toml.
  All models: the test-set predictions of fold --scatter_fold (default 10) at d = 9 are
  written to results_scatter.toml as scatter_{model}_true / _pred / _fold.

Hyperparameters are read from the optimization results for the chosen --scaling
(optim_results_{linear,kernel}{suffix}.toml); MLP parameters are given with --neural_params.
"""

from itertools import combinations

import numpy as np
import toml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from common import CV_RANDOM_STATE, N_SPLITS, SCALING_SUFFIX, ensure_dir
from init_ml import (
    data,
    data_single,
    features,
    fold,
    model_type,
    neural_params,
    pd,
    scaling,
    scatter_fold,
    sys,
    target_folder,
    target_variable,
    toml_path,
    toml_scatter_path,
    train_data_percentage,
)

if model_type == "linear":
    from sklearn.linear_model import Ridge
elif model_type == "kernel":
    from sklearn.kernel_ridge import KernelRidge
elif model_type == "neural":
    from sklearn.neural_network import MLPRegressor

if scaling == "standard":
    from sklearn.preprocessing import StandardScaler
elif scaling == "normal":
    from sklearn.preprocessing import MinMaxScaler


if model_type in ("linear", "kernel"):
    optim_path = target_folder / f"optim_results_{model_type}{SCALING_SUFFIX[scaling]}.toml"
    try:
        best_params = toml.load(optim_path)["best_params"]
    except (FileNotFoundError, KeyError):
        sys.exit(f"{optim_path} not found or has no best_params: run the {model_type} optimization first")
    if model_type == "linear":
        model = Ridge(**best_params)
    else:
        model = KernelRidge(**best_params, kernel="rbf")
else:
    if neural_params is None:
        sys.exit("--model neural requires --neural_params (JSON dict of MLPRegressor parameters)")
    model = MLPRegressor(**neural_params)

if model_type == "linear" and (train_data_percentage is not None or fold is not None):
    sys.exit("--model linear writes complete learning curves; run it without --fold/--train_data_percentage")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
fold_indices = [test_idx for _, test_idx in kf.split(data)]

results = []  # (d, fold, mae_train, mae_test)
scatter = None

for data_percentage_k in range(1, N_SPLITS):
    if train_data_percentage is not None and data_percentage_k != train_data_percentage:
        continue

    for f, train_fold_ids in enumerate(combinations(range(N_SPLITS), data_percentage_k), start=1):
        if fold is not None and f != fold:
            continue

        train_index = np.concatenate([fold_indices[i] for i in train_fold_ids])
        test_index = np.concatenate([fold_indices[i] for i in range(N_SPLITS) if i not in train_fold_ids])

        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_data = pd.concat([train_data, data_single], ignore_index=True)

        train_X = train_data[features]
        train_y = train_data[[target_variable]]

        test_X = test_data[features]
        test_y = test_data[[target_variable]]

        if scaling == "unscaled":
            train_X_scaled = train_data[features]
            train_y_scaled = train_data[target_variable]

            test_X_scaled = test_data[features]

        else:
            if scaling == "standard":
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
            elif scaling == "normal":
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

            scaler_X.fit(train_X)
            scaler_y.fit(train_y.values.reshape(-1, 1))

            train_X_scaled = pd.DataFrame(scaler_X.transform(train_X), columns=train_X.columns)
            train_y_scaled = pd.DataFrame(scaler_y.transform(train_y.values.reshape(-1, 1)), columns=train_y.columns)
            test_X_scaled = pd.DataFrame(scaler_X.transform(test_X), columns=test_X.columns)

        if model_type == "neural":
            model.fit(train_X_scaled, train_y_scaled.squeeze())
        else:
            model.fit(train_X_scaled, train_y_scaled)

        y_pred_train_scaled = model.predict(train_X_scaled)
        y_pred_test_scaled = model.predict(test_X_scaled)

        if scaling == "unscaled":
            y_pred_train = y_pred_train_scaled
            y_pred_test = y_pred_test_scaled
        else:
            y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1))
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

        mae_train = float(mean_absolute_error(train_y, y_pred_train))
        mae_test = float(mean_absolute_error(test_y, y_pred_test))
        results.append((data_percentage_k, f, mae_train, mae_test))

        if data_percentage_k == N_SPLITS - 1 and f == scatter_fold:
            scatter = {
                f"scatter_{model_type}_true": [float(v) for v in np.ravel(test_y.to_numpy())],
                f"scatter_{model_type}_pred": [float(v) for v in np.ravel(y_pred_test)],
                f"scatter_{model_type}_fold": f,
                f"scatter_{model_type}_scaling": scaling,
            }

if not results:
    sys.exit("No (train_data_percentage, fold) combination selected")

# Scatter data: each model stores its own true values, so pairs always come from the same fold.
if scatter is not None:
    toml_scatter_data = toml.load(toml_scatter_path) if toml_scatter_path.exists() else {}
    toml_scatter_data.update(scatter)
    with open(toml_scatter_path, "w", encoding="utf-8") as file:
        toml.dump(toml_scatter_data, file)

if model_type == "linear":
    mae_train_per_d = [float(np.mean([r[2] for r in results if r[0] == d])) for d in range(1, N_SPLITS)]
    mae_test_per_d = [float(np.mean([r[3] for r in results if r[0] == d])) for d in range(1, N_SPLITS)]
    toml_data = toml.load(toml_path) if toml_path.exists() else {}
    toml_data["MAE_percent_list_linear_train"] = mae_train_per_d
    toml_data["MAE_percent_list_linear_test"] = mae_test_per_d
    toml_data["MAE_percent_list_linear_scaling"] = scaling
    with open(toml_path, "w", encoding="utf-8") as file:
        toml.dump(toml_data, file)
else:
    partial_folder = ensure_dir(target_folder / f"{model_type}_partial_results")
    for d, f, mae_train, mae_test in results:
        with open(partial_folder / f"partial_results_d{d}_f{f}.toml", "w", encoding="utf-8") as file:
            toml.dump({"train_data_percentage": d, "fold": f, "mae_train": mae_train,
                       "mae_test": mae_test, "scaling": scaling}, file)
