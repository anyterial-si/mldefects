from itertools import combinations

import numpy as np
import toml
from init_ml import (
    data,
    data_single,
    fold,
    model_type,
    os,
    features,
    pd,
    scaling,
    sys,
    target_folder,
    material,
    defect_type,
    target_variable,
    toml_path,
    toml_scatter_path,
    train_data_percentage,
    cross_validation,
)
if model_type == "neural":
    from init_ml import neural_params
    from init_ml import data_single

if model_type == "linear":
    from sklearn.linear_model import Ridge
elif model_type == "kernel":
    from sklearn.kernel_ridge import KernelRidge
elif model_type == "neural":
    from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

if scaling == "standard":
    from sklearn.preprocessing import StandardScaler
elif scaling == "normal":
    from sklearn.preprocessing import MinMaxScaler


if model_type == "linear":
    try:
        if scaling == "standard":
            toml_data_params = toml.load(os.path.join(target_folder, "optim_results_linear.toml"))
        elif scaling == "normal":
            toml_data_params = toml.load(os.path.join(target_folder, "optim_results_linear_normal.toml"))
        elif scaling == "unscaled":
            toml_data_params = toml.load(os.path.join(target_folder, "optim_results_linear_unscaled.toml"))
        best_params = toml_data_params["best_params"]
        model = Ridge(**best_params)

    except (FileNotFoundError, KeyError):
        print("\nLinear model not optimized\n")
        sys.exit(0)

elif model_type == "kernel":
    try:
        toml_data_params = toml.load(os.path.join(target_folder, "optim_results_kernel_unscaled.toml"))
        best_params = toml_data_params["best_params"]
        best_params["kernel"] = "rbf"
        model = KernelRidge(**best_params)

    except (FileNotFoundError, KeyError):
        print("\nKernel model not optimised\n")
        sys.exit(0)

elif model_type == "neural":
    try:
        model = MLPRegressor(**neural_params)

    except Exception:
        print("\ngive neural net params\n")
        sys.exit(0)

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mae_percent_list_train = []
mae_percent_list_test = []
scatter_pred = []
scatter_true = []

if cross_validation == "varied":
    data_percentage_k_limit = 10
else:
    data_percentage_k_limit = 11

for data_percentage_k in range(1, data_percentage_k_limit):

    if train_data_percentage is None:
        pass
    elif data_percentage_k != train_data_percentage:
        continue

    if cross_validation == "varied":
        folds = list(kf.split(data))
        fold_indices = [test_idx for _, test_idx in folds]

        combos_iter = combinations(range(n_splits), data_percentage_k)
        combos_list = list(combos_iter)
    else:
        fraction_data = data.sample(frac=data_percentage_k/10, random_state=42)
        folds = list(kf.split(fraction_data))
        fold_indices = [test_idx for _, test_idx in folds]

        combos_iter = combinations(range(n_splits), 9)
        combos_list = list(combos_iter)

    mae_fold_list_train = []
    mae_fold_list_test = []

    for f, train_fold_ids in enumerate(combos_list):

        if fold is None:
            pass
        elif f != (fold-1):
            continue

        train_index = np.concatenate([fold_indices[i] for i in train_fold_ids])
        test_index = np.concatenate([fold_indices[i] for i in range(n_splits) if i not in train_fold_ids])

        if cross_validation == "varied":
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]
        else:
            train_data = fraction_data.iloc[train_index]
            test_data = fraction_data.iloc[test_index]
        train_data = pd.concat([train_data, data_single], ignore_index=True)



        train_X = train_data[features]
        train_y = train_data[[target_variable]]

        test_X = test_data[features]
        test_y = test_data[[target_variable]]

        if scaling == "unscaled":
            train_X_scaled = train_data[features]
            train_y_scaled = train_data[target_variable]

            test_X_scaled = test_data[features]
            test_y_scaled = test_data[target_variable]

        else:
            if scaling == "standard":
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
            elif scaling == "normal":
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

            scaler_X.fit(train_X)
            scaler_y.fit(train_y.values.reshape(-1, 1))

            train_X_scaled = scaler_X.transform(train_X)
            train_y_scaled = scaler_y.transform(train_y.values.reshape(-1, 1))

            test_X_scaled = scaler_X.transform(test_X)
            test_y_scaled = scaler_y.transform(test_y.values.reshape(-1, 1))

            train_X_scaled = pd.DataFrame(train_X_scaled, columns=train_X.columns)
            train_y_scaled = pd.DataFrame(train_y_scaled, columns=train_y.columns)
            test_X_scaled = pd.DataFrame(test_X_scaled, columns=test_X.columns)
            test_y_scaled = pd.DataFrame(test_y_scaled, columns=test_y.columns)


        if model_type == "linear":
            model.fit(train_X_scaled, train_y_scaled)

        elif model_type == "kernel":
            model.fit(train_X_scaled, train_y_scaled)

        elif model_type == "neural":
            model.fit(train_X_scaled, train_y_scaled.squeeze())

        y_pred_train_scaled = model.predict(train_X_scaled)
        y_pred_test_scaled = model.predict(test_X_scaled)

        if scaling == "unscaled":
            y_pred_train = y_pred_train_scaled
            y_pred_test = y_pred_test_scaled

        else:
            y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1))
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

        mae_fold_list_train.append(mean_absolute_error(train_y, y_pred_train))
        mae_fold_list_test.append(mean_absolute_error(test_y, y_pred_test))

        if data_percentage_k == 9:
            scatter_pred = y_pred_test.tolist()
            scatter_true = test_y.to_numpy().tolist()

    mae_percent_list_train.append(np.mean(mae_fold_list_train))
    mae_percent_list_test.append(np.mean(mae_fold_list_test))


mae_percent_list_train = [float(val) for val in mae_percent_list_train]
mae_percent_list_test = [float(val) for val in mae_percent_list_test]

toml_scatter_data = toml.load(toml_scatter_path)


if scatter_pred and cross_validation == "varied":
    try:
        scatter_pred = [float(val[0]) for val in scatter_pred]
    except TypeError:
        scatter_pred = [float(val) for val in scatter_pred]
    try:
        scatter_true = [float(val[0]) for val in scatter_true]
    except TypeError:
        scatter_true = [float(val) for val in scatter_true]

    toml_scatter_data["scatter_true"] = scatter_true
    print(len(scatter_true))

    if model_type == "linear":
        toml_scatter_data["scatter_linear_pred"] = scatter_pred
    elif model_type == "kernel":
        toml_scatter_data["scatter_kernel_pred"] = scatter_pred
    elif model_type == "neural":
        toml_scatter_data["scatter_neural_pred"] = scatter_pred

    with open(toml_scatter_path, "w", encoding="utf-8") as file:
        toml.dump(toml_scatter_data, file)


if cross_validation == "varied":
    data_percentage_folds_nrs = [
        10,
        45,
        120,
        210,
        252,
        210,
        120,
        45,
        10,
    ]
    if model_type == "linear":
        toml_data = toml.load(toml_path)
        toml_data["MAE_percent_list_linear_train"] = mae_percent_list_train
        toml_data["MAE_percent_list_linear_test"] = mae_percent_list_test

        with open(toml_path, "w", encoding="utf-8") as file:
            toml.dump(toml_data, file)

    else:
        toml_partial_data = {
            "train_data_percentage": int(train_data_percentage),
            "fold": int(fold),
            "mae_train": float(mae_percent_list_train[0]),
            "mae_test": float(mae_percent_list_test[0]),
        }

        with open(f"../../results/{material}/{defect_type}/{target_variable}/{model_type}_partial_results/partial_results_d{train_data_percentage}_f{fold}.toml", "w", encoding="utf-8") as file:
            toml.dump(toml_partial_data, file)

        # toml_data[f"MAE_percent_list_{model_type}_train_{train_data_percentage}0"][fold-1] = mae_percent_list_train[0]
        # toml_data[f"MAE_percent_list_{model_type}_test_{train_data_percentage}0"][fold-1] = mae_percent_list_test[0]
        # with open(toml_path, "w", encoding="utf-8") as file:
        #     toml.dump(toml_data, file)

    # elif model_type == "neural":
    #     try:
    #         assert toml_data[f"MAE_percent_list_neural_train_{train_data_percentage}0"]
    #     except KeyError:
    #         for i, data_percentage_folds_nr in data_percentage_folds_nrs:
    #             toml_data[f"MAE_percent_list_neural_train_{i+1}0"] = [0.0] * data_percentage_folds_nr
    #             toml_data[f"MAE_percent_list_neural_test_{i+1}0"] = [0.0] * data_percentage_folds_nr

    #     toml_data[f"MAE_percent_list_neural_train_{train_data_percentage}0"][fold-1] = mae_percent_list_train[0]
    #     toml_data[f"MAE_percent_list_neural_test_{train_data_percentage}0"][fold-1] = mae_percent_list_test[0]


    # with open(toml_path, "w", encoding="utf-8") as file:
    #     toml.dump(toml_data, file)

else:
    toml_data = toml.load(toml_path)

    if model_type == "linear":
        toml_data["MAE_percent_list_linear_train_10_fold"] = mae_percent_list_train
        toml_data["MAE_percent_list_linear_test_10_fold"] = mae_percent_list_test

    else:
        try:
            assert toml_data[f"MAE_percent_list_{model_type}_train_{train_data_percentage}0_10_fold"]
        except KeyError:
            for i in range(1, data_percentage_k_limit):
                toml_data[f"MAE_percent_list_{model_type}_train_{i}0_10_fold"] = [0.0] * n_splits
                toml_data[f"MAE_percent_list_{model_type}_test_{i}0_10_fold"] = [0.0] * n_splits

        toml_data[f"MAE_percent_list_{model_type}_train_{train_data_percentage}0_10_fold"][fold-1] = mae_percent_list_train[0]
        toml_data[f"MAE_percent_list_{model_type}_test_{train_data_percentage}0_10_fold"][fold-1] = mae_percent_list_test[0]


    with open(toml_path, "w", encoding="utf-8") as file:
        toml.dump(toml_data, file)
