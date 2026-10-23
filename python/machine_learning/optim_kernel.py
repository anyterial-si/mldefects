import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import toml
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

sys.path.append(os.getcwd())
from init_ml import (
    data,
    data_single,
    gamma,
    features,
    pd,
    scaling,
    target_variable,
    kernel_optim_target,
    kernel_optim_alpha_index,
    alphas_argparse,
    material,
    defect_type,

)

if scaling == "standard":
    from sklearn.preprocessing import StandardScaler
elif scaling == "normal":
    from sklearn.preprocessing import MinMaxScaler


alpha_mae = []
if alphas_argparse:
    alphas = alphas_argparse
else:
    alphas = [10**e for e in np.linspace(-8, 2, 11)]
alphas_part = [alphas[kernel_optim_alpha_index]]
## max just above 80, go 80
# alphas = [10**e for e in np.linspace(-8, 2, 80)]
for alpha in alphas_part:
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(kf.split(data))
    fold_indices = [test_idx for _, test_idx in folds]

    k = 9
    combos_iter = combinations(range(n_splits), k)
    combos = list(combos_iter)

    mae_fold_list_test = []

    for train_fold_ids in combos:
        train_index = np.concatenate([fold_indices[i] for i in train_fold_ids])
        test_index = np.concatenate([fold_indices[i] for i in range(n_splits) if i not in train_fold_ids])

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

        model = KernelRidge(alpha=alpha, gamma=gamma, kernel="rbf")
        model.fit(train_X_scaled, train_y_scaled)

        y_pred_test_scaled = model.predict(test_X_scaled)

        if scaling == "unscaled":
            y_pred_test = y_pred_test_scaled

        else:
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

        mae_fold_list_test.append(mean_absolute_error(test_y, y_pred_test))

    alpha_mae.append(np.mean(mae_fold_list_test))

toml_optim_data = {}
toml_optim_data["kernel_optim_target"] = kernel_optim_target
toml_optim_data["alpha_mae"] = float(alpha_mae[0])
toml_optim_data["gamma"] = gamma
toml_optim_data["kernel_optim_alpha_index"] = kernel_optim_alpha_index
toml_optim_data["alphas"] = alphas
toml_optim_data["alphas_part"] = alphas_part

toml_optim_data["material"] = material
toml_optim_data["defect_type"] = defect_type
toml_optim_data["target_variable"] = target_variable

toml_path = f"../sbatch/kernel_toml_results/{material}_{defect_type}_{target_variable}_gamma{gamma}_alpha{alphas_part[0]}_{scaling}.toml"
with open(toml_path, "w", encoding="utf-8") as file:
    toml.dump(toml_optim_data, file)
