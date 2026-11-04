"""Ridge optimization: 10-fold CV test MAE for 100 alphas (10^-8 ... 10^16.75, or --alphas_argparse).

Writes results/<descriptor>/<defect_type>/<target>/optim_results_linear{,_unscaled,_normal}.toml.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import toml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from init_ml import (
    alphas_argparse,
    data,
    data_single,
    features,
    scaling,
    target_folder,
    target_variable,
)

if scaling == "standard":
    from sklearn.preprocessing import StandardScaler
    linear_optim_target = ""
elif scaling == "normal":
    from sklearn.preprocessing import MinMaxScaler
    linear_optim_target = "_normal"
else:
    linear_optim_target = "_unscaled"


alpha_maes = []
# default grid: 100 values 10^-8 ... 10^16.75; --alphas_argparse "[a1, a2]" evaluates only the given alphas
alphas = alphas_argparse if alphas_argparse else [10**e for e in np.arange(-8, 17, 0.25)]

for alpha in alphas:
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

        model = Ridge(alpha=alpha)
        model.fit(train_X_scaled, train_y_scaled)

        y_pred_test_scaled = model.predict(test_X_scaled)

        if scaling == "unscaled":
            y_pred_test = y_pred_test_scaled

        else:
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

        mae_fold_list_test.append(mean_absolute_error(test_y, y_pred_test))

    alpha_maes.append(np.mean(mae_fold_list_test))

alpha_maes = [float(val) for val in alpha_maes]
alphas = [float(val) for val in alphas]
best_alpha = alphas[alpha_maes.index(min(alpha_maes))]

toml_data = {
    "best_params": {
        "alpha": best_alpha,
    },
    "scores": {
        "mae": alpha_maes,
        "alphas": alphas,
    },
}

toml_optim_path = target_folder / f"optim_results_linear{linear_optim_target}.toml"
with open(toml_optim_path, "w", encoding="utf-8") as file:
    toml.dump(toml_data, file)
