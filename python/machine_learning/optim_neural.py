import os
import sys
import toml
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

sys.path.append(os.getcwd())
from init_ml import (
    data,
    data_single,
    fold,
    neural_params,
    features,
    pd,
    scaling,
    material,
    defect_type,
    target_variable,
    neural_target
)

if scaling == "standard":
    from sklearn.preprocessing import StandardScaler
elif scaling == "normal":
    from sklearn.preprocessing import MinMaxScaler


n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = list(kf.split(data))
fold_indices = [test_idx for _, test_idx in folds]

combos_iter = combinations(range(n_splits), 9)
combos = list(combos_iter)
train_fold_ids = combos[fold-1]

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

model = MLPRegressor(**neural_params)
model.fit(train_X_scaled, train_y_scaled.to_numpy().ravel())

y_pred_test_scaled = model.predict(test_X_scaled)

if scaling == "unscaled":
    y_pred_test = y_pred_test_scaled

else:
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

mae_fold = mean_absolute_error(test_y, y_pred_test)

toml_data = neural_params
toml_data["hidden_layer_sizes"] = str(neural_params["hidden_layer_sizes"])
toml_data["mae_fold"] = mae_fold
toml_data["neural_target"] = neural_target

toml_data["material"] = material
toml_data["defect_type"] = defect_type
toml_data["target_variable"] = target_variable

toml_path = f"../sbatch/neural_toml_results/{material}_{defect_type}_{target_variable}_{toml_data['hidden_layer_sizes']}_{toml_data['activation']}_{toml_data['solver']}_{toml_data['alpha']}_fold{fold}.toml"
with open(toml_path, "w", encoding="utf-8") as file:
    toml.dump(toml_data, file)

