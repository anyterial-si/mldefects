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
    target_folder,
    target_variable,
    kernel_optim_target,
    kernel_optim_alpha_index
)
alphas = [10**e for e in np.linspace(-8, 2, 11)]
toml_optim_path = os.path.join(target_folder, f"optim_results_kernel{kernel_optim_target}.toml")
toml_data = toml.load(toml_optim_path)

best_mae = 9999999
best_gamma = -42
best_alpha = -12
for gamma_string in toml_data["scores"]:
    if min(toml_data["scores"][gamma_string]) < best_mae:
        best_mae = min(toml_data["scores"][gamma_string])
        best_index = toml_data["scores"][gamma_string].index(best_mae)
        best_gamma = gamma_string
        best_alpha = alphas[best_index]

# print(best_gamma)
# print(best_alpha)

toml_data["best_params"] = {
    "alpha": float(best_alpha),
    "gamma": float(best_gamma),
}

with open(toml_optim_path, "w", encoding="utf-8") as file:
    toml.dump(toml_data, file)
