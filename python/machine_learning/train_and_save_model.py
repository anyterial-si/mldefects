"""Train the best models on all data and save them to saved_models/ (not used for the paper's results).

Note: the saved WL interstitial models do not match the paper (see README).
"""
from pathlib import Path

from joblib import dump

import numpy as np
import toml

import argparse
import os
import sys
import pandas as pd

from common import ensure_dir, load_datasets, target_folder as _target_folder
from input_parameters import FEATURES

parser = argparse.ArgumentParser()

parser.add_argument(
    "--descriptor",
    choices=[
        "sic_no_wyckoff",
        "sic_wyckoff_letter",
        "sic_wyckoff",
    ],
    default="sic_wyckoff",
)
parser.add_argument(
    "--defect_type",
    choices=[
        "vacancy-substitutional",
        "interstitial",
    ],
    default="vacancy-substitutional",
)
parser.add_argument(
    "--target_variable",
    choices=[
        "FE",
        "ZPL",
    ],
    default="FE",
)

args = parser.parse_args()

descriptor = args.descriptor
defect_type = args.defect_type
target_variable = args.target_variable

model_targets = {
    "ZPL": {
        "vacancy-substitutional": {
            "sic_no_wyckoff": {
                "model": "kernel",
                "scaling": "unscaled",
                },
            "sic_wyckoff": {
                "model": "kernel",
                "scaling" :"unscaled",
                },
            "sic_wyckoff_letter": {
                "model": "kernel",
                "scaling": "unscaled",
                },
        },
        "interstitial": {
            "sic_no_wyckoff": {
                "model": "kernel",
                "scaling": "unscaled",
                },
            "sic_wyckoff": {
                "model": "kernel",
                "scaling": "unscaled",
                },
            "sic_wyckoff_letter": {
                "model": "kernel",
                "scaling": "unscaled",
                },
        },
    },
    "FE": {
        "vacancy-substitutional": {
            "sic_no_wyckoff": {
                "model": "neural",
                "scaling": "standard",
                },
            "sic_wyckoff": {
                "model": "neural",
                "scaling": "standard",
                },
            "sic_wyckoff_letter": {
                "model": "neural",
                "scaling": "standard",
                },
        },
        "interstitial": {
            "sic_no_wyckoff": {
                "model": "neural",
                "scaling": "standard",
                },
            "sic_wyckoff": {
                "model": "neural",
                "scaling": "standard",
                },
            "sic_wyckoff_letter": {
                "model": "neural",
                "scaling": "standard",
                },
        },
    },
}

model_type = model_targets[target_variable][defect_type][descriptor]["model"]
scaling = model_targets[target_variable][defect_type][descriptor]["scaling"]

if model_type == "neural":

    neural_hyperparameters = {
        "FE": {
            "vacancy-substitutional": {
                "sic_no_wyckoff": {
                    'hidden_layer_sizes': [24, 24, 24, 24, 24, 24, 24, 24],
                    'activation': 'tanh',
                    'solver': 'lbfgs',
                    'alpha': 0.01,
                    'max_iter': 9999999999999,
                    'tol': 0.0001,
                    'max_fun': 9999999999999,
                    'random_state': 42
                    },
                "sic_wyckoff_letter": {
                    'hidden_layer_sizes': [43, 43, 43, 43, 43, 43],
                    'activation': 'tanh',
                    'solver': 'lbfgs',
                    'alpha': 0.0001,
                    'max_iter': 9999999999999,
                    'tol': 0.0001,
                    'max_fun': 9999999999999,
                    'random_state': 42
                    },
                "sic_wyckoff": {
                    'hidden_layer_sizes': [36, 36, 36, 36, 36],
                    'activation': 'tanh',
                    'solver': 'lbfgs',
                    'alpha': 1e-3,
                    'max_iter': 9999999999999,
                    'tol': 0.0001,
                    'max_fun': 9999999999999,
                    'random_state': 42
                    },
                },
            "interstitial": {
                "sic_no_wyckoff": {
                    'hidden_layer_sizes': [20, 20, 20],
                    'activation': 'tanh',
                    'solver': 'lbfgs',
                    'alpha': 0.1,
                    'max_iter': 9999999999999,
                    'tol': 0.0001,
                    'max_fun': 9999999999999,
                    'random_state': 42
                    },
                "sic_wyckoff_letter": {
                    'hidden_layer_sizes': [48, 48, 48, 48, 48],
                    'activation': 'tanh',
                    'solver': 'lbfgs',
                    'alpha': 0.1,
                    'max_iter': 9999999999999,
                    'tol': 0.0001,
                    'max_fun': 9999999999999,
                    'random_state': 42
                    },
                "sic_wyckoff": {
                    'hidden_layer_sizes': [80, 80, 80, 80],
                    'activation': 'tanh',
                    'solver': 'lbfgs',
                    'alpha': 0.1,
                    'max_iter': 9999999999999,
                    'tol': 0.0001,
                    'max_fun': 9999999999999,
                    'random_state': 42
                    },
                }
        },
    }

    mlp_params = neural_hyperparameters[target_variable][defect_type][descriptor]

target_folder = _target_folder(descriptor, defect_type, target_variable)
features = FEATURES[descriptor]
data, data_single = load_datasets(descriptor, defect_type, columns=features + [target_variable])
saved_models_dir = ensure_dir(Path(__file__).resolve().parent / "saved_models")

if model_type == "kernel":
    from sklearn.kernel_ridge import KernelRidge
elif model_type == "neural":
    from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

if scaling == "standard":
    from sklearn.preprocessing import StandardScaler
elif scaling == "normal":
    from sklearn.preprocessing import MinMaxScaler


if model_type == "kernel":
    try:
        toml_data_params = toml.load(target_folder / "optim_results_kernel_unscaled.toml")
        best_params = toml_data_params["best_params"]
        best_params["kernel"] = "rbf"
        model = KernelRidge(**best_params)

    except (FileNotFoundError, KeyError):
        print("\nKernel model not optimised\n")
        sys.exit(0)

elif model_type == "neural":
    try:
        model = MLPRegressor(**mlp_params)

    except Exception:
        print("\ngive neural net params\n")
        sys.exit(0)

train_data = pd.concat([data, data_single], ignore_index=True)

train_X = train_data[features]
train_y = train_data[[target_variable]]

if scaling == "unscaled":
    train_X_scaled = train_data[features]
    train_y_scaled = train_data[target_variable]

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

    train_X_scaled = pd.DataFrame(train_X_scaled, columns=train_X.columns)
    train_y_scaled = pd.DataFrame(train_y_scaled, columns=train_y.columns)

if model_type in ["linear", "kernel"]:
    model.fit(train_X_scaled, train_y_scaled)
else:
    model.fit(train_X_scaled, train_y_scaled.squeeze())

if not scaling == "unscaled":
    dump(scaler_X, saved_models_dir / f"{descriptor}_{defect_type}_{target_variable}_scaler_X.joblib")
    dump(scaler_y, saved_models_dir / f"{descriptor}_{defect_type}_{target_variable}_scaler_y.joblib")

dump(model, saved_models_dir / f"{descriptor}_{defect_type}_{target_variable}_model.joblib")
