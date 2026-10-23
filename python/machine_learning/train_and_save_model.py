from joblib import dump

import numpy as np
import toml

import argparse
import glob
import os
import sys
import pandas as pd

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

python_folder = os.path.join(os.getcwd(), "..")
database_folder = os.path.join(python_folder, "database")
material_folder = os.path.join(python_folder, "..", f"results/{descriptor}")
defect_folder = os.path.join(material_folder, defect_type)
target_folder = os.path.join(defect_folder, target_variable)

sys.path.append(database_folder)
if descriptor == "sic_wyckoff":
    from input_parameters import sic_input_parameters as features  # type: ignore
elif descriptor == "sic_wyckoff_nw_data":
    from input_parameters import sic_input_parameters as features  # type: ignore
elif descriptor == "sic_wyckoff_letter":
    from input_parameters import sic_wyckoff_letter_input_parameters as features  # type: ignore
elif descriptor == "sic_wyckoff_letter_nw_data":
    from input_parameters import sic_wyckoff_letter_input_parameters as features  # type: ignore
elif descriptor == "sic_no_wyckoff":
    from input_parameters import atom_distance_charge_spin_input_parameters as features  # type: ignore

csv_files = glob.glob(os.path.join(defect_folder, "*.csv"))
double_index = next(i for i, s in enumerate(csv_files) if s.endswith("double.csv"))
cols = features + [target_variable]
data = pd.read_csv(csv_files[double_index], usecols=cols)
data_single = pd.read_csv(csv_files[double_index - 1], usecols=cols)

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
        toml_data_params = toml.load(os.path.join(target_folder, "optim_results_kernel_unscaled.toml"))
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
    dump(scaler_X, f"./saved_models/{descriptor}_{defect_type}_{target_variable}_scaler_X.joblib")
    dump(scaler_y, f"./saved_models/{descriptor}_{defect_type}_{target_variable}_scaler_y.joblib")

dump(model, f"./saved_models/{descriptor}_{defect_type}_{target_variable}_model.joblib")
