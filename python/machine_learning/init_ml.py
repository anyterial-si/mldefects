"""Command-line arguments and data loading shared by ml.py, data_average.py and optim_*.py.

Importing this module parses sys.argv and loads the double- and single-defect
datasets for the selected descriptor ("--material") and defect type.
"""

import argparse
import ast
import json
import os
import sys

import pandas as pd

from common import (
    DEFECT_TYPES,
    DESCRIPTORS,
    SCALINGS,
    TARGET_VARIABLES,
    load_datasets,
    target_folder as _target_folder,
)
from input_parameters import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--material", choices=DESCRIPTORS, default="sic_wyckoff",
                    help="descriptor / dataset (see README)")
parser.add_argument("--defect_type", choices=DEFECT_TYPES, default="vacancy-substitutional")
parser.add_argument("--target_variable", choices=TARGET_VARIABLES, default="FE")
parser.add_argument("--model", choices=["linear", "kernel", "neural"], default="linear")
parser.add_argument("--scaling", choices=SCALINGS, default="standard")
parser.add_argument("--fold", type=int, required=False,
                    help="1-based index of the cross-validation fold combination")
parser.add_argument("--train_data_percentage", type=int, required=False,
                    help="number of the 10 folds used for training (1-9), i.e. training fraction in tenths")
parser.add_argument("--scatter_fold", type=int, default=10,
                    help="ml.py: fold (at 9/10 training data) whose test predictions are saved for the scatter plot")
parser.add_argument("--gamma", type=float, required=False)
parser.add_argument("--alphas_argparse", type=ast.literal_eval, required=False)
parser.add_argument("--kernel_optim_target", choices=["", "_unscaled", "_normal"], default="",
                    help="suffix of the optim_results_kernel*.toml file the job belongs to")
parser.add_argument("--kernel_optim_alpha_index", type=int, required=False)
parser.add_argument("--neural_params", type=str, required=False,
                    help="JSON dict of MLPRegressor parameters")
parser.add_argument("--neural_target", choices=["", "_clean"], default="",
                    help="suffix of the optim_results_neural*.db the job belongs to")

args = parser.parse_args()

material = args.material
defect_type = args.defect_type
target_variable = args.target_variable
model_type = args.model
scaling = args.scaling
fold = args.fold
train_data_percentage = args.train_data_percentage
scatter_fold = args.scatter_fold
gamma = args.gamma
alphas_argparse = args.alphas_argparse
kernel_optim_target = args.kernel_optim_target
kernel_optim_alpha_index = args.kernel_optim_alpha_index
neural_target = args.neural_target
neural_params = json.loads(args.neural_params) if args.neural_params else None

target_folder = _target_folder(material, defect_type, target_variable)
toml_path = target_folder / "results.toml"
toml_scatter_path = target_folder / "results_scatter.toml"

features = FEATURES[material]
data, data_single = load_datasets(material, defect_type, columns=features + [target_variable])

N = data.shape[0]
n = data_single.shape[0]
