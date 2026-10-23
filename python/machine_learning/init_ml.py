
import ast
import argparse
import glob
import os
import sys
import json

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "--material",
    choices=[
        "sic_no_wyckoff",
        "sic_wyckoff",
        "sic_wyckoff_nw_data",
        "sic_wyckoff_letter",
        "sic_wyckoff_letter_nw_data",
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
parser.add_argument(
    "--model",
    type=str,
    choices=[
        "linear",
        "kernel",
        "neural",
    ],
    default="linear",
)
parser.add_argument(
    "--fold",
    type=int,
    required=False,
)
parser.add_argument(
    "--train_data_percentage",
    type=int,
    required=False,
)
parser.add_argument(
    "--scaling",
    type=str,
    choices=[
        "unscaled",
        "standard",
        "normal",
    ],
    default="standard",
)
parser.add_argument(
    "--cross_validation",
    type=str,
    choices=[
        "varied",
        "10-fold",
    ],
    default="varied",
)
parser.add_argument(
    "--gamma",
    type=float,
    required=False,
)
parser.add_argument(
    "--alphas_argparse",
    type=ast.literal_eval,
    required=False,
)
parser.add_argument(
    "--kernel_optim_target",
    type=str,
        choices=[
        "",
        "_finer",
        "_unscaled",
        "_unscaled_finer",
        "_normal",
    ],
    default="",
)
parser.add_argument(
    "--kernel_optim_alpha_index",
    type=int,
    required=False,
)
parser.add_argument(
    "--neural_params",
    type=str,
    required=False,
)
parser.add_argument(
    "--neural_target",
    choices=[
        "",
        "_normal",
        "_clean",
    ],
    type=str,
    default="",
)

args = parser.parse_args()

material = args.material
defect_type = args.defect_type
target_variable = args.target_variable
model_type = args.model
kernel_optim_target = args.kernel_optim_target
scaling = args.scaling
cross_validation = args.cross_validation
alphas_argparse = args.alphas_argparse
neural_target = args.neural_target
try:
    fold = args.fold
except AttributeError:
    pass
try:
    train_data_percentage = args.train_data_percentage
except AttributeError:
    pass
try:
    gamma = args.gamma
except AttributeError:
    pass
try:
    kernel_optim_alpha_index = args.kernel_optim_alpha_index
except AttributeError:
    pass
try:
    neural_params = json.loads(args.neural_params)
except (AttributeError, TypeError):
    pass

# create folder paths used
python_folder = os.path.join(os.getcwd(), "..")
database_folder = os.path.join(python_folder, "database")
material_folder = os.path.join(python_folder, "..", f"results/{material}")
defect_folder = os.path.join(material_folder, defect_type)
target_folder = os.path.join(defect_folder, target_variable)
if cross_validation == "varied":
    toml_path = os.path.join(target_folder, "results.toml")
else:
    toml_path = os.path.join(target_folder, "results_10_fold.toml")

toml_scatter_path = os.path.join(target_folder, "results_scatter.toml")

# import features
sys.path.append(database_folder)
if material == "diamond":
    from input_parameters import diamond_input_parameters as features  # type: ignore
if material == "diamond_no_vac":
    from input_parameters import diamond_input_parameters as features  # type: ignore
elif material == "diamond_geometry":
    from input_parameters import diamond_geometry_input_parameters as features  # type: ignore
elif material == "diamond_geometry_distance":
    from input_parameters import diamond_geometry_distance_input_parameters as features  # type: ignore
elif material == "diamond_geometry_distance_charge":
    from input_parameters import diamond_geometry_distance_charge_input_parameters as features  # type: ignore
elif material == "diamond_geometry_distance_spin":
    from input_parameters import diamond_geometry_distance_spin_input_parameters as features  # type: ignore
elif material == "diamond_geometry_charge_spin":
    from input_parameters import diamond_geometry_charge_spin_input_parameters as features  # type: ignore
elif material == "diamond_geometry_charge":
    from input_parameters import diamond_geometry_charge_input_parameters as features  # type: ignore
elif material == "diamond_geometry_spin":
    from input_parameters import diamond_geometry_spin_input_parameters as features  # type: ignore
elif material == "diamond_no_atom":
    from input_parameters import diamond_input_parameters as features  # type: ignore
elif material == "diamond_no_atom_no_vac":
    from input_parameters import diamond_input_parameters as features  # type: ignore
elif material == "diamond_atom":
    from input_parameters import atom_input_parameters as features  # type: ignore
elif material == "diamond_atom_distance":
    from input_parameters import atom_distance_input_parameters as features  # type: ignore
elif material == "diamond_atom_charge_spin":
    from input_parameters import atom_charge_spin_input_parameters as features  # type: ignore
elif material == "diamond_atom_charge":
    from input_parameters import atom_charge_input_parameters as features  # type: ignore
elif material == "diamond_atom_spin":
    from input_parameters import atom_spin_input_parameters as features  # type: ignore
elif material == "diamond_atom_distance_charge_spin":
    from input_parameters import atom_distance_charge_spin_input_parameters as features  # type: ignore
elif material == "diamond_atom_distance_charge":
    from input_parameters import atom_distance_charge_input_parameters as features  # type: ignore
elif material == "diamond_atom_distance_spin":
    from input_parameters import atom_distance_spin_input_parameters as features  # type: ignore
elif material == "sic_wyckoff":
    from input_parameters import sic_input_parameters as features  # type: ignore
elif material == "sic_wyckoff_nw_data":
    from input_parameters import sic_input_parameters as features  # type: ignore
elif material == "sic_wyckoff_letter":
    from input_parameters import sic_wyckoff_letter_input_parameters as features  # type: ignore
elif material == "sic_wyckoff_letter_nw_data":
    from input_parameters import sic_wyckoff_letter_input_parameters as features  # type: ignore
elif material == "sic_no_vac":
    from input_parameters import sic_input_parameters as features  # type: ignore
elif material == "sic_geometry":
    from input_parameters import sic_geometry_input_parameters as features  # type: ignore
elif material == "sic_geometry_distance":
    from input_parameters import sic_geometry_distance_input_parameters as features  # type: ignore
elif material == "sic_geometry_distance_charge":
    from input_parameters import sic_geometry_distance_charge_input_parameters as features  # type: ignore
elif material == "sic_geometry_distance_spin":
    from input_parameters import sic_geometry_distance_spin_input_parameters as features  # type: ignore
elif material == "sic_geometry_charge_spin":
    from input_parameters import sic_geometry_charge_spin_input_parameters as features  # type: ignore
elif material == "sic_geometry_charge":
    from input_parameters import sic_geometry_charge_input_parameters as features  # type: ignore
elif material == "sic_geometry_spin":
    from input_parameters import sic_geometry_spin_input_parameters as features  # type: ignore
elif material == "sic_no_atom":
    from input_parameters import sic_input_parameters as features  # type: ignore
elif material == "sic_no_atom_no_vac":
    from input_parameters import sic_input_parameters as features  # type: ignore
elif material == "sic_atom":
    from input_parameters import atom_input_parameters as features  # type: ignore
elif material == "sic_atom_distance":
    from input_parameters import atom_distance_input_parameters as features  # type: ignore
elif material == "sic_atom_charge_spin":
    from input_parameters import atom_charge_spin_input_parameters as features  # type: ignore
elif material == "sic_atom_charge":
    from input_parameters import atom_charge_input_parameters as features  # type: ignore
elif material == "sic_atom_spin":
    from input_parameters import atom_spin_input_parameters as features  # type: ignore
elif material == "sic_no_wyckoff":
    from input_parameters import atom_distance_charge_spin_input_parameters as features  # type: ignore
elif material == "sic_atom_distance_charge":
    from input_parameters import atom_distance_charge_input_parameters as features  # type: ignore
elif material == "sic_atom_distance_spin":
    from input_parameters import atom_distance_spin_input_parameters as features  # type: ignore


csv_files = glob.glob(os.path.join(defect_folder, "*.csv"))
double_index = next(i for i, s in enumerate(csv_files) if s.endswith("double.csv"))
cols = features + [target_variable]
data = pd.read_csv(csv_files[double_index], usecols=cols)
data_single = pd.read_csv(csv_files[double_index - 1], usecols=cols)

# data = data.head(10)
# data_single = data_single.head(10)

# special cases of target variables
if target_variable == "ZPL_!=_0":
    data = data[data["ZPL"] != 0]
    data.rename(columns={"ZPL": "ZPL_!=_0"}, inplace=True)
    data_single = data_single[data_single["ZPL"] != 0]
    data_single.rename(columns={"ZPL": "ZPL_!=_0"}, inplace=True)
elif target_variable == "ZPL_>_0":
    data = data[data["ZPL"] > 0]
    data.rename(columns={"ZPL": "ZPL_>_0"}, inplace=True)
    data_single = data_single[data_single["ZPL"] > 0]
    data_single.rename(columns={"ZPL": "ZPL_>_0"}, inplace=True)
elif target_variable == "ZPL_>_0.5":
    data = data[data["ZPL"] > 0.5]
    data.rename(columns={"ZPL": "ZPL_>_0.5"}, inplace=True)
    data_single = data_single[data_single["ZPL"] > 0.5]
    data_single.rename(columns={"ZPL": "ZPL_>_0.5"}, inplace=True)
elif target_variable == "AE_>_0":
    data = data[data["AE"] > 0]
    data.rename(columns={"AE": "AE_>_0"}, inplace=True)
    data_single = data_single[data_single["AE"] > 0]
    data_single.rename(columns={"AE": "AE_>_0"}, inplace=True)

N = data.shape[0]
n = data_single.shape[0]
