"""Shared paths, constants and dataset loading for all scripts.

Paths are resolved relative to this file, so the scripts can be run from any
working directory. Set the environment variable MLDEFECTS_RESULTS_DIR to write
(and read) results somewhere other than <repo>/results, e.g. for testing.
"""

import os
from pathlib import Path

import pandas as pd

from input_parameters import FEATURES

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("MLDEFECTS_RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"
SUPPLEMENTARY_FIGURES_DIR = FIGURES_DIR / "supplementary"
TABLES_DIR = RESULTS_DIR / "tables"
JOBS_DIR = RESULTS_DIR / "jobs"  # one TOML per optimization job, see aggregate_results.py

DESCRIPTORS = [
    "sic_no_wyckoff",              # Atomic Descriptor
    "sic_wyckoff_letter",          # Wyckoff Letter Descriptor
    "sic_wyckoff",                 # Wyckoff Coordinate Descriptor
    "sic_wyckoff_letter_nw_data",  # Wyckoff Letter Compare Descriptor
    "sic_wyckoff_nw_data",         # Wyckoff Coordinate Compare Descriptor
]
DEFECT_TYPES = ["vacancy-substitutional", "interstitial"]
TARGET_VARIABLES = ["FE", "ZPL"]
SCALINGS = ["unscaled", "standard", "normal"]

# file-name suffix used for each scaling in results/*/optim_results_{model}{suffix}.toml
SCALING_SUFFIX = {"standard": "", "unscaled": "_unscaled", "normal": "_normal"}

# Kernel ridge gamma grid used in the paper (one optim_kernel.py job per gamma and alpha index;
# the 11 alphas 10^-8 ... 10^2 are defined in optim_kernel.py, the ridge alphas in optim_linear.py).
# 1/5, 1/15 and 1/59 are 1/(number of input parameters) of the three descriptors.
KERNEL_GAMMAS = [0.0, 1 / 59, 1 / 15, 1 / 5] + [10.0**e for e in range(-4, 5)]
N_KERNEL_ALPHAS = 11

N_SPLITS = 10
CV_RANDOM_STATE = 42


def dataset_path(descriptor, defect_type, size):
    """size is 'single' or 'double'."""
    return DATA_DIR / f"{descriptor}_{defect_type}_{size}_cleaned.csv"


def load_datasets(descriptor, defect_type, columns=None):
    """Return (double, single) DataFrames for a descriptor and defect type."""
    data = pd.read_csv(dataset_path(descriptor, defect_type, "double"), usecols=columns)
    data_single = pd.read_csv(dataset_path(descriptor, defect_type, "single"), usecols=columns)
    return data, data_single


def target_folder(descriptor, defect_type, target_variable):
    return RESULTS_DIR / descriptor / defect_type / target_variable


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
