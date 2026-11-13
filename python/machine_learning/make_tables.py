"""Paper Table 3 (all MAEs and optimal hyperparameters) and Table 4 (projection scores) as LaTeX.

Reads results/<case>/optim_results_{linear,kernel}*.toml, optim_results_neural_clean.db and
results.toml; writes results/tables/all_results_table.tex and results/tables/projections.tex.

With --from-weights, all MAEs (Table 3 and Table 4) are instead taken from results/<case>/cv_scores.toml,
which `python cv_models.py evaluate` computes from the saved per-fold model weights. The
hyperparameters shown are the same in both modes.
"""
import sqlite3
import toml
import os
import sys
import numpy as np
from fractions import Fraction
import ast

descriptors = [
    "sic_no_wyckoff",
    "sic_wyckoff_letter",
    "sic_wyckoff",
    "sic_wyckoff_letter_nw_data",
    "sic_wyckoff_nw_data",
]
defect_types = [
    "vacancy-substitutional",
    "interstitial"
]
target_variables = [
    "FE",
    "ZPL"
]

from common import SCALING_SUFFIX, TABLES_DIR, ensure_dir, target_folder

FROM_WEIGHTS = "--from-weights" in sys.argv[1:]


def weights_score(descriptor, defect_type, target_variable, name, kind="mean_test"):
    """Mean MAE over the 10 folds computed from the saved weights (cv_models.py evaluate)."""
    path = target_folder(descriptor, defect_type, target_variable) / "cv_scores.toml"
    try:
        return float(toml.load(path)[name][kind])
    except (FileNotFoundError, KeyError):
        raise SystemExit(f"{path}: no {name} scores; train the weights and run `python cv_models.py evaluate`")

def to_latex(x):

    if x == 1/5:
        return r"\(\displaystyle 1/5\)"
        # return r"\(\displaystyle \frac{1}{5}\)"
    elif x == 1/15:
        return r"\(\displaystyle 1/15\)"
        # return r"\(\displaystyle \frac{1}{15}\)"
    elif x == 1/59:
        return r"\(\displaystyle 1/59\)"
        # return r"\(\displaystyle \frac{1}{59}\)"
    elif x == 0:
        return r"\(\displaystyle 1\)"

    e = np.log10(x)
    k = round((e + 8) * 4)

    exponent = Fraction(-8, 1) + Fraction(k, 4)

    num = exponent.numerator
    den = exponent.denominator

    if den == 1:
        if num == 0:
            return r"\(\displaystyle 1\)"
            # pass
        return rf"\(10^{{\displaystyle {num}}}\)"
    elif num < 0:
        return rf"\(10^{{ \displaystyle - {-num}/{den} }}\)"
        # return rf"\(10^{{\displaystyle -\frac{{{-num}}}{{{den}}}}}\)"
    else:
        return rf"\(10^{{ \displaystyle {num}/{den} }}\)"
        # return rf"\(10^{{\displaystyle \frac{{{num}}}{{{den}}}}}\)"

neural_hyperparameters_dict = {
    "tanh": "Tanh",
    "relu": "ReLU",
    "logistic": "Logistic",

    "lbfgs": "L-BFGS",
    "sgd": "NAG",
    "adam": "Adam",
}

no_value = "\\(\\varnothing\\)"
# no_value = "--"

all_scores_dict = {
    "vacancy-substitutional": {
        "ridge": {
            "FE": [],
            "ZPL": [],
            },
        "kernel_ridge": {
            "FE": [],
            "ZPL": [],
            },
        "mlpregressor": {
            "FE": [],
            "ZPL": [],
            },
    },
    "interstitial": {
        "ridge": {
            "FE": [],
            "ZPL": [],
            },
        "kernel_ridge": {
            "FE": [],
            "ZPL": [],
            },
        "mlpregressor": {
            "FE": [],
            "ZPL": [],
            },
    }
}

optim_params_ridge_dict = {
    "vacancy-substitutional": {
        "ridge": {
            "FE": [],
            "ZPL": [],
        },
        "kernel_ridge": {
            "FE": {
                "alpha": [],
                "gamma": [],
                },
            "ZPL": {
                "alpha": [],
                "gamma": [],
                },
        },
    },
    "interstitial": {
        "ridge": {
            "FE": [],
            "ZPL": [],
        },
        "kernel_ridge": {
            "FE": {
                "alpha": [],
                "gamma": [],
                },
            "ZPL": {
                "alpha": [],
                "gamma": [],
                },
        },
    },
}

optim_params_ridge = [[],[],[],[],[],[],[],[],[],[]]
optim_params_neural = [[],[],[],[],[],[],[],[],[],[]]

n = -1
for defect_type in defect_types:
    for descriptor in descriptors:
        n = n + 1
        for scaling in ["_unscaled", "", "_normal"]:
            for target_variable in target_variables:


                results_folder = target_folder(descriptor, defect_type, target_variable) / f"optim_results_linear{scaling}.toml"
                results = toml.load(results_folder)
                all_scores_dict[defect_type]["ridge"][target_variable].append(
                    weights_score(descriptor, defect_type, target_variable, f"ridge{scaling}") if FROM_WEIGHTS
                    else float(min(results["scores"]["mae"])))
                optim_params_ridge_dict[defect_type]["ridge"][target_variable].append(to_latex(results["best_params"]["alpha"]))
                optim_params_ridge[n].append(to_latex(results["best_params"]["alpha"]))

            for target_variable in target_variables:

                try:
                    results_folder = target_folder(descriptor, defect_type, target_variable) / f"optim_results_kernel{scaling}.toml"
                    results = toml.load(results_folder)
                    all_scores_dict[defect_type]["kernel_ridge"][target_variable].append(
                        weights_score(descriptor, defect_type, target_variable, f"kernel{scaling}") if FROM_WEIGHTS
                        else float(min(results["scores"][str(results["best_params"]["gamma"])])))
                    optim_params_ridge[n].append(to_latex(results["best_params"]["alpha"]))
                    optim_params_ridge[n].append(to_latex(results["best_params"]["gamma"]))
                    optim_params_ridge_dict[defect_type]["kernel_ridge"][target_variable]["alpha"].append(to_latex(results["best_params"]["alpha"]))
                    optim_params_ridge_dict[defect_type]["kernel_ridge"][target_variable]["gamma"].append(to_latex(results["best_params"]["gamma"]))
                except Exception:
                    all_scores_dict[defect_type]["kernel_ridge"][target_variable].append(no_value)
                    optim_params_ridge[n].append(no_value)
                    optim_params_ridge[n].append(no_value)
                    optim_params_ridge_dict[defect_type]["kernel_ridge"][target_variable]["alpha"].append(no_value)
                    optim_params_ridge_dict[defect_type]["kernel_ridge"][target_variable]["gamma"].append(no_value)

            for target_variable in target_variables:
                if not scaling:
                    # Table 3 uses the "clean" re-evaluation database of the selected MLP configurations
                    db_optim_path = target_folder(descriptor, defect_type, target_variable) / "optim_results_neural_clean.db"

                    conn = sqlite3.connect(db_optim_path, timeout=3600)
                    conn.execute("PRAGMA locking_mode = EXCLUSIVE")
                    conn.execute("PRAGMA foreign_keys = ON")

                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT id
                        FROM hyperparameters
                        """,
                    )
                    hyperparameters_ids = cursor.fetchall()
                    neural_maeas = []
                    for hyperparameters_id in hyperparameters_ids:
                        cursor.execute(
                            """
                            SELECT fold_mae
                            FROM maes
                            WHERE hyperparameter_id = ?
                            """,
                            # """
                            # SELECT fold_mae
                            # FROM maes
                            # WHERE hyperparameter_id = ?
                            # LIMIT 10
                            # """,
                            hyperparameters_id
                        )
                        maes = cursor.fetchall()
                        neural_maeas.append(float(np.mean(maes)))

                    try:
                        neural_min_mae = float(min(neural_maeas))
                        all_scores_dict[defect_type]["mlpregressor"][target_variable].append(
                            weights_score(descriptor, defect_type, target_variable, "neural") if FROM_WEIGHTS
                            else neural_min_mae)
                        hyperparameters_index = neural_maeas.index(neural_min_mae)
                        hyperparameters_selected_id = hyperparameters_ids[hyperparameters_index]
                        cursor.execute(
                            """
                            SELECT hidden_layer_sizes, activation, solver, alpha
                            FROM hyperparameters
                            WHERE id = ?
                            """,
                            hyperparameters_selected_id
                        )
                        neural_hyperparameters = cursor.fetchall()
                        neural_hyperparameters = neural_hyperparameters[0]
                        neural_hyperparameters_append_list = [0] * 5
                        hidden_layers = ast.literal_eval(neural_hyperparameters[0])
                        neural_hyperparameters_append_list[0] = len(hidden_layers)
                        neural_hyperparameters_append_list[1] = hidden_layers[0]
                        neural_hyperparameters_append_list[2] = neural_hyperparameters_dict[neural_hyperparameters[1]]
                        neural_hyperparameters_append_list[3] = neural_hyperparameters_dict[neural_hyperparameters[2]]
                        neural_hyperparameters_append_list[4] = f"{neural_hyperparameters[3]:.0e}"
                        if defect_type == "vacancy-substitutional" and target_variable == "FE" and descriptor == "sic_no_wyckoff" or defect_type == "interstitial" and target_variable == "FE" and descriptor == "sic_wyckoff":
                            for i in range(0,5):
                                neural_hyperparameters_append_list[i] = f"\\textcolor{{mplgreen}}{{ {neural_hyperparameters_append_list[i]} }}"
                        optim_params_neural[n].append(neural_hyperparameters_append_list)

                    except ValueError:
                        all_scores_dict[defect_type]["mlpregressor"][target_variable].append(no_value)
                        optim_params_neural[n].append([no_value, no_value, no_value, no_value, no_value])

fe_vac_sub = all_scores_dict["vacancy-substitutional"]["ridge"]["FE"] + all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"] + all_scores_dict["vacancy-substitutional"]["mlpregressor"]["FE"]
zpl_vac_sub = all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"] + all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"] + all_scores_dict["vacancy-substitutional"]["mlpregressor"]["ZPL"]
fe_int = all_scores_dict["interstitial"]["ridge"]["FE"] + all_scores_dict["interstitial"]["kernel_ridge"]["FE"] + all_scores_dict["interstitial"]["mlpregressor"]["FE"]
zpl_int = all_scores_dict["interstitial"]["ridge"]["ZPL"] + all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"] + all_scores_dict["interstitial"]["mlpregressor"]["ZPL"]

fe_vac_sub_min = min([val for val in fe_vac_sub if val != no_value])
zpl_vac_sub_min = min([val for val in zpl_vac_sub if val != no_value])
fe_int_min = min([val for val in fe_int if val != no_value])
zpl_int_min = min([val for val in zpl_int if val != no_value])

# fe_mins = [fe_vac_sub_min, fe_int_min]
# zpl_mins = [zpl_vac_sub_min, zpl_int_min]
all_global_mins = [fe_vac_sub_min, fe_int_min, zpl_vac_sub_min, zpl_int_min]

fe_vac_sub_min = f"{fe_vac_sub_min:.3f}"
zpl_vac_sub_min = f"{zpl_vac_sub_min:.3f}"
fe_int_min = f"{fe_int_min:.3f}"
zpl_int_min = f"{zpl_int_min:.3f}"

def highlight_best(highlighted_value, model_color_func):
    if isinstance(highlighted_value, float):
        return f"\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{ \\textcolor{{{model_color_func}}}{{ {highlighted_value:.3f} }} }}"
    else:
        return f"\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{ \\textcolor{{{model_color_func}}}{{ {highlighted_value} }} }}"

def highlight_model(highlighted_value, model_color_func):
    if isinstance(highlighted_value, float):
        return f" \\textcolor{{{model_color}}}{{ {highlighted_value:.3f} }}"
    else:
        return f" \\textcolor{{{model_color}}}{{ {highlighted_value} }}"


for defect_type in all_scores_dict:
    for model_type in all_scores_dict[defect_type]:
        for target_variable in all_scores_dict[defect_type][model_type]:
            min_value = min(x for x in all_scores_dict[defect_type][model_type][target_variable] if isinstance(x, float))
            min_index = all_scores_dict[defect_type][model_type][target_variable].index(min_value)
            if model_type == "ridge":
                model_color = "mplblue"
            elif model_type == "kernel_ridge":
                model_color = "mplorange"
            elif model_type == "mlpregressor":
                model_color = "mplgreen"

            if min_value in all_global_mins:
                all_scores_dict[defect_type][model_type][target_variable][min_index] = highlight_best(all_scores_dict[defect_type][model_type][target_variable][min_index], model_color)
                if model_type == "ridge":
                    optim_params_ridge_dict[defect_type][model_type][target_variable][min_index] = highlight_best(optim_params_ridge_dict[defect_type][model_type][target_variable][min_index], model_color)
                elif model_type == "kernel_ridge":
                    optim_params_ridge_dict[defect_type][model_type][target_variable]["alpha"][min_index] = highlight_best(optim_params_ridge_dict[defect_type][model_type][target_variable]["alpha"][min_index], model_color)
                    optim_params_ridge_dict[defect_type][model_type][target_variable]["gamma"][min_index] = highlight_best(optim_params_ridge_dict[defect_type][model_type][target_variable]["gamma"][min_index], model_color)
            else:
                all_scores_dict[defect_type][model_type][target_variable][min_index] = highlight_model(all_scores_dict[defect_type][model_type][target_variable][min_index], model_color)
                if model_type == "ridge":
                    optim_params_ridge_dict[defect_type][model_type][target_variable][min_index] = highlight_model(optim_params_ridge_dict[defect_type][model_type][target_variable][min_index], model_color)
                elif model_type == "kernel_ridge":
                    optim_params_ridge_dict[defect_type][model_type][target_variable]["alpha"][min_index] = highlight_model(optim_params_ridge_dict[defect_type][model_type][target_variable]["alpha"][min_index], model_color)
                    optim_params_ridge_dict[defect_type][model_type][target_variable]["gamma"][min_index] = highlight_model(optim_params_ridge_dict[defect_type][model_type][target_variable]["gamma"][min_index], model_color)
            for i, _ in enumerate(all_scores_dict[defect_type][model_type][target_variable]):
                if i != min_index and isinstance(all_scores_dict[defect_type][model_type][target_variable][i], float):
                    all_scores_dict[defect_type][model_type][target_variable][i] = f"{all_scores_dict[defect_type][model_type][target_variable][i]:.3f}"


best_targets = {
    ('vacancy-substitutional', 'FE'): {
        "linear": {'scaling': 'normal', 'descriptor': 'sic_wyckoff_letter'},
        "kernel": {'scaling': 'unscaled', 'descriptor': 'sic_no_wyckoff'},
        "neural": {'scaling': 'standard', 'descriptor': 'sic_no_wyckoff'},
        },
    ('interstitial', 'FE'): {
        "linear": {'scaling': 'standard', 'descriptor': 'sic_wyckoff'},
        "kernel": {'scaling': 'unscaled', 'descriptor': 'sic_wyckoff'},
        "neural": {'scaling': 'standard', 'descriptor': 'sic_wyckoff'},
        },
    ('vacancy-substitutional', 'ZPL'): {
        "linear": {'scaling': 'standard', 'descriptor': 'sic_wyckoff_letter'},
        "kernel": {'scaling': 'unscaled', 'descriptor': 'sic_no_wyckoff'},
        "neural": {'scaling': 'standard', 'descriptor': 'sic_no_wyckoff'},
        },
    ('interstitial', 'ZPL'): {
        "linear": {'scaling': 'normal', 'descriptor': 'sic_wyckoff_nw_data'},
        "kernel": {'scaling': 'unscaled', 'descriptor': 'sic_no_wyckoff'},
        "neural": {'scaling': 'standard', 'descriptor': 'sic_no_wyckoff'},
        },
}

projection_scores = {
    ('vacancy-substitutional', 'FE'): {
        "linear": 0,
        "kernel": 0,
        "neural": 0,
        },
    ('interstitial', 'FE'): {
        "linear": 0,
        "kernel": 0,
        "neural": 0,
        },
    ('vacancy-substitutional', 'ZPL'): {
        "linear": 0,
        "kernel": 0,
        "neural": 0,
        },
    ('interstitial', 'ZPL'): {
        "linear": 0,
        "kernel": 0,
        "neural": 0,
        },
}
for defect_type in defect_types:
    for target_variable in target_variables:
        for model in ["linear", "kernel", "neural"]:
            best = best_targets[(defect_type, target_variable)][model]
            toml_path = target_folder(best["descriptor"], defect_type, target_variable) / "results.toml"
            toml_data = toml.load(toml_path)

            try:
                if FROM_WEIGHTS:
                    name = {"linear": "ridge", "kernel": "kernel", "neural": "neural"}[model] + SCALING_SUFFIX[best["scaling"]]
                    best_test_score = weights_score(best["descriptor"], defect_type, target_variable, name, "mean_test")
                    best_train_score = weights_score(best["descriptor"], defect_type, target_variable, name, "mean_train")
                elif model == "linear":
                    best_test_score = toml_data[f"MAE_percent_list_linear_test"][8]
                    best_train_score = toml_data[f"MAE_percent_list_linear_train"][8]
                else:
                    best_test_score = float(np.mean(toml_data[f"MAE_percent_list_{model}_test_90"]))
                    best_train_score = float(np.mean(toml_data[f"MAE_percent_list_{model}_train_90"]))

                projection = f"{float(np.mean([best_test_score, best_train_score])):.3f}"
            except KeyError:
                best_test_score = 0
                projection = 0

            projection_scores[(defect_type, target_variable)][model] = projection

projection_table = f"""
\\begin{{table}}
\\centering
\\begin{{tabular}}{{ll|ccc}}
\\textbf{{Defect Type}} & \\textbf{{Energy}} & \\textbf{{R}} & \\textbf{{KR}} & \\textbf{{MLP}} \\\\
\\midrule
\\multirow{{2}}{{*}}{{Vac.-Sub.}} & FE & {projection_scores[("vacancy-substitutional", "FE")]["linear"]} & {projection_scores[("vacancy-substitutional", "FE")]["kernel"]} & {projection_scores[("vacancy-substitutional", "FE")]["neural"]} \\\\
& ZPL & {projection_scores[("vacancy-substitutional", "ZPL")]["linear"]} & {projection_scores[("vacancy-substitutional", "ZPL")]["kernel"]} & {projection_scores[("vacancy-substitutional", "ZPL")]["neural"]} \\\\
\\midrule
\\multirow{{2}}{{*}}{{Int.}} & FE & {projection_scores[("interstitial", "FE")]["linear"]} & {projection_scores[("interstitial", "FE")]["kernel"]} & {projection_scores[("interstitial", "FE")]["neural"]} \\\\
& ZPL & {projection_scores[("interstitial", "ZPL")]["linear"]} & {projection_scores[("interstitial", "ZPL")]["kernel"]} & {projection_scores[("interstitial", "ZPL")]["neural"]} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Projection scores corresponding to the best performing combination of scaling and descriptor for the ridge (R), the kernel ridge (KR), and the multilayer perceptron (MLP) model for formation energy (FE) and zero-phonon line (ZPL) predictions in vacancies, substitutions, and interstitials. These projected scores correspond to the best scores of each model highlighted in Table~\\ref{{tab:all_results}}.}}
\\label{{tab:projections}}
\\end{{table}}
"""

with open(ensure_dir(TABLES_DIR) / "projections.tex", "w", encoding="utf-8") as file:
    file.write(projection_table)

# best_targets = {
#     ('vacancy-substitutional', 'FE'): {'model': 'neural', 'scaling': 'standard', 'descriptor': 'sic_no_wyckoff'},
#     ('interstitial', 'FE'): {'model': 'neural', 'scaling': 'standard', 'descriptor': 'sic_wyckoff'},
#     ('vacancy-substitutional', 'ZPL'): {'model': 'kernel', 'scaling': 'unscaled', 'descriptor': 'sic_no_wyckoff'},
#     ('interstitial', 'ZPL'): {'model': 'kernel', 'scaling': 'unscaled', 'descriptor': 'sic_no_wyckoff'}
# }

# best_test_scores = []
# projection_scores = []
# for defect_type in defect_types:
#     for target_variable in target_variables:
#         best = best_targets[(defect_type, target_variable)]
#         toml_path = target_folder(best["descriptor"], defect_type, target_variable) / "results.toml"
#         toml_data = toml.load(toml_path)

#         try:
#             best_test_score = float(np.mean(toml_data[f"MAE_percent_list_{best["model"]}_test_90"]))

#             best_train_score = float(np.mean(toml_data[f"MAE_percent_list_{best["model"]}_train_90"]))
#             if target_variable == "FE":
#                 projection = f"{float(np.mean([best_test_score, best_train_score])):.3f}"
#             else:
#                 projection = f"{float(np.mean([best_test_score, best_train_score])):.3f}"
#         except KeyError:
#             best_test_score = 0
#             projection = 0

#         best_test_scores.append(best_test_score)
#         projection_scores.append(projection)

# best_mae_table = f"""

# \\begin{{table}}
# \\centering

# \\begin{{subtable}}{{0.48\\textwidth}}
# \\centering
# \\begin{{tabular}}{{ll|cc}}
# \\toprule
# \\textbf{{Energy}} & \\textbf{{Defect Type}} & \\textbf{{Best Score}} & \\textbf{{Projection}} \\\\
# \\midrule
# \\multirow{{2}}{{*}}{{Formation Energy}}
# & Vacancy-Substitution & {fe_vac_sub_min} & {projection_scores[0]} \\\\
# & Interstitial      & {fe_int_min} & {projection_scores[2]} \\\\
# \\midrule
# \\multirow{{2}}{{*}}{{Zero-Phonon Line}}
# & Vacancy-Substitution & {zpl_vac_sub_min} & {projection_scores[1]} \\\\
# & Interstitial         & {zpl_int_min} & {projection_scores[3]} \\\\
# \\bottomrule
# \\end{{tabular}}
# \\caption{{}}
# \\label{{tab:best_maes}}
# \\end{{subtable}}

# \\hfill

# \\begin{{subtable}}{{0.48\\textwidth}}
# \\centering
# \\begin{{tabular}}{{ll|ccc}}
# \\textbf{{Energy}} & \\textbf{{Defect Type}} & \\textbf{{Descriptor}} & \\textbf{{Scaling}} & \\textbf{{Model}} \\\\
# \\midrule
# \\multirow{{2}}{{*}}{{FE}} & Vac.-Sub. & Atomic & Standard & Multilayer Perc.\\\\
# & Int. & W. Coordinate & Standard & Multilayer Perc.\\\\
# \\midrule
# \\multirow{{2}}{{*}}{{ZPL}} & Vac.-Sub. & Atomic & Unscaled & Kernel Ridge\\\\
# & Int. & Atomic & Unscaled & Kernel Ridge\\\\
# \\bottomrule
# \\end{{tabular}}
# % \\caption{{Best model configurations}}
# \\caption{{}}
# \\label{{tab:best_mae_combinations}}
# \\end{{subtable}}

# \\caption{{The best mean absolute error averages across 10-fold cross-validation and the corresponding projected performance for formation energy (FE) and zero-phonon line (ZPL) predictions for each defect type are shown in (a). The corresponding best combinations of descriptor, scaling, and machine learning model to achieve these best predictions are shown in (b).}}
# \\label{{tab:combined_results}}

# \\end{{table}}
# """

# with open("./../../latex/rapport/best_results.tex", "w", encoding="utf-8") as file:
#     file.write(best_mae_table)

all_table = f"""

\\begin{{table*}}
\\centering
\\scriptsize

% ===================== SUBTABLE 1 =====================
\\begin{{subtable}}{{\\textwidth}}
\\begin{{tabular}}{{ll
cccc|
cccccc|
cccc}}
\\toprule


& &
\\multicolumn{{4}}{{c|}}{{\\textbf{{Unscaled}}}} &
\\multicolumn{{6}}{{c|}}{{\\textbf{{Standardized}}}} &
\\multicolumn{{4}}{{c}}{{\\textbf{{Normalized}}}} \\\\

\\cmidrule(r){{3-6}} \\cmidrule(r){{7-12}} \\cmidrule(r){{13-16}}

& &
\\multicolumn{{2}}{{c}}{{\\textbf{{Ridge}}}} &
\\multicolumn{{2}}{{c|}}{{\\textbf{{Kernel Ridge}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Ridge}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Kernel Ridge}}}} &
\\multicolumn{{2}}{{c|}}{{\\textbf{{Multilayer Perceptron}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Ridge}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Kernel Ridge}}}} \\\\

\\cmidrule(r){{3-4}} \\cmidrule(r){{5-6}}
\\cmidrule(r){{7-8}} \\cmidrule(r){{9-10}} \\cmidrule(r){{11-12}}
\\cmidrule(r){{13-14}} \\cmidrule(r){{15-16}}

\\textbf{{Defect Type}} & \\textbf{{Descriptor}}
&
\\textbf{{FE}} & \\textbf{{ZPL}} &
\\textbf{{FE}} & \\textbf{{ZPL}} &
\\textbf{{FE}} & \\textbf{{ZPL}} &
\\textbf{{FE}} & \\textbf{{ZPL}} &
\\textbf{{FE}} & \\textbf{{ZPL}} &
\\textbf{{FE}} & \\textbf{{ZPL}} &
\\textbf{{FE}} & \\textbf{{ZPL}} \\\\

\\midrule

\\multirow{{5}}{{*}}{{Vacancy-Substitution}}
& Atomic
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][0]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][0]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][0]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][0]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][1]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][1]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][1]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][1]}  & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["FE"][0]} & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["ZPL"][0]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][2]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][2]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][2]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][2]}  \\\\

& Wyckoff Letter
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][3]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][3]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][3]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][3]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][4]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][4]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][4]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][4]}  & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["FE"][1]} & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["ZPL"][1]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][5]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][5]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][5]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][5]}  \\\\

& Wyckoff Coordinate
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][6]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][6]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][6]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][6]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][7]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][7]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][7]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][7]}  & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["FE"][2]} & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["ZPL"][2]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][8]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][8]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][8]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][8]}  \\\\

& Wyckoff Letter Compare
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][9]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][9]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][9]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][9]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][10]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][10]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][10]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][10]}  & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["FE"][3]} & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["ZPL"][3]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][11]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][11]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][11]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][11]}  \\\\

& Wyckoff Coordinate Compare
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][12]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][12]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][12]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][12]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][13]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][13]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][13]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][13]}  & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["FE"][4]} & {all_scores_dict["vacancy-substitutional"]["mlpregressor"]["ZPL"][4]}
& {all_scores_dict["vacancy-substitutional"]["ridge"]["FE"][14]} & {all_scores_dict["vacancy-substitutional"]["ridge"]["ZPL"][14]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["FE"][14]} & {all_scores_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"][14]}  \\\\

\\midrule

\\multirow{{5}}{{*}}{{Interstitial}}
& Atomic
& {all_scores_dict["interstitial"]["ridge"]["FE"][0]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][0]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][0]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][0]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][1]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][1]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][1]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][1]}  & {all_scores_dict["interstitial"]["mlpregressor"]["FE"][0]} & {all_scores_dict["interstitial"]["mlpregressor"]["ZPL"][0]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][2]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][2]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][2]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][2]}  \\\\

& Wyckoff Letter
& {all_scores_dict["interstitial"]["ridge"]["FE"][3]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][3]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][3]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][3]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][4]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][4]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][4]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][4]}  & {all_scores_dict["interstitial"]["mlpregressor"]["FE"][1]} & {all_scores_dict["interstitial"]["mlpregressor"]["ZPL"][1]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][5]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][5]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][5]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][5]}  \\\\

& Wyckoff Coordinate
& {all_scores_dict["interstitial"]["ridge"]["FE"][6]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][6]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][6]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][6]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][7]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][7]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][7]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][7]}  & {all_scores_dict["interstitial"]["mlpregressor"]["FE"][2]} & {all_scores_dict["interstitial"]["mlpregressor"]["ZPL"][2]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][8]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][8]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][8]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][8]}  \\\\

& Wyckoff Letter Compare
& {all_scores_dict["interstitial"]["ridge"]["FE"][9]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][9]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][9]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][9]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][10]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][10]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][10]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][10]}  & {all_scores_dict["interstitial"]["mlpregressor"]["FE"][3]} & {all_scores_dict["interstitial"]["mlpregressor"]["ZPL"][3]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][11]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][11]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][11]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][11]}  \\\\

& Wyckoff Coordinate Compare
& {all_scores_dict["interstitial"]["ridge"]["FE"][12]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][12]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][12]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][12]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][13]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][13]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][13]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][13]}  & {all_scores_dict["interstitial"]["mlpregressor"]["FE"][4]} & {all_scores_dict["interstitial"]["mlpregressor"]["ZPL"][4]}
& {all_scores_dict["interstitial"]["ridge"]["FE"][14]} & {all_scores_dict["interstitial"]["ridge"]["ZPL"][14]} & {all_scores_dict["interstitial"]["kernel_ridge"]["FE"][14]} & {all_scores_dict["interstitial"]["kernel_ridge"]["ZPL"][14]}  \\\\

\\bottomrule
\\end{{tabular}}
\\centering
\\caption{{All the best final scores for all the different cases.}}
\\label{{tab:all_scores}}
\\end{{subtable}}

\\vspace{{0.5cm}}

% ===================== SUBTABLE 2 =====================
\\begin{{subtable}}{{\\textwidth}}
\\begin{{tabular}}{{ll
cccccc|
cccccc|
cccccc}}
\\toprule

& &
\\multicolumn{{6}}{{c|}}{{\\textbf{{Unscaled}}}} &
\\multicolumn{{6}}{{c|}}{{\\textbf{{Standardized}}}} &
\\multicolumn{{6}}{{c}}{{\\textbf{{Normalized}}}} \\\\

\\cmidrule(r){{3-8}} \\cmidrule(r){{9-14}} \\cmidrule(r){{15-20}}

& &
\\multicolumn{{2}}{{c}}{{\\textbf{{Ridge}}}} &
\\multicolumn{{4}}{{c|}}{{\\textbf{{Kernel Ridge}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Ridge}}}} &
\\multicolumn{{4}}{{c|}}{{\\textbf{{Kernel Ridge}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Ridge}}}} &
\\multicolumn{{4}}{{c}}{{\\textbf{{Kernel Ridge}}}} \\\\

\\cmidrule(r){{3-4}} \\cmidrule(r){{5-8}}
\\cmidrule(r){{9-10}} \\cmidrule(r){{11-14}}
\\cmidrule(r){{15-16}} \\cmidrule(r){{17-20}}

& &
\\multicolumn{{1}}{{c}}{{\\textbf{{FE}}}} & \\multicolumn{{1}}{{c}}{{\\textbf{{ZPL}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{FE}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{ZPL}}}} &
\\multicolumn{{1}}{{c}}{{\\textbf{{FE}}}} & \\multicolumn{{1}}{{c}}{{\\textbf{{ZPL}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{FE}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{ZPL}}}} &
\\multicolumn{{1}}{{c}}{{\\textbf{{FE}}}} & \\multicolumn{{1}}{{c}}{{\\textbf{{ZPL}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{FE}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{ZPL}}}} \\\\

\\cmidrule(r){{3-3}} \\cmidrule(r){{4-4}}
\\cmidrule(r){{5-6}} \\cmidrule(r){{7-8}}
\\cmidrule(r){{9-9}} \\cmidrule(r){{10-10}}
\\cmidrule(r){{11-12}} \\cmidrule(r){{13-14}}
\\cmidrule(r){{15-15}} \\cmidrule(r){{16-16}}
\\cmidrule(r){{17-18}} \\cmidrule(r){{19-20}}

\\textbf{{Type}} & \\textbf{{Descriptor}}
&
\\multicolumn{{2}}{{c}}{{\\(\\alpha\\)}} &
\\(\\alpha\\) & \\(\\gamma\\) & \\(\\alpha\\) & \\(\\gamma\\) &
\\multicolumn{{2}}{{c}}{{\\(\\alpha\\)}} &
\\(\\alpha\\) & \\(\\gamma\\) & \\(\\alpha\\) & \\(\\gamma\\) &
\\multicolumn{{2}}{{c}}{{\\(\\alpha\\)}} &
\\(\\alpha\\) & \\(\\gamma\\) & \\(\\alpha\\) & \\(\\gamma\\) \\\\

\\midrule

\\multirow{{5}}{{*}}[-0.8em]{{V.-S.}}

& Atomic
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][0]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][0]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][0]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][0]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][0]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][0]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][1]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][1]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][1]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][1]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][1]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][1]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][2]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][2]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][2]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][2]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][2]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][2]} \\\\

& W. Letter
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][3]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][3]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][3]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][3]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][3]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][3]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][4]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][4]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][4]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][4]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][4]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][4]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][5]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][5]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][5]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][5]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][5]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][5]} \\\\

& W. Coor.
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][6]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][6]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][6]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][6]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][6]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][6]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][7]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][7]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][7]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][7]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][7]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][7]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][8]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][8]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][8]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][8]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][8]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][8]} \\\\

& W. Letter C.
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][9]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][9]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][9]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][9]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][9]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][9]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][10]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][10]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][10]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][10]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][10]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][10]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][11]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][11]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][11]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][11]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][11]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][11]} \\\\

& W. Coord. C.
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][12]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][12]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][12]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][12]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][12]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][12]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][13]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][13]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][13]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][13]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][13]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][13]}
& {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["FE"][14]} & {optim_params_ridge_dict["vacancy-substitutional"]["ridge"]["ZPL"][14]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["alpha"][14]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["FE"]["gamma"][14]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["alpha"][14]} & {optim_params_ridge_dict["vacancy-substitutional"]["kernel_ridge"]["ZPL"]["gamma"][14]} \\\\

\\midrule

\\multirow{{5}}{{*}}[-0.8em]{{Int.}}

& Atomic
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][0]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][0]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][0]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][0]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][0]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][0]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][1]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][1]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][1]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][1]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][1]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][1]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][2]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][2]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][2]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][2]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][2]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][2]} \\\\

& W. Letter
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][3]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][3]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][3]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][3]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][3]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][3]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][4]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][4]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][4]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][4]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][4]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][4]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][5]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][5]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][5]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][5]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][5]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][5]} \\\\

& W. Coor.
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][6]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][6]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][6]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][6]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][6]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][6]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][7]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][7]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][7]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][7]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][7]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][7]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][8]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][8]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][8]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][8]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][8]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][8]} \\\\

& W. Letter C.
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][9]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][9]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][9]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][9]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][9]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][9]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][10]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][10]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][10]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][10]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][10]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][10]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][11]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][11]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][11]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][11]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][11]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][11]} \\\\

& W. Coord. C.
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][12]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][12]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][12]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][12]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][12]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][12]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][13]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][13]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][13]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][13]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][13]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][13]}
& {optim_params_ridge_dict["interstitial"]["ridge"]["FE"][14]} & {optim_params_ridge_dict["interstitial"]["ridge"]["ZPL"][14]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["alpha"][14]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["FE"]["gamma"][14]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["alpha"][14]} & {optim_params_ridge_dict["interstitial"]["kernel_ridge"]["ZPL"]["gamma"][14]} \\\\

\\bottomrule
\\end{{tabular}}
\\centering
\\caption{{Best ridge and kernel ridge model hyperparameters. The ridge model optimizes the value of \\(\\alpha\\) and the kernel ridge model paired with the Gaussian kernel optimizes the value of \\(\\alpha\\) and \\(\\gamma\\).}}
\\label{{tab:ridge_best_hyperparameters}}
\\end{{subtable}}

\\vspace{{0.5cm}}

% ===================== SUBTABLE 3 =====================
\\begin{{subtable}}{{\\textwidth}}
\\centering
\\begin{{tabular}}{{ll|ccccc|ccccc}}
\\toprule


& &
\\multicolumn{{5}}{{c|}}{{\\textbf{{Formation Energy}}}} &
\\multicolumn{{5}}{{c}}{{\\textbf{{Zero-Phonon Line}}}} \\\\

\\cmidrule(r){{3-7}} \\cmidrule(r){{8-12}}

\\textbf{{Type}} & \\textbf{{Descriptor}}
& \\textbf{{Nr. Hidden Layers}} & \\textbf{{Nr. Neurons}} & \\textbf{{Activation}} & \\textbf{{Solver}} & \\(\\alpha\\)
& \\textbf{{Nr. Hidden Layers}} & \\textbf{{Nr. Neurons}} & \\textbf{{Activation}} & \\textbf{{Solver}} & \\(\\alpha\\) \\\\

\\midrule

\\multirow{{5}}{{*}}{{Vacancy-Substitution}}
& Atomic
& {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[0][0][0]}}}}}}} & {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[0][0][1]}}}}}}} & {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[0][0][2]}}}}}}} & {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[0][0][3]}}}}}}} & {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[0][0][4]}}}}}}}
& \\textcolor{{mplgreen}}{{{optim_params_neural[0][1][0]}}} & \\textcolor{{mplgreen}}{{{optim_params_neural[0][1][1]}}} & \\textcolor{{mplgreen}}{{{optim_params_neural[0][1][2]}}} & \\textcolor{{mplgreen}}{{{optim_params_neural[0][1][3]}}} & \\textcolor{{mplgreen}}{{{optim_params_neural[0][1][4]}}} \\\\

& Wyckoff Letter
& {optim_params_neural[1][0][0]} & {optim_params_neural[1][0][1]} & {optim_params_neural[1][0][2]} & {optim_params_neural[1][0][3]} & {optim_params_neural[1][0][4]}
& {optim_params_neural[1][1][0]} & {optim_params_neural[1][1][1]} & {optim_params_neural[1][1][2]} & {optim_params_neural[1][1][3]} & {optim_params_neural[1][1][4]} \\\\

& Wyckoff Coordinate
& {optim_params_neural[2][0][0]} & {optim_params_neural[2][0][1]} & {optim_params_neural[2][0][2]} & {optim_params_neural[2][0][3]} & {optim_params_neural[2][0][4]}
& {optim_params_neural[2][1][0]} & {optim_params_neural[2][1][1]} & {optim_params_neural[2][1][2]} & {optim_params_neural[2][1][3]} & {optim_params_neural[2][1][4]} \\\\

& Wyckoff Letter C.
& {optim_params_neural[3][0][0]} & {optim_params_neural[3][0][1]} & {optim_params_neural[3][0][2]} & {optim_params_neural[3][0][3]} & {optim_params_neural[3][0][4]}
& {optim_params_neural[3][1][0]} & {optim_params_neural[3][1][1]} & {optim_params_neural[3][1][2]} & {optim_params_neural[3][1][3]} & {optim_params_neural[3][1][4]} \\\\

& Wyckoff Coordinate C.
& {optim_params_neural[4][0][0]} & {optim_params_neural[4][0][1]} & {optim_params_neural[4][0][2]} & {optim_params_neural[4][0][3]} & {optim_params_neural[4][0][4]}
& {optim_params_neural[4][1][0]} & {optim_params_neural[4][1][1]} & {optim_params_neural[4][1][2]} & {optim_params_neural[4][1][3]} & {optim_params_neural[4][1][4]} \\\\

\\midrule

\\multirow{{5}}{{*}}{{Interstitial}}

& Atomic
& {optim_params_neural[5][0][0]} & {optim_params_neural[5][0][1]} & {optim_params_neural[5][0][2]} & {optim_params_neural[5][0][3]} & {optim_params_neural[5][0][4]}
& \\textcolor{{mplgreen}}{{{optim_params_neural[5][1][0]}}} & \\textcolor{{mplgreen}}{{{optim_params_neural[5][1][1]}}} & \\textcolor{{mplgreen}}{{{optim_params_neural[5][1][2]}}} & \\textcolor{{mplgreen}}{{{optim_params_neural[5][1][3]}}} & \\textcolor{{mplgreen}}{{{optim_params_neural[5][1][4]}}} \\\\

& Wyckoff Letter
& {optim_params_neural[6][0][0]} & {optim_params_neural[6][0][1]} & {optim_params_neural[6][0][2]} & {optim_params_neural[6][0][3]} & {optim_params_neural[6][0][4]}
& {optim_params_neural[6][1][0]} & {optim_params_neural[6][1][1]} & {optim_params_neural[6][1][2]} & {optim_params_neural[6][1][3]} & {optim_params_neural[6][1][4]} \\\\

& Wyckoff Coordinate
& {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[7][0][0]}}}}}}} & {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[7][0][1]}}}}}}} & {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[7][0][2]}}}}}}} & {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[7][0][3]}}}}}}} & {{\\setlength{{\\fboxsep}}{{1.5pt}}\\fcolorbox{{optimumcolor}}{{white}}{{\\textcolor{{mplgreen}}{{{optim_params_neural[7][0][4]}}}}}}}
& {optim_params_neural[7][1][0]} & {optim_params_neural[7][1][1]} & {optim_params_neural[7][1][2]} & {optim_params_neural[7][1][3]} & {optim_params_neural[7][1][4]} \\\\

& Wyckoff Letter C.
& {optim_params_neural[8][0][0]} & {optim_params_neural[8][0][1]} & {optim_params_neural[8][0][2]} & {optim_params_neural[8][0][3]} & {optim_params_neural[8][0][4]}
& {optim_params_neural[8][1][0]} & {optim_params_neural[8][1][1]} & {optim_params_neural[8][1][2]} & {optim_params_neural[8][1][3]} & {optim_params_neural[8][1][4]} \\\\

& Wyckoff Coordinate C.
& {optim_params_neural[9][0][0]} & {optim_params_neural[9][0][1]} & {optim_params_neural[9][0][2]} & {optim_params_neural[9][0][3]} & {optim_params_neural[9][0][4]}
& {optim_params_neural[9][1][0]} & {optim_params_neural[9][1][1]} & {optim_params_neural[9][1][2]} & {optim_params_neural[9][1][3]} & {optim_params_neural[9][1][4]} \\\\

\\bottomrule
\\end{{tabular}}
\\caption{{Best multilayer perceptron model hyperparameters. The multilayer perceptron model optimizes the number of hidden layers and the same number of neurons in each of them, the activation function, the solver, and the value of \\(\\alpha\\).}}
\\label{{tab:neural_best_hyperparameters}}
\\end{{subtable}}

% ===================== MAIN CAPTION =====================
\\caption{{All best mean absolute errors averaged across 10-fold cross-validation for each predicted energy, defect type, descriptor, scaling, and machine learning model are shown in (a). The corresponding hyperparameters for both the ridge and the kernel ridge model are shown in (b), and the equivalent for the multilayer perceptron model is shown in (c). Lastly, the best score for each predicted energy and each defect type, and the corresponding hyperparameters, are highlighted in different colors: blue for the best ridge model score, orange for the best kernel ridge model score, green for the best multilayer perceptron model score, and the red boxes are for the best overall scores.}}
\\label{{tab:all_results}}
\\end{{table*}}

"""

with open(ensure_dir(TABLES_DIR) / "all_results_table.tex", "w", encoding="utf-8") as file:
    file.write(all_table)
