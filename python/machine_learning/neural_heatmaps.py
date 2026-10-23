import ast
import os
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{newtxtext}
        \usepackage{newtxmath}
    """,
    "font.size": 12,
})

descriptors = [
    "sic_no_wyckoff",
    "sic_wyckoff_letter",
    "sic_wyckoff",
    # "sic_wyckoff_letter_nw_data",
    # "sic_wyckoff_nw_data",
]
defect_types = [
    "vacancy-substitutional",
    "interstitial",
]
target_variables = [
    "FE",
    "ZPL"
]

combo_dict = {
    # =========================
    # sic_no_wyckoff
    # =========================

    # vacancy-substitutional
    ("sic_no_wyckoff", "vacancy-substitutional", "FE"): {
        "activation": "tanh",
        "solver": "lbfgs",
        "optimal_nr_layers": 8,
        "alpha_start": -9,
        "alpha_end": 0,
        "nr_layer_start": 1,
        "nr_layer_end": 12,
        "gamma_layer": 0.4,
        "gamma_alpha": 0.8,
        "y_nr_ticks_layer": 3,
        "y_nr_ticks_alpha": 4,
    },
    ("sic_no_wyckoff", "vacancy-substitutional", "ZPL"): {
        "activation": "relu",
        "solver": "adam",
        "optimal_nr_layers": 4,
        "alpha_start": -7,
        "alpha_end": 0,
        "nr_layer_start": 1,
        "nr_layer_end": 19,
        "gamma_layer": 0.8,
        "gamma_alpha": 0.8,
        "y_nr_ticks_layer": 6,
        "y_nr_ticks_alpha": 5,
    },

    # interstitial
    ("sic_no_wyckoff", "interstitial", "FE"): {
        "activation": "tanh",
        "solver": "lbfgs",
        "optimal_nr_layers": 3,
        "alpha_start": -7,
        "alpha_end": 1,
        "nr_layer_start": 1,
        "nr_layer_end": 9,
        "gamma_layer": 0.4,
        "gamma_alpha": 0.35,
        "y_nr_ticks_layer": 3,
        "y_nr_ticks_alpha": 2,
    },
    ("sic_no_wyckoff", "interstitial", "ZPL"): {
        "activation": "relu",
        "solver": "adam",
        "optimal_nr_layers": 15,
        "alpha_start": -7,
        "alpha_end": 0,
        "nr_layer_start": 1,
        "nr_layer_end": 20,
        "gamma_layer": 0.8,
        "gamma_alpha": 0.6,
        "y_nr_ticks_layer": 6,
        "y_nr_ticks_alpha": 6,
    },

    # =========================
    # sic_wyckoff_letter
    # =========================

    # vacancy-substitutional
    ("sic_wyckoff_letter", "vacancy-substitutional", "FE"): {
        "activation": "tanh",
        "solver": "lbfgs",
        "optimal_nr_layers": 6,
        "alpha_start": -8,
        "alpha_end": 2,
        "nr_layer_start": 1,
        "nr_layer_end": 9,
        "gamma_layer": 0.4,
        "gamma_alpha": 0.4,
        "y_nr_ticks_layer": 3,
        "y_nr_ticks_alpha": 2,
    },
    ("sic_wyckoff_letter", "vacancy-substitutional", "ZPL"): {
        "activation": "relu",
        "solver": "adam",
        "optimal_nr_layers": 1,
        "alpha_start": -6,
        "alpha_end": 0,
        "nr_layer_start": 1,
        "nr_layer_end": 7,
        "gamma_layer": 0.7,
        "gamma_alpha": 0.9,
        "y_nr_ticks_layer": 5,
        "y_nr_ticks_alpha": 5,
    },

    # interstitial
    ("sic_wyckoff_letter", "interstitial", "FE"): {
        "activation": "tanh",
        "solver": "lbfgs",
        "optimal_nr_layers": 5,
        "alpha_start": -9,
        "alpha_end": 1,
        "nr_layer_start": 1,
        "nr_layer_end": 9,
        "gamma_layer": 0.6,
        "gamma_alpha": 0.4,
        "y_nr_ticks_layer": 4,
        "y_nr_ticks_alpha": 4,
    },
    ("sic_wyckoff_letter", "interstitial", "ZPL"): {
        "activation": "tanh",
        "solver": "adam",
        "optimal_nr_layers": 17,
        "alpha_start": -7,
        "alpha_end": 0,
        "nr_layer_start": 1,
        "nr_layer_end": 23,
        "gamma_layer": 0.7,
        "gamma_alpha": 0.7,
        "y_nr_ticks_layer": 3,
        "y_nr_ticks_alpha": 3,
    },

    # =========================
    # sic_wyckoff
    # =========================

    # vacancy-substitutional
    ("sic_wyckoff", "vacancy-substitutional", "FE"): {
        "activation": "tanh",
        "solver": "lbfgs",
        "optimal_nr_layers": 5,
        "alpha_start": -9,
        "alpha_end": 0,
        "nr_layer_start": 1,
        "nr_layer_end": 9,
        "gamma_layer": 0.5,
        "gamma_alpha": 0.6,
        "y_nr_ticks_layer": 3,
        "y_nr_ticks_alpha": 4,
    },
    ("sic_wyckoff", "vacancy-substitutional", "ZPL"): {
        "activation": "relu",
        "solver": "adam",
        "optimal_nr_layers": 2,
        "alpha_start": -7,
        "alpha_end": 1,
        "nr_layer_start": 1,
        "nr_layer_end": 9,
        "gamma_layer": 0.8,
        "gamma_alpha": 0.8,
        "y_nr_ticks_layer": 3,
        "y_nr_ticks_alpha": 4,
    },

    # interstitial
    ("sic_wyckoff", "interstitial", "FE"): {
        "activation": "tanh",
        "solver": "lbfgs",
        "optimal_nr_layers": 4,
        "alpha_start": -10,
        "alpha_end": 2,
        "nr_layer_start": 1,
        "nr_layer_end": 11,
        "gamma_layer": 0.7,
        "gamma_alpha": 0.6,
        "y_nr_ticks_layer": 4,
        "y_nr_ticks_alpha": 6,
    },
    ("sic_wyckoff", "interstitial", "ZPL"): {
        "activation": "tanh",
        "solver": "sgd",
        "optimal_nr_layers": 10,
        "alpha_start": -7,
        "alpha_end": 1,
        "nr_layer_start": 1,
        "nr_layer_end": 11,
        "gamma_layer": 0.6,
        "gamma_alpha": 0.7,
        "y_nr_ticks_layer": 5,
        "y_nr_ticks_alpha": 5,
    },

#     # =========================
#     # sic_wyckoff_letter_nw_data
#     # =========================

#     # vacancy-substitutional
#     ("sic_wyckoff_letter_nw_data", "vacancy-substitutional", "FE"): {
#         "activation": "tanh",
#         "solver": "lbfgs",
#         "optimal_nr_layers": 10,
#         "alpha_start": -11,
#         "alpha_end": 2,
#         "nr_layer_start": 1,
#         "nr_layer_end": 30,
#         "gamma_layer": 1,
#         "gamma_alpha": 1,
#         "y_nr_ticks_layer": 2,
#         "y_nr_ticks_alpha": 2,
#     },
#     ("sic_wyckoff_letter_nw_data", "vacancy-substitutional", "ZPL"): {
#         "activation": "relu",
#         "solver": "adam",
#         "optimal_nr_layers": 10,
#         "alpha_start": -11,
#         "alpha_end": 2,
#         "nr_layer_start": 1,
#         "nr_layer_end": 30,
#         "gamma_layer": 1,
#         "gamma_alpha": 1,
#         "y_nr_ticks_layer": 2,
#         "y_nr_ticks_alpha": 2,
#     },

#     # interstitial
#     ("sic_wyckoff_letter_nw_data", "interstitial", "FE"): {
#         "activation": "tanh",
#         "solver": "lbfgs",
#         "optimal_nr_layers": 10,
#         "alpha_start": -11,
#         "alpha_end": 2,
#         "nr_layer_start": 1,
#         "nr_layer_end": 30,
#         "gamma_layer": 1,
#         "gamma_alpha": 1,
#         "y_nr_ticks_layer": 2,
#         "y_nr_ticks_alpha": 2,
#     },
#     ("sic_wyckoff_letter_nw_data", "interstitial", "ZPL"): {
#         "activation": "tanh",
#         "solver": "adam",
#         "optimal_nr_layers": 10,
#         "alpha_start": -11,
#         "alpha_end": 2,
#         "nr_layer_start": 1,
#         "nr_layer_end": 30,
#         "gamma_layer": 1,
#         "gamma_alpha": 1,
#         "y_nr_ticks_layer": 2,
#         "y_nr_ticks_alpha": 2,
#     },
#     # =========================
#     # sic_wyckoff_nw_data
#     # =========================

#     # vacancy-substitutional
#     ("sic_wyckoff_nw_data", "vacancy-substitutional", "FE"): {
#         "activation": "tanh",
#         "solver": "lbfgs",
#         "optimal_nr_layers": 10,
#         "alpha_start": -11,
#         "alpha_end": 2,
#         "nr_layer_start": 1,
#         "nr_layer_end": 30,
#         "gamma_layer": 1,
#         "gamma_alpha": 1,
#         "y_nr_ticks_layer": 2,
#         "y_nr_ticks_alpha": 2,
#     },
#     ("sic_wyckoff_nw_data", "vacancy-substitutional", "ZPL"): {
#         "activation": "relu",
#         "solver": "adam",
#         "optimal_nr_layers": 10,
#         "alpha_start": -11,
#         "alpha_end": 2,
#         "nr_layer_start": 1,
#         "nr_layer_end": 30,
#         "gamma_layer": 1,
#         "gamma_alpha": 1,
#         "y_nr_ticks_layer": 2,
#         "y_nr_ticks_alpha": 2,
#     },

#     # interstitial
#     ("sic_wyckoff_nw_data", "interstitial", "FE"): {
#         "activation": "tanh",
#         "solver": "lbfgs",
#         "optimal_nr_layers": 10,
#         "alpha_start": -11,
#         "alpha_end": 2,
#         "nr_layer_start": 1,
#         "nr_layer_end": 30,
#         "gamma_layer": 1,
#         "gamma_alpha": 1,
#         "y_nr_ticks_layer": 2,
#         "y_nr_ticks_alpha": 2,
#     },
#     ("sic_wyckoff_nw_data", "interstitial", "ZPL"): {
#         "activation": "tanh",
#         "solver": "sgd",
#         "optimal_nr_layers": 10,
#         "alpha_start": -11,
#         "alpha_end": 2,
#         "nr_layer_start": 1,
#         "nr_layer_end": 30,
#         "gamma_layer": 1,
#         "gamma_alpha": 1,
#         "y_nr_ticks_layer": 2,
#         "y_nr_ticks_alpha": 2,
#     },
}

def calculate_mean_nr_folds(table):
    if not table:
        return table

    new_table = table
    for i,row in enumerate(table):
        new_table[i][-2] = round(row[-1] / sum(1 for alpha_value in row[1:-2] if alpha_value != " "), 1)
        new_table[i] = new_table[i][:-1]
    return new_table

def add_alpha_to_hidden_layer_row(hidden_layer_rows, new_hidden_layer, new_mae, alpha_index, nr_fold_maes_func):

    new_hidden_layer_rows = hidden_layer_rows
    added_hidden_layer_bool = False
    for i, row in enumerate(hidden_layer_rows):
        if row[0] == new_hidden_layer:
            new_hidden_layer_rows[i][alpha_index+1] = new_mae
            # new_hidden_layer_rows[i][alpha_index+1] = round(new_mae,nr_mae_decimals)
            new_hidden_layer_rows[i][-1] = row[-1] + nr_fold_maes_func
            added_hidden_layer_bool = True
            break

    if not added_hidden_layer_bool:
        alpha_initial = [" " for _ in alphas]
        alpha_initial[alpha_index] = new_mae
        # alpha_initial[alpha_index] = round(new_mae,nr_mae_decimals)
        new_hidden_layer_rows.append([new_hidden_layer] + alpha_initial + [0,nr_fold_maes_func])

    return new_hidden_layer_rows

font_size_heatmap = 47

for descriptor in descriptors:
    for defect_type in defect_types:
        for target_variable in target_variables:

            print(
                f"\nDescriptor: {descriptor}, Defect Type: {defect_type}, "
                f"Target Variable: {target_variable}"
            )

            db_optim_path = os.path.join(
                "./../../results",
                descriptor,
                defect_type,
                target_variable,
                "optim_results_neural.db",
            )


            conn = sqlite3.connect(db_optim_path, timeout=3600)
            conn.execute("PRAGMA locking_mode = EXCLUSIVE")
            conn.execute("PRAGMA foreign_keys = ON")

            cursor = conn.cursor()

            activation = combo_dict[(descriptor, defect_type, target_variable)]["activation"]
            solver = combo_dict[(descriptor, defect_type, target_variable)]["solver"]
            optimal_nr_layers = combo_dict[(descriptor, defect_type, target_variable)]["optimal_nr_layers"]
            alpha_start = combo_dict[(descriptor, defect_type, target_variable)]["alpha_start"]
            alpha_end = combo_dict[(descriptor, defect_type, target_variable)]["alpha_end"]
            nr_layer_start = combo_dict[(descriptor, defect_type, target_variable)]["nr_layer_start"]
            nr_layer_end = combo_dict[(descriptor, defect_type, target_variable)]["nr_layer_end"]
            y_nr_ticks_alpha = combo_dict[(descriptor, defect_type, target_variable)]["y_nr_ticks_alpha"]
            y_nr_ticks_layer = combo_dict[(descriptor, defect_type, target_variable)]["y_nr_ticks_layer"]
            gamma_layer = combo_dict[(descriptor, defect_type, target_variable)]["gamma_layer"]
            gamma_alpha = combo_dict[(descriptor, defect_type, target_variable)]["gamma_alpha"]

            # alphas = [-12,-11,-10,-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 10]
            alphas = [10.0**i for i in range(alpha_start, alpha_end)]
            default_alpha_index = alphas.index(0.0001)

            nr_mae_decimals = 5

            alpha_optim_bool = False
            first_alpha_bool = True
            # min_activation = []
            # best_min_mae_solver_activation = 9999999
            # next_best_min_mae_solver_activation = 9999999
            layer_tables_4 = {i: [] for i in range(1, 31)}
            layer_tables_10 = {i: [] for i in range(1, 31)}

            for alpha_index, alpha in enumerate(alphas):

                cursor.execute(
                    """
                    SELECT id, hidden_layer_sizes
                    FROM hyperparameters
                    WHERE activation = ?
                    AND solver = ?
                    AND alpha = ?
                    """,
                    (activation, solver, alpha),
                )

                hyperparams = cursor.fetchall()

                if hyperparams:
                    for hyper_id, hidden_layers in hyperparams:

                        cursor.execute(
                            # """
                            # SELECT fold_mae
                            # FROM maes
                            # WHERE hyperparameter_id = ?;
                            # """,
                            # """
                            # SELECT DISTINCT fold_mae
                            # FROM maes
                            # WHERE hyperparameter_id = ?;
                            # """,
                            f"""
                            SELECT fold_mae
                            FROM maes
                            WHERE hyperparameter_id = ?
                            LIMIT 10
                            """,
                            # f"""
                            # SELECT fold_mae
                            # FROM maes
                            # WHERE hyperparameter_id = ?
                            # LIMIT {nr_folds}
                            # """,
                            (hyper_id,),
                        )

                        fold_maes = [row[0] for row in cursor.fetchall()]

                        if fold_maes:
                            nr_fold_maes = len(fold_maes)
                            # if nr_folds != -1:
                            #     if nr_fold_maes != nr_folds:
                            #         continue
                            if nr_fold_maes >= 4:
                                mae_4 = sum(fold_maes[:4]) / 4
                            if nr_fold_maes == 10:
                                mae_10 = sum(fold_maes) / 10
                            # min_activation.append(mae_4)
                            # if alpha_index != default_alpha_index:
                                # alpha_optim_bool = True
                        else:
                            mae_4 = 0
                            mae_10 = 0

                        layers = ast.literal_eval(hidden_layers)

                        num_layers = len(layers)


                        layer_tables_4[num_layers] = add_alpha_to_hidden_layer_row(
                            layer_tables_4[num_layers],
                            hidden_layers,
                            mae_4,
                            alpha_index,
                            4
                        )
                        if nr_fold_maes == 10:
                            layer_tables_10[num_layers] = add_alpha_to_hidden_layer_row(
                                layer_tables_10[num_layers],
                                hidden_layers,
                                mae_10,
                                alpha_index,
                                10
                            )


            for num_layers, table in layer_tables_4.items():
                table.sort(
                    key=lambda row: int(row[0].strip('[]').split(',')[0]),
                    reverse=True
                )
                table = calculate_mean_nr_folds(table)
                layer_tables_4[num_layers] = table

            for num_layers, table in layer_tables_10.items():
                table.sort(
                    key=lambda row: int(row[0].strip('[]').split(',')[0]),
                    reverse=True
                )
                table = calculate_mean_nr_folds(table)
                layer_tables_10[num_layers] = table

            layers_nr_neurons_set = set()
            for num_layers, table in layer_tables_4.items():
                if table:
                    for row in table:
                        y_label = ast.literal_eval(row[0])[0]
                        layers_nr_neurons_set.add(y_label)
            layers_df = pd.DataFrame(np.nan, index=sorted(layers_nr_neurons_set, reverse=True), columns=range(nr_layer_start, nr_layer_end))

            for num_layers, table in layer_tables_4.items():
                if table:
                    for row in table:
                        y_label = ast.literal_eval(row[0])[0]
                        if row:
                            layers_df.loc[y_label,num_layers] = min([x for x in row[1:-1] if isinstance(x, float)])


            for num_layers, table in layer_tables_10.items():
                if num_layers == optimal_nr_layers and table:

                    alpha_rows = []
                    alpha_index = []

                    for row in table:
                        y_label = ast.literal_eval(row[0])[0]
                        alpha_index.append(y_label)

                        values = []
                        for v in row[1:-1]:
                            # print(v)
                            if v == ' ' or v == '':
                                values.append(np.nan)
                            else:
                                values.append(float(v))
                            # if v == best_min_mae_solver_activation:
                        alpha_rows.append(values)

                    df = pd.DataFrame(alpha_rows, index=alpha_index)
                    df.rename(
                        columns={i: str(alpha_start + i) for i in range(alpha_end - alpha_start)},
                        inplace=True
                    )
                    xticklabels = [rf"$10^{{{c}}}$" for c in df.columns]

                    # print(df)

                    plt.figure(figsize=(20, 12), dpi=300)
                    ax = sns.heatmap(df, cmap="plasma", annot=False, norm=PowerNorm(gamma=gamma_alpha))
                    ax.set_xticklabels(xticklabels)
                    min_val = df.min().min()
                    min_pos = np.where(df == min_val)
                    for y, x in zip(*min_pos):
                        plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=3))
                    # plt.xlabel(r"$\alpha = 10^{x}$", fontsize=font_size_heatmap, labelpad=20)
                    # plt.ylabel(f"Number of Neurons", fontsize=font_size_heatmap, labelpad=20)
                    plt.xticks(fontsize=font_size_heatmap)
                    plt.yticks(fontsize=font_size_heatmap)
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=font_size_heatmap)
                    labels = ax.get_yticklabels()
                    # hide every other label
                    for i, label in enumerate(labels):
                        if i % y_nr_ticks_alpha != 0:
                            label.set_visible(False)

                    ax.tick_params(axis='y', labelsize=font_size_heatmap)
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    ax.tick_params(axis='both',
                                    which='major',
                                    length=10,     # length of tick marks
                                    width=2,      # thickness
                                    color='black')  # tick color
                    # plt.title("Heatmap")
                    # plt.show()
                    plt.savefig(f"../../latex/rapport/figures/{descriptor}_{defect_type}_{target_variable}_neural_heatmap_alpha.pdf", dpi=300, bbox_inches="tight")
                    plt.close("all")
            # plasma
            # Blues

            plt.figure(figsize=(20, 12), dpi=300)
            ax = sns.heatmap(layers_df, cmap="plasma", annot=False, norm=PowerNorm(gamma=gamma_layer))
            min_val = layers_df.min().min()
            min_pos = np.where(layers_df == min_val)
            for y, x in zip(*min_pos):
                plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=3))
            # plt.xlabel("Number of Hidden Layers", fontsize=font_size_heatmap, labelpad=20)
            # plt.ylabel(f"Number of Neurons", fontsize=font_size_heatmap, labelpad=20)
            plt.xticks(fontsize=font_size_heatmap)
            plt.yticks(fontsize=font_size_heatmap)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=font_size_heatmap)
            labels = ax.get_yticklabels()
            for i, label in enumerate(labels):
                if i % y_nr_ticks_layer != 0:
                    label.set_visible(False)

            labels = ax.get_xticklabels()
            if len(labels) > 12:
                for i, label in enumerate(labels):
                    if i % 2 == 0:
                        label.set_visible(False)
            ax.tick_params(axis='y', labelsize=font_size_heatmap)
            plt.yticks(rotation=0)
            plt.xticks(rotation=0)
            ax.tick_params(axis='both',
                                which='major',
                                length=10,     # length of tick marks
                                width=2,      # thickness
                                color='black')  # tick color
            # plt.title("Heatmap")
            # plt.show()
            plt.savefig(f"../../latex/rapport/figures/{descriptor}_{defect_type}_{target_variable}_neural_heatmap_layer.pdf", dpi=300, bbox_inches="tight")
            plt.close("all")

            conn.close()
            # quit()
