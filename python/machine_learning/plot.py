import re
# import traceback
import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
# from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm

import matplotlib.ticker as mticker
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import toml
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "text.latex.preamble": r"\usepackage{times}",
#     "font.size": 14,
# })

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{newtxtext}
        \usepackage{newtxmath}
    """,
    "font.size": 12,
})

titles_bool = True

materials = [
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
    "ZPL",
]

material_title_dict = {
    "sic_wyckoff": "Wyckoff",
    "sic_no_wyckoff": "No Wyckoff",
    "sic_wyckoff_letter": "Wyckoff Letter",
    "sic_wyckoff_nw_data": "Wyckoff Compare",
    "sic_wyckoff_letter_nw_data": "Wyckoff Letter Compare"
}

gamma_dict = {
    "sic_no_wyckoff": {
        "vacancy-substitutional": {
            "FE": {"_unscaled": 0.2, "": 0.3, "_normal": 0.2},
            "ZPL": {"_unscaled": 0.15, "": 0.25, "_normal": 0.2},
        },
        "interstitial": {
            "FE": {"_unscaled": 0.11, "": 0.23, "_normal": 0.23},
            "ZPL": {"_unscaled": 0.1, "": 0.15, "_normal": 0.15},
        },
    },

    "sic_wyckoff_letter": {
        "vacancy-substitutional": {
            "FE": {"_unscaled": 0.3, "": 0.25, "_normal": 0.2},
            "ZPL": {"_unscaled": 0.2, "": 0.2, "_normal": 0.15},
        },
        "interstitial": {
            "FE": {"_unscaled": 0.12, "": 0.22, "_normal": 0.25},
            "ZPL": {"_unscaled": 0.075, "": 0.15, "_normal": 0.15},
        },
    },

    "sic_wyckoff": {
        "vacancy-substitutional": {
            "FE": {"_unscaled": 0.3, "": 0.2, "_normal": 0.25},
            "ZPL": {"_unscaled": 0.2, "": 0.15, "_normal": 0.15},
        },
        "interstitial": {
            "FE": {"_unscaled": 0.2, "": 0.2, "_normal": 0.12},
            "ZPL": {"_unscaled": 0.15, "": 0.15, "_normal": 0.075},
        },
    },
}

def gamma_to_str(gamma_key):
    import math

    # --- convert string → float if needed ---
    if isinstance(gamma_key, str):
        try:
            gamma_key = float(gamma_key)
        except ValueError:
            return gamma_key  # already a label like "1/5"

    special_map = {
        1/5: "1/5",
        1/15: "1/15",
        1/59: "1/59",
    }

    # match special values with tolerance
    for k, v in special_map.items():
        if abs(gamma_key - k) < 1e-12:
            return v

    # match powers of 10
    if gamma_key > 0:
        exp = round(math.log10(gamma_key))
        if abs(gamma_key - 10**exp) < 1e-12:
            return f"1e{int(exp)}"

    return str(gamma_key)

def gamma_str_to_float(x):
    # handle fractional strings like "1/5"
    if "/" in x:
        num, den = x.split("/")
        return float(num) / float(den)

    # handle scientific notation like "1e-3"
    if "e" in x or "E" in x:
        return float(x)

    # fallback
    return float(x)

def does_overlap(ax, corner_loc, legend):
    temp_text = AnchoredText("Line 1\nLine 2\nLine 3", loc=corner_loc, prop=dict(size=12), frameon=True)
    ax.add_artist(temp_text)
    plt.draw()
    text_bbox = temp_text.get_window_extent()
    legend_bbox = legend.get_window_extent()
    temp_text.remove()
    return text_bbox.overlaps(legend_bbox)

def to_latex_label(x):
    if "^" in x:
        base, exp = x.split("^")
        return rf"${base}^{{{exp}}}$"
    return x

def parse_number(s: str) -> float:
    """
    Convert strings like:
      '$10^{-2}$' -> 0.01
      '10^{-2}'   -> 0.01
      '1/59'      -> 0.016949...
    """
    s = s.strip().replace("$", "")

    # Case 1: fraction like "1/59"
    frac_match = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if frac_match:
        num = float(frac_match.group(1))
        den = float(frac_match.group(2))
        return num / den

    # Case 2: 10^exponent or 10^{-exponent}
    exp_match = re.fullmatch(r"10\^\{?(-?\d+)\}?", s)
    if exp_match:
        exp = int(exp_match.group(1))
        return 10 ** exp

    raise ValueError(f"Unsupported format: {s}")

def place_text_box_avoiding_legend_overlap(x, y, text, font_size, ax=None):
    if ax is None:
        ax = plt.gca()

    legend = ax.get_legend()
    if legend is None:
        legend_bbox = None
    else:
        plt.draw()
        legend_bbox = legend.get_window_extent()

    possible_locs = ['upper left', 'upper right', 'lower left', 'lower right']
    best_loc = None
    min_density = float('inf')

    xdata = (x - np.min(x)) / (np.max(x) - np.min(x))
    ydata = (y - np.min(y)) / (np.max(y) - np.min(y))

    corner_boxes = {
        'upper left':  ((0.0, 0.75), (0.25, 1.0)),
        'upper right': ((0.75, 0.75), (1.0, 1.0)),
        'lower left':  ((0.0, 0.0), (0.25, 0.25)),
        'lower right': ((0.75, 0.0), (1.0, 0.25)),
    }

    for loc in possible_locs:
        if legend and does_overlap(ax, loc, legend):
            continue
        (xmin, ymin), (xmax, ymax) = corner_boxes[loc]
        count = np.sum((xdata >= xmin) & (xdata <= xmax) & (ydata >= ymin) & (ydata <= ymax))
        if count < min_density:
            min_density = count
            best_loc = loc

    if best_loc is None:
        best_loc = 'upper left'

    text_box = AnchoredText(text, loc=best_loc, prop=dict(size=font_size, family='serif'), frameon=True)
    text_box.patch.set_boxstyle("round,pad=0.05,rounding_size=0.3")
    text_box.patch.set_facecolor("white")
    text_box.patch.set_edgecolor("lightgrey")
    text_box.patch.set_linewidth(0.8)
    text_box.patch.set_alpha(0.7)
    ax.add_artist(text_box)

for material in materials:
    # continue
    for defect_type in defect_types:
        for target_variable in target_variables:

            print(f"__________ {material} {defect_type} {target_variable} __________")

            path_dir = f"../../results/{material}/{defect_type}/{target_variable}"
            plot_data = toml.load(os.path.join(path_dir, "results.toml"))


            # linear optim
            for scaling in ["", "unscaled", "normal"]:
                plt.figure(figsize=(8, 5), dpi=300)
                if scaling:
                    linear_optim_path_dir = f"../../results/{material}/{defect_type}/{target_variable}/optim_results_linear_{scaling}.toml"
                else:
                    linear_optim_path_dir = f"../../results/{material}/{defect_type}/{target_variable}/optim_results_linear.toml"
                linear_optim_data = toml.load(linear_optim_path_dir)
                highlight_alpha = []
                try:
                    optim_alpha = linear_optim_data["best_params"]["alpha"]
                    highlight_alpha.append("black")
                    plt.plot(
                        linear_optim_data["scores"]["alphas"],
                        linear_optim_data["scores"]["mae"],
                        marker="o",
                        linestyle="-",
                        markersize=4,
                    )
                    if float(linear_optim_data["best_params"]["alpha"]) == 0:
                        optimal_alpha = "0"
                    else:
                        log_result = math.log10(float(linear_optim_data["best_params"]["alpha"]))
                        if log_result.is_integer():
                            optimal_alpha = "\\(10^{" + str(int(log_result)) + "}\\)"
                        else:
                            optimal_alpha = f"\\({float(linear_optim_data["best_params"]["alpha"]):.4f}\\)"

                    # Find the index of the minimum y
                    min_alpha_meas = linear_optim_data["scores"]["mae"]
                    min_index = min_alpha_meas.index(min(min_alpha_meas))
                    x_min = linear_optim_data["scores"]["alphas"][min_index]
                    y_min = min_alpha_meas[min_index]

                    # if scaling:
                    #     linear_optim_path_dir = f"../../results/{material}/{defect_type}/{target_variable}/optim_results_linear_{scaling}.toml"
                    # else:
                    #     linear_optim_path_dir = f"../../results/{material}/{defect_type}/{target_variable}/optim_results_linear.toml"
                    # linear_optim_data["best_params"]["alpha"] = x_min
                    # with open(linear_optim_path_dir, "w") as f:
                    #     toml.dump(linear_optim_data, f)

                    # Highlight the minimum point
                    plt.scatter(x_min, y_min, color="#DC143C", label=fr"Best \(\alpha\) = {optimal_alpha}", zorder=5, s=100)
                    # highlight_alpha.append("#DC143C")
                    # if material == "sic_wyckoff" and defect_type == "vacancy-substitutional" and target_variable == "ZPL" and scaling == "unscaled":
                    #     plt.annotate(f"{y_min:.3f}", (x_min, y_min), textcoords="offset points", xytext=(0, 15), ha='center', color="#DC143C")
                    # elif defect_type == "vacancy-substitutional" and target_variable == "ZPL":
                    #     plt.annotate(f"{y_min:.3f}", (x_min, y_min), textcoords="offset points", xytext=(30, -5), ha='center', color="#DC143C")
                    # else:
                    #     plt.annotate(f"{y_min:.3f}", (x_min, y_min), textcoords="offset points", xytext=(0, 15), ha='center', color="#DC143C")

                    # plt.xlabel(fr"Value of \(\alpha\)", labelpad=20, fontsize=fontsize_linear)
                    # plt.ylabel(f"Mean Absolute Error Averaged Across 10-Fold Cross-Validation", labelpad=20, fontsize=fontsize_linear)
                    plt.xscale("log")
                    plt.grid(True)
                    fontsize_linear = 22
                    plt.xticks(fontsize=fontsize_linear)
                    plt.yticks(fontsize=fontsize_linear)
                    if target_variable == "ZPL":
                        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
                    else:
                        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    # plt.legend(labelcolor=highlight_alpha)


                    # if titles_bool:
                    #     if scaling == "unscaled":
                    #         plt.title(f"{material_title_dict[material]} {defect_type} {target_variable} Linear Unscaled")
                    #     elif scaling == "normal":
                    #         plt.title(f"{material_title_dict[material]} {defect_type} {target_variable} Linear Normal")
                    #     else:
                    #         plt.title(f"{material_title_dict[material]} {defect_type} {target_variable} Linear Standard")
                    if scaling:
                        scaling = "_" + scaling
                    plt.tight_layout()
                    save_path = f"{path_dir}/{material}_{defect_type}_{target_variable}_optim_linear{scaling}.pdf"
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                    save_path = f"../../latex/rapport/figures_2/{material}_{defect_type}_{target_variable}_optim_linear{scaling}.pdf"
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")

                    plt.close('all')
                except KeyError as error:
                    print(error)
                    plt.close('all')

            # kernel optim
            y_lim_lower = 99
            exponents = range(-8, 3)
            cols = [f"10^{e}" for e in exponents]

            for kernel_specification in ["", "_unscaled", "_normal"]:
            # for kernel_specification in ["", "_finer", "_unscaled", "_unscaled_finer", "_normal"]:

                try:
                    gamma = gamma_dict[material][defect_type][target_variable][kernel_specification]
                except KeyError:
                    gamma = 0.3

                highlight_gamma = []

                styles = [
                    {"color": "b", "marker": "o", "linestyle": "-"},
                    {"color": "g", "marker": "s", "linestyle": "-"},
                    {"color": "purple", "marker": "D", "linestyle": "-"},
                    {"color": "c", "marker": "^", "linestyle": "-"},
                    {"color": "m", "marker": "v", "linestyle": "-"},
                    {"color": "y", "marker": "<", "linestyle": "-"},
                    {"color": "k", "marker": ">", "linestyle": "-"},
                    {"color": "saddlebrown", "marker": "p", "linestyle": "-"},
                    {"color": "brown", "marker": "*", "linestyle": "-"},
                    {"color": "gray", "marker": "h", "linestyle": "-"},
                    {"color": "lime", "marker": "x", "linestyle": "-"},
                    {"color": "navy", "marker": "+", "linestyle": "-"},
                    {"color": "darkred", "marker": ".", "linestyle": "-"},
                    {"color": "teal", "marker": "X", "linestyle": "-"},
                    {"color": "orange", "marker": "1", "linestyle": "-"},
                    {"color": "gold", "marker": "2", "linestyle": "-"},
                    {"color": "darkgreen", "marker": "3", "linestyle": "-"},
                    {"color": "skyblue", "marker": "4", "linestyle": "-"},
                    {"color": "indigo", "marker": "|", "linestyle": "-"},
                    {"color": "crimson", "marker": "_", "linestyle": "-"},
                    {"color": "olive", "marker": ".", "linestyle": "--"},
                    {"color": "chocolate", "marker": ",", "linestyle": "--"},
                    {"color": "dodgerblue", "marker": "o", "linestyle": ":"},
                    {"color": "deeppink", "marker": "s", "linestyle": "-."},
                    {"color": "lightcoral", "marker": "P", "linestyle": "-"},
                    {"color": "turquoise", "marker": "H", "linestyle": "-"},
                    {"color": "darkorange", "marker": "8", "linestyle": "-"},
                    {"color": "slateblue", "marker": "d", "linestyle": "-"},
                    {"color": "forestgreen", "marker": "|", "linestyle": "--"},
                    {"color": "royalblue", "marker": "_", "linestyle": "--"},
                    {"color": "firebrick", "marker": "1", "linestyle": ":"},
                    {"color": "mediumseagreen", "marker": "2", "linestyle": ":"},
                    {"color": "midnightblue", "marker": "3", "linestyle": "-."},
                    {"color": "orchid", "marker": "4", "linestyle": "-."},
                ]
                kernel_optim_path_dir = f"../../results/{material}/{defect_type}/{target_variable}/optim_results_kernel{kernel_specification}.toml"
                kernel_optim_data = toml.load(kernel_optim_path_dir)

                df = pd.DataFrame(columns=cols)

                try:
                    optim_gamma = kernel_optim_data["best_params"]["gamma"]
                    for i, gamma_key in enumerate(sorted(kernel_optim_data["scores"].keys(), key=float)):

                        # highlight optimal gamma in legend
                        if gamma_key == str(optim_gamma):
                            highlight_gamma.append("#DC143C")
                            label_addon = "Best "
                            style = {**styles[i], "color": "#DC143C", "zorder": 5}
                        else:
                            highlight_gamma.append("black")
                            label_addon = ""
                            style = styles[i]

                        # find the y limits for a readable plot
                        if y_lim_lower > min(kernel_optim_data["scores"][gamma_key]):
                            y_lim_lower = min(kernel_optim_data["scores"][gamma_key])

                        # create a latex friendly key for the label
                        if gamma_key == "0.0":
                            label_gamma_key = "0"

                            gamma_0_mean = float(np.mean(kernel_optim_data["scores"][gamma_key]))
                            ylim_addon = gamma_0_mean * 0.1

                        else:
                            log_result = math.log10(float(gamma_key))
                            if log_result.is_integer():
                                label_gamma_key = "10^" + str(int(log_result))
                            else:
                                label_gamma_key = f"{float(gamma_key):.4f}"



                        # plot
                        plt.plot(
                            kernel_optim_data["alphas"],
                            kernel_optim_data["scores"][gamma_key],
                            label=fr"{label_addon}\(\gamma\) = \({label_gamma_key}\)",
                            **style,
                            markersize=4,
                        )

                        if gamma_key == "0.0":
                            gamma_df = "0"
                        else:
                            gamma_df = gamma_to_str(gamma_key)

                        df.loc[gamma_df] = kernel_optim_data["scores"][gamma_key]

                        # create a latex friendly key for the label
                        if optim_gamma == 0:
                            best_gamma = "0"
                        else:
                            log_result = math.log10(optim_gamma)
                            if log_result.is_integer():
                                best_gamma = "\\(10^{" + str(int(log_result)) + "}\\)"
                            else:
                                best_gamma = f"\\({float(kernel_optim_data['best_params']['gamma']):.4f}\\)"
                        if float(kernel_optim_data["best_params"]["alpha"]) == 0:
                            optimal_alpha = "0"
                        else:
                            log_result = math.log10(float(kernel_optim_data["best_params"]["alpha"]))
                            if log_result.is_integer():
                                optimal_alpha = "\\(10^{" + str(int(log_result)) + "}\\)"
                            else:
                                optimal_alpha = f"\\({float(kernel_optim_data['best_params']['alpha']):.4f}\\)"

                    df_sorted = df.copy()
                    df_sorted["__sort_key__"] = df_sorted.index.map(gamma_str_to_float)

                    df_sorted = df_sorted.sort_values("__sort_key__").drop(columns="__sort_key__")

                    df = df_sorted
                    df.columns = [
                        c.decode() if isinstance(c, bytes) else str(c)
                        for c in df.columns
                    ]
                    df.columns = [to_latex_label(c) for c in df.columns]

                    df.columns = [
                        rf"$10^{{{k}}}$"
                        for k in range(-8, 3)
                    ]
                    # print(df)
                    df.index = [
                        rf"$10^{{{str(idx).replace("1e", "")}}}$"
                        if str(idx).startswith("1e")
                        else idx
                        for idx in df.index
                    ]
                    # print(df)

                    plt.close("all")
                    font_size_heatmap = 45
                    plt.figure(figsize=(20, 12), dpi=300)
                    ax = sns.heatmap(df, cmap="plasma", annot=False, norm=PowerNorm(gamma=gamma), vmax=6)
                    min_val = df.min().min()
                    min_pos = np.where(df == min_val)
                    for row_pos, col_pos in zip(*min_pos):
                        # print("Index:", df.index[row_pos], "best_gamma:", kernel_optim_data['best_params']['gamma'])
                        # print("Column:", df.columns[col_pos], "best_alpha:", kernel_optim_data['best_params']['alpha'])

                        kernel_optim_data["best_params"]["alpha"] = float(parse_number(df.columns[col_pos]))
                        kernel_optim_data["best_params"]["gamma"] = float(parse_number(df.index[row_pos]))

                        kernel_optim_path_dir = f"../../results/{material}/{defect_type}/{target_variable}/optim_results_kernel{kernel_specification}.toml"
                        with open(kernel_optim_path_dir, "w") as f:
                            toml.dump(kernel_optim_data, f)

                        # print("df_score:", df.iloc[row_pos, col_pos], "best_score:", kernel_optim_data["scores"][str(kernel_optim_data['best_params']['gamma'])][kernel_optim_data["alphas"].index([kernel_optim_data['best_params']['alpha']][0])])

                        # alpha = [kernel_optim_data['best_params']['alpha']]
                        # print(alpha[0])
                        # print(print(kernel_optim_data["alphas"]))
                        # print(kernel_optim_data["alphas"].index([kernel_optim_data['best_params']['alpha']][0]))
                        # pass
                    for y, x in zip(*min_pos):
                        plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=3))
                    # plt.ylabel(r"Value of \(\gamma\)", fontsize=font_size_heatmap, labelpad=20)
                    # plt.xlabel(r"Value of \(\alpha\)", fontsize=font_size_heatmap, labelpad=20)
                    plt.xticks(fontsize=font_size_heatmap)
                    plt.yticks(fontsize=font_size_heatmap)
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=font_size_heatmap)
                    labels = cbar.ax.get_yticklabels()
                    for i, label in enumerate(labels):
                        label.set_visible(i == 0 or i % 2 == 1)
                    # labels = ax.get_yticklabels()
                    # hide every other label
                    # for i, label in enumerate(labels):
                    #     if i % y_nr_ticks_layer != 0:
                    #         label.set_visible(False)
                    ax.tick_params(axis='y', labelsize=font_size_heatmap)
                    plt.yticks(rotation=0)
                    plt.xticks(rotation=45)
                    # ax.set_xscale("log")
                    # ax.set_yscale("log")
                    ax.tick_params(axis='both',
                                        which='major',
                                        length=10,     # length of tick marks
                                        width=2,      # thickness
                                        color='black')  # tick color
                    # plt.title("Heatmap")
                    # plt.show()
                    plt.savefig(f"./../../latex/rapport/figures/{material}_{defect_type}_{target_variable}_optim_kernel{kernel_specification}.pdf", dpi=300, bbox_inches="tight")
                    # plt.show()
                    plt.close("all")

                    # Find the index of the minimum y
                    min_gamma_meas = kernel_optim_data["scores"][str(optim_gamma)]
                    min_index = min_gamma_meas.index(min(min_gamma_meas))
                    x_min = kernel_optim_data["alphas"][min_index]
                    y_min = min_gamma_meas[min_index]

                    # Highlight the minimum point
                    plt.scatter(x_min, y_min, color='#DC143C', marker=styles[highlight_gamma.index("#DC143C")]["marker"], label=fr"Best \(\alpha\) = {optimal_alpha}", zorder=5, s=150)
                    highlight_gamma.append("#DC143C")
                    plt.annotate(f"{y_min:.3f}", (x_min, y_min), textcoords="offset points", xytext=(0, -20), ha='center', color="#DC143C")

                    plt.ylim(y_lim_lower - ylim_addon, gamma_0_mean + ylim_addon)
                    plt.xlabel(fr"Value of \(\alpha\)")
                    plt.xscale("log")
                    plt.grid(True)
                    plt.ylabel(f"Mean Absolute Error Averaged Across 10-Fold Cross-Validation")
                    plt.legend(labelcolor=highlight_gamma, fontsize=12, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))

                    if titles_bool:
                        if kernel_specification == "_unscaled":
                            plt.title(f"{material_title_dict[material]} {defect_type} {target_variable} Kernel Unscaled")
                        elif kernel_specification == "_unscaled_finer":
                            plt.title(f"{material_title_dict[material]} {defect_type} {target_variable} Kernel Unscaled Finer")
                        elif kernel_specification == "_normal":
                            plt.title(f"{material_title_dict[material]} {defect_type} {target_variable} Kernel Normal")
                        else:
                            plt.title(f"{material_title_dict[material]} {defect_type} {target_variable} Kernel Standard")

                    save_path = f"{path_dir}/{material}_{defect_type}_{target_variable}_optim_kernel{kernel_specification}.pdf"
                    # plt.tight_layout()
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")

                    plt.close('all')
                except KeyError as error:
                    print(error)
                    plt.close('all')

            continue

            styles = [
                {"color": "darkorange", "marker": "p", "linestyle": "-"},
                {"color": "darkgreen", "marker": "*", "linestyle": ":"},
                {"color": "darkgoldenrod", "marker": "h", "linestyle": ":"},
                {"color": "green", "marker": "x", "linestyle": "-"},
                {"color": "blue", "marker": "+", "linestyle": ":"},
                {"color": "navy", "marker": ".", "linestyle": "-"},
                {"color": "purple", "marker": "X", "linestyle": ":"},



                {"color": "b", "marker": "o", "linestyle": "-"},
                {"color": "g", "marker": "s", "linestyle": ":"},
                {"color": "purple", "marker": "D", "linestyle": ":"},
                {"color": "c", "marker": "^", "linestyle": "-"},
                {"color": "m", "marker": "v", "linestyle": ":"},
                {"color": "y", "marker": "<", "linestyle": "-"},
                {"color": "k", "marker": ">", "linestyle": ":"},
            ]

            ## compare graph
            try:

                # Linear
                plt.plot([], [], linestyle="none", label=" ")
                plt.plot([], [], linestyle="none", label="Linear")

                base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["MAE_percent_list_linear_test"],
                    marker="o",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"Test ({plot_data['MAE_percent_list_linear_test'][-1]:.3f})",
                    color=base_color
                )

                r, g, b = mcolors.to_rgb(base_color)
                lighter_color = (r + (1 - r) * 0.5,
                                 g + (1 - g) * 0.5,
                                 b + (1 - b) * 0.5)
                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["MAE_percent_list_linear_train"],
                    marker=".",
                    linestyle="--",
                    linewidth=1.5,
                    color=lighter_color,
                    label=f"Train ({plot_data['MAE_percent_list_linear_train'][-1]:.3f})",
                )

                base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]

                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["mean_target_variable_test"],
                    marker="o",
                    linestyle=":",
                    linewidth=1.5,
                    label="Mean Test",
                    markersize=11,
                    color=base_color,
                )

                r, g, b = mcolors.to_rgb(base_color)
                lighter_color = (r + (1 - r) * 0.5,
                                 g + (1 - g) * 0.5,
                                 b + (1 - b) * 0.5)

                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["mean_target_variable_train"],
                    marker="D",
                    markersize=7,
                    linestyle=":",
                    linewidth=1.5,
                    label="Mean Train",
                    color=lighter_color
                )



# styles = [
#     {"color": "b", "marker": "o", "linestyle": "-"},
#     {"color": "g", "marker": "s", "linestyle": "-"},
#     {"color": "purple", "marker": "D", "linestyle": "-"},
#     {"color": "c", "marker": "^", "linestyle": "-"},
#     {"color": "m", "marker": "v", "linestyle": "-"},
#     {"color": "y", "marker": "<", "linestyle": "-"},
#     {"color": "k", "marker": ">", "linestyle": "-"},
#     {"color": "saddlebrown", "marker": "p", "linestyle": "-"},
#     {"color": "brown", "marker": "*", "linestyle": "-"},
#     {"color": "gray", "marker": "h", "linestyle": "-"},
#     {"color": "lime", "marker": "x", "linestyle": "-"},
#     {"color": "navy", "marker": "+", "linestyle": "-"},
#     {"color": "darkred", "marker": ".", "linestyle": "-"},
#     {"color": "teal", "marker": "X", "linestyle": "-"},
# ]

                # Kernel
                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["mean_target_variable"],
                    marker="s",
                    markersize=12,
                    linestyle=":",
                    label=f"Mean All",
                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][8]
                )
                plt.plot([], [], linestyle="none", label="Kernel")
                base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["MAE_percent_list_kernel_test"],
                    marker="o",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"Test ({plot_data['MAE_percent_list_kernel_test'][-1]:.3f})",
                    color=base_color
                )

                r, g, b = mcolors.to_rgb(base_color)
                lighter_color = (r + (1 - r) * 0.5,
                                 g + (1 - g) * 0.5,
                                 b + (1 - b) * 0.5)

                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["MAE_percent_list_kernel_train"],
                    marker=".",
                    linestyle="--",
                    linewidth=1.5,
                    color=lighter_color,
                    label=f"Train ({plot_data['MAE_percent_list_kernel_train'][-1]:.3f})",
                )

                base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]

                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["mean_target_variable_test"],
                    marker="o",
                    linestyle=":",
                    linewidth=1.5,
                    markersize=11,
                    label="Mean Test",
                    color=base_color,
                )

                r, g, b = mcolors.to_rgb(base_color)
                lighter_color = (r + (1 - r) * 0.5,
                                 g + (1 - g) * 0.5,
                                 b + (1 - b) * 0.5)

                plt.plot(
                    plot_data["nr_data_points"],
                    plot_data["mean_target_variable_train"],
                    marker="D",
                    markersize=7,
                    linestyle=":",
                    linewidth=1.5,
                    label="Mean Train",
                    color=lighter_color
                )

                # Neural
                try:
                    assert plot_data["MAE_percent_list_neural_test"]
                    plt.plot([], [], linestyle="none", label=" ")
                    plt.plot([], [], linestyle="none", label="Neural")
                    base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]

                    plt.plot(
                        plot_data["nr_data_points"],
                        plot_data["MAE_percent_list_neural_test"],
                        marker="o",
                        linestyle="-",
                        linewidth=1.5,
                        label=f"Test ({plot_data['MAE_percent_list_neural_test'][-1]:.3f})",
                        color=base_color
                    )

                    r, g, b = mcolors.to_rgb(base_color)
                    lighter_color = (r + (1 - r) * 0.5,
                                     g + (1 - g) * 0.5,
                                     b + (1 - b) * 0.5)

                    plt.plot(
                        plot_data["nr_data_points"],
                        plot_data["MAE_percent_list_neural_train"],
                        marker=".",
                        linestyle="--",
                        linewidth=1.5,
                        color=lighter_color,
                        label=f"Train ({plot_data['MAE_percent_list_neural_train'][-1]:.3f})",
                    )

                    base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][4]

                    plt.plot(
                        plot_data["nr_data_points"],
                        plot_data["mean_target_variable_neural_test"],
                        marker="o",
                        linestyle=":",
                        linewidth=1.5,
                        markersize=6,
                        color=base_color,
                        label="Mean Test",
                    )

                    r, g, b = mcolors.to_rgb(base_color)
                    lighter_color = (r + (1 - r) * 0.5,
                                     g + (1 - g) * 0.5,
                                     b + (1 - b) * 0.5)

                    plt.plot(
                        plot_data["nr_data_points"],
                        plot_data["mean_target_variable_neural_train"],
                        marker=".",
                        linestyle=":",
                        linewidth=1.5,
                        markersize=5,
                        label="Mean Train",
                        color=lighter_color
                    )

                    plt.legend(
                        ncol=3,
                        fontsize=12,
                        # columnspacing=1.5,
                        # handletextpad=0.6,
                    )
                except KeyError:
                    plt.legend(
                        ncol=2,
                        fontsize=12,
                        # columnspacing=1.5,
                        # handletextpad=0.6,
                    )






                plt.xlabel("Number of Data Points on a Log10 Scale")
                plt.xscale("log")
                plt.grid(True)
                plt.ylabel(f"MAEA of 10-Fold Cross-Validation [eV]")
                # plt.legend(fontsize=12)

                # if defect_type == 'regular':
                nr_data_legend = f'N: {plot_data['N']}\nn: {plot_data['n']}'
                try:
                    # title = f"N: {plot_data['N']}\nn: {plot_data['n']}\nBest final: {min([plot_data["MAE_percent_list_linear_test"][-1], plot_data["MAE_percent_list_kernel_test"][-1], plot_data["MAE_percent_list_neural_test"][-1]]):.3f} eV"
                    minimum_mae = min([plot_data["MAE_percent_list_linear_test"][-1], plot_data["MAE_percent_list_kernel_test"][-1], plot_data["MAE_percent_list_neural_test"][-1]])
                except KeyError:
                    # title = f"N: {plot_data['N']}\nn: {plot_data['n']}\nBest final: {min([plot_data["MAE_percent_list_linear_test"][-1], plot_data["MAE_percent_list_kernel_test"][-1]]):.3f} eV"
                    minimum_mae = min([plot_data["MAE_percent_list_linear_test"][-1], plot_data["MAE_percent_list_kernel_test"][-1]])
                # else:
                #     nr_data_legend = f'N: {plot_data['N']}'
                #     try:
                #         minimum_mae = min([plot_data["MAE_percent_list_linear_test"][-1], plot_data["MAE_percent_list_kernel_test"][-1], plot_data["MAE_percent_list_neural_test"][-1]])
                #         # title = f"N: {plot_data['N']}\nBest final: {min([plot_data["MAE_percent_list_linear_test"][-1], plot_data["MAE_percent_list_kernel_test"][-1], plot_data["MAE_percent_list_neural_test"][-1]]):.3f} eV"
                #     except KeyError:
                #         minimum_mae = min([plot_data["MAE_percent_list_linear_test"][-1], plot_data["MAE_percent_list_kernel_test"][-1]])
                        # title = f"N: {plot_data['N']}\nBest final: {min([plot_data["MAE_percent_list_linear_test"][-1], plot_data["MAE_percent_list_kernel_test"][-1]]):.3f} eV"
                # plt.text(0.02, 0.03, title, transform=plt.gca().transAxes, fontsize=12, fontfamily='serif', verticalalignment='bottom', horizontalalignment='left', color='black', bbox=dict(boxstyle='round,pad=0.3,rounding_size=0.3', facecolor='white', edgecolor='lightgrey', linewidth=0.8))
                try:
                    place_text_box_avoiding_legend_overlap(8 * plot_data["nr_data_points"], plot_data["MAE_percent_list_linear_test"] + plot_data["MAE_percent_list_kernel_test"] + plot_data["MAE_percent_list_neural_test"] + plot_data["mean_target_variable"] + plot_data["mean_target_variable_train"] + plot_data["mean_target_variable_test"] + plot_data["mean_target_variable_neural_test"] + plot_data["mean_target_variable_neural_test"], nr_data_legend, 12)
                except KeyError:
                    place_text_box_avoiding_legend_overlap(5 * plot_data["nr_data_points"], plot_data["MAE_percent_list_linear_test"] + plot_data["MAE_percent_list_kernel_test"] + plot_data["mean_target_variable"] + plot_data["mean_target_variable_train"] + plot_data["mean_target_variable_test"], nr_data_legend, 12)

                # plt.scatter(plot_data["nr_data_points"][-1], minimum_mae, color='#DC143C', zorder=5)

                plt.tight_layout()
                plt.savefig(f"{path_dir}/{material}_{defect_type}_{target_variable}_compare_report.pdf", dpi=300, bbox_inches="tight")
                plt.close('all')
                print("success")
            except KeyError as error:
                print(error)
                plt.close('all')































# styles = [
#     {"color": "b", "marker": "o", "linestyle": "-"},
#     {"color": "g", "marker": "s", "linestyle": "-"},
#     {"color": "purple", "marker": "D", "linestyle": "-"},
#     {"color": "c", "marker": "^", "linestyle": "-"},
#     {"color": "m", "marker": "v", "linestyle": "-"},
#     {"color": "y", "marker": "<", "linestyle": "-"},
#     {"color": "k", "marker": ">", "linestyle": "-"},
#     {"color": "saddlebrown", "marker": "p", "linestyle": "-"},
#     {"color": "brown", "marker": "*", "linestyle": "-"},
#     {"color": "gray", "marker": "h", "linestyle": "-"},
#     {"color": "lime", "marker": "x", "linestyle": "-"},
#     {"color": "navy", "marker": "+", "linestyle": "-"},
#     {"color": "darkred", "marker": ".", "linestyle": "-"},
#     {"color": "teal", "marker": "X", "linestyle": "-"},
# ]




def get_order_indices(items):
    items_lower = [x.lower() for x in items]

    models = ['linear', 'kernel', 'neural']
    data = {m: {} for m in models}

    # map items to their original indices
    for i, x in enumerate(items_lower):
        if x in models:
            continue
        if x.startswith('test_'):
            model = x.replace('test_', '')
            data[model]['test'] = i
        elif x.startswith('train_'):
            model = x.replace('train_', '')
            data[model]['train'] = i

    order = []

    # (optional) model names themselves usually ignored for reordering raw lists
    for m in models:
        order.append(data[m]['test'])

    for m in models:
        order.append(data[m]['train'])

    return order

def apply_order(items, order):
    return [items[i] for i in order]



linear_targets = {
    ('vacancy-substitutional', 'FE'): {'scaling': 'normal', 'descriptor': 'sic_wyckoff_letter'},
    ('vacancy-substitutional', 'ZPL'): {'scaling': 'standard', 'descriptor': 'sic_wyckoff_letter'},
    ('interstitial', 'FE'): {'scaling': 'standard', 'descriptor': 'sic_wyckoff'},
    ('interstitial', 'ZPL'): {'scaling': 'normal', 'descriptor': 'sic_wyckoff_nw_data'},
}
kernel_targets = {
    ('vacancy-substitutional', 'FE'): {'scaling': "unscaled", 'descriptor': "sic_no_wyckoff"},
    ('vacancy-substitutional', 'ZPL'): {'scaling': "unscaled", 'descriptor': "sic_no_wyckoff"},
    ('interstitial', 'FE'): {'scaling': "unscaled", 'descriptor': "sic_wyckoff"},
    ('interstitial', 'ZPL'): {'scaling': "unscaled", 'descriptor': "sic_no_wyckoff"}
}
neural_targets = {
    ('vacancy-substitutional', 'FE'): {'scaling': 'standard', 'descriptor': 'sic_no_wyckoff'},
    ('vacancy-substitutional', 'ZPL'): {'scaling': 'standard', 'descriptor': 'sic_no_wyckoff'},
    ('interstitial', 'FE'): {'scaling': 'standard', 'descriptor': 'sic_wyckoff'},
    ('interstitial', 'ZPL'): {'scaling': 'standard', 'descriptor': 'sic_no_wyckoff'}
}

# fig, axs = plt.subplots(4, 1)
# fig, axs = plt.subplots(4, 1, figsize=(9, 14))
# axs[1].sharex(axs[0])
# axs[3].sharex(axs[2])

labelsize = 30
lightning = 0.4

for train_line in ["", "_train"]:
    continue
    for cross_validation in [""]:
    # for cross_validation in ["", "_10_fold"]:
    # for cross_validation in [""]:
        n = 0
        for defect_type in defect_types:
            for target_variable in target_variables:
                n = n + 1

                if not cross_validation:
                    d_max = 10
                else:
                    d_max = 11

                linear_descriptor = linear_targets[(defect_type,target_variable)]["descriptor"]
                path_dir = f"../../results/{linear_descriptor}/{defect_type}/{target_variable}"
                linear_plot_data = toml.load(os.path.join(path_dir, f"results{cross_validation}.toml"))
                linear_maes = linear_plot_data[f"MAE_percent_list_linear_test{cross_validation}"]
                linear_maes_train = linear_plot_data[f"MAE_percent_list_linear_train{cross_validation}"]
                linear_nr_data = linear_plot_data[f"nr_data_points{cross_validation}"]

                kernel_descriptor = kernel_targets[(defect_type,target_variable)]["descriptor"]
                path_dir = f"../../results/{kernel_descriptor}/{defect_type}/{target_variable}"
                kernel_plot_data = toml.load(os.path.join(path_dir, f"results{cross_validation}.toml"))
                kernel_nr_data = kernel_plot_data[f"nr_data_points{cross_validation}"]
                kernel_maes = [0.0] * (d_max-1)
                for d in range(1,d_max):
                    try:
                        kernel_maes[d-1] = np.mean(kernel_plot_data[f"MAE_percent_list_kernel_test_{d}0{cross_validation}"])
                    except KeyError:
                        pass
                kernel_maes_train = [0.0] * (d_max-1)
                for d in range(1,d_max):
                    try:
                        kernel_maes_train[d-1] = np.mean(kernel_plot_data[f"MAE_percent_list_kernel_train_{d}0{cross_validation}"])
                    except KeyError:
                        pass

                neural_descriptor = neural_targets[(defect_type,target_variable)]["descriptor"]
                path_dir = f"../../results/{neural_descriptor}/{defect_type}/{target_variable}"
                neural_plot_data = toml.load(os.path.join(path_dir, f"results{cross_validation}.toml"))
                neural_nr_data = neural_plot_data[f"nr_data_points{cross_validation}"]
                neural_maes = [0.0] * (d_max-1)
                for d in range(1,d_max):
                    try:
                        neural_maes[d-1] = np.mean(neural_plot_data[f"MAE_percent_list_neural_test_{d}0{cross_validation}"])
                    except KeyError:
                        pass
                neural_maes_train = [0.0] * (d_max-1)
                for d in range(1,d_max):
                    try:
                        neural_maes_train[d-1] = np.mean(neural_plot_data[f"MAE_percent_list_neural_train_{d}0{cross_validation}"])
                    except KeyError:
                        pass


                try:

                    # r, g, b = mcolors.to_rgb(base_color)
                    # lighter_color = (r + (1 - r) * 0.5,
                    #                     g + (1 - g) * 0.5,
                    #                     b + (1 - b) * 0.5)


                    linewidth = 2.5
                    markersize = 10
                    # if n == 1:
                    fig, ax = plt.subplots(figsize=(9, 3.5))
                    # fig.subplots_adjust(left=0.1)

                    ax.set_xscale("log")
                    ax.grid(True)

                    # fig, axs = plt.subplots(4, 1, figsize=(6, 8))
                    if train_line:
                        # plt.plot([], [], linestyle="none", label="Ridge")
                        label = "R Test"
                    else:
                        label = "Ridge"
                    base_color_ridge = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
                    line_linear = ax.plot(
                        linear_nr_data,
                        linear_maes,
                        marker="o",
                        linestyle="-",
                        linewidth=linewidth,
                        markersize=markersize,
                        # linewidth=1.5,
                        color=base_color_ridge,
                        label=label,
                    )
                    if train_line:
                        r, g, b = mcolors.to_rgb(base_color_ridge)
                        lighter_color_ridge = (r + (1 - r) * lightning,
                                         g + (1 - g) * lightning,
                                         b + (1 - b) * lightning)
                        line_linear = ax.plot(
                            linear_nr_data,
                            linear_maes_train,
                            marker="o",
                            linestyle="--",
                            linewidth=linewidth*0.85,
                            markersize=markersize*0.85,
                            markerfacecolor='none',
                            # linewidth=1.5,
                            color=lighter_color_ridge,
                            label="R Train",
                        )
                    if train_line:
                        # plt.plot([], [], linestyle="none", label="Kernel Ridge")
                        label = "KR Test"
                    else:
                        label = "Kernel Ridge"
                    base_color_kernel = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
                    line_kernel = ax.plot(
                        kernel_nr_data,
                        kernel_maes,
                        marker="s",
                        linestyle="-",
                        linewidth=linewidth,
                        markersize=markersize,
                        # linewidth=1.5,
                        color=base_color_kernel,
                        label=label,
                    )
                    if train_line:
                        r, g, b = mcolors.to_rgb(base_color_kernel)
                        lighter_color_kernel = (r + (1 - r) * lightning,
                                         g + (1 - g) * lightning,
                                         b + (1 - b) * lightning)
                        base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
                        line_kernel = ax.plot(
                            kernel_nr_data,
                            kernel_maes_train,
                            marker="s",
                            linestyle="--",
                            linewidth=linewidth*0.85,
                            markersize=markersize*0.85,
                            markerfacecolor='none',
                            # linewidth=1.5,
                            color=lighter_color_kernel,
                            label="KR Train",
                        )
                        # plt.plot([], [], linestyle="none", label="MLP")
                        label = "MLP Test"
                    else:
                        label = "Multilayer Perceptron"
                    base_color_neural = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
                    line_neural = ax.plot(
                        neural_nr_data,
                        neural_maes,
                        marker="^",
                        linestyle="-",
                        # linewidth=1.5,
                        linewidth=linewidth,
                        markersize=markersize,
                        color=base_color_neural,
                        label=label,
                    )

                    if train_line:
                        r, g, b = mcolors.to_rgb(base_color_neural)
                        lighter_color_neural = (r + (1 - r) * lightning,
                                         g + (1 - g) * lightning,
                                         b + (1 - b) * lightning)
                        line_neural = ax.plot(
                            neural_nr_data,
                            neural_maes_train,
                            marker="^",
                            linestyle="--",
                            # linewidth=1.5,
                            linewidth=linewidth*0.85,
                            markersize=markersize*0.85,
                            markerfacecolor='none',
                            color=lighter_color_neural,
                            label="MLP Train",
                        )

                    ax.tick_params(axis='both', labelsize=labelsize, length=10, width=3.5)
                    ax.tick_params(which="minor", length=6, width=3)
                    fig.tight_layout()
                    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

                    fig.savefig(f"../../results/mae_compare_article_{n}{cross_validation}{train_line}.pdf", dpi=300, bbox_inches="tight")
                    fig.savefig(f"../../latex/rapport/figures/mae_compare_article_{n}{cross_validation}{train_line}.pdf", dpi=300, bbox_inches="tight")

                    if n == 1 and train_line:
                        handles, labels = ax.get_legend_handles_labels()

                        fig_legend = plt.figure(figsize=(6, 1))
                        ax_legend = fig_legend.add_axes([0, 0, 1, 1])
                        ax_legend.axis("off")

                        legend = ax_legend.legend(
                            handles,
                            labels,
                            # title="Model / Split",
                            loc="center",
                            bbox_to_anchor=(0.5, 0.5),
                            ncol=3,
                            frameon=False,

                            # bigger text
                            fontsize=14,
                            # title_fontsize=18,

                            # bigger markers
                            markerscale=0.7,

                            # compact entry padding
                            handlelength=1.0,
                            handletextpad=0.25,
                            borderpad=0.0,
                            borderaxespad=0.0,

                            # spacing between entries
                            columnspacing=1.6,
                            labelspacing=0.0,

                            alignment="center",
                        )

                        # style legend handles
                        for h in legend.legend_handles:

                            # thicker line entries
                            if hasattr(h, "set_linewidth"):
                                h.set_linewidth(2)

                            # clean scatter entries
                            if hasattr(h, "set_linewidths"):
                                h.set_linewidths(0)

                            if hasattr(h, "set_edgecolors"):
                                h.set_edgecolors("none")

                        fig_legend.canvas.draw()

                        # tightly crop to legend
                        bbox = legend.get_window_extent().transformed(
                            fig_legend.dpi_scale_trans.inverted()
                        )

                        pad = 0.05
                        fig_legend.set_size_inches(
                            bbox.width,
                            # bbox.width + pad,
                            bbox.height,
                            # bbox.height + pad,
                        )

                        plt.savefig(f"../../latex/rapport/figures/mae_legend{train_line}.pdf", bbox_inches="tight", dpi=300,pad_inches=0)

                    if n == 1 and not train_line:

                        handles, labels = ax.get_legend_handles_labels()

                        fig_legend = plt.figure(figsize=(6, 1))
                        ax_legend = fig_legend.add_axes([0, 0, 1, 1])
                        ax_legend.axis("off")

                        legend = ax_legend.legend(
                            handles,
                            labels,
                            loc="center",
                            bbox_to_anchor=(0.5, 0.5),
                            ncol=3,
                            frameon=False,

                            # bigger text
                            fontsize=24,

                            # bigger markers
                            markerscale=1.4,

                            # compact entry padding
                            handlelength=1.0,
                            handletextpad=0.15,
                            borderpad=0.0,
                            borderaxespad=0.0,

                            # spacing between entries
                            columnspacing=0.4,
                            labelspacing=0.0,

                            alignment="center",
                        )

                        # style legend handles
                        for h in legend.legend_handles:

                            # thicker line entries
                            if hasattr(h, "set_linewidth"):
                                h.set_linewidth(3)

                            # clean scatter entries
                            if hasattr(h, "set_linewidths"):
                                h.set_linewidths(0)

                            if hasattr(h, "set_edgecolors"):
                                h.set_edgecolors("none")

                        fig_legend.canvas.draw()

                        # tightly crop to legend
                        bbox = legend.get_window_extent().transformed(
                            fig_legend.dpi_scale_trans.inverted()
                        )

                        pad = 0.05
                        fig_legend.set_size_inches(
                            bbox.width,
                            # bbox.width + pad,
                            bbox.height,
                            # bbox.height + pad,
                        )

                        fig_legend.savefig(f"../../latex/rapport/figures/mae_legend{train_line}.pdf", bbox_inches="tight", dpi=300,pad_inches=0)

                        # if train_line:
                        #     plt.close('all')


                        #     labels = ["Linear", "Kernel", "Neural"]

                        #     fig, ax = plt.subplots(figsize=(6, 1))
                        #     ax.axis("off")

                        #     # automatic evenly spaced positions with built-in padding
                        #     x_positions = np.linspace(0.1, 0.9, len(labels))

                        #     for x, label in zip(x_positions, labels):
                        #         ax.text(x, 0.5, label, ha="center", va="center", fontsize=18)

                        #     ax.set_xlim(0, 1)
                        #     ax.set_ylim(0, 1)
                        #     ax.margins(0)

                        #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)



                        #     plt.show()

                    plt.close('all')

                except KeyError as error:
                    # print(error)
                    plt.close('all')

best_targets = {
    ('vacancy-substitutional', 'FE'): {'model': 'neural', 'scaling': 'standard', 'descriptor': 'sic_no_wyckoff'},
    ('interstitial', 'FE'): {'model': 'neural', 'scaling': 'standard', 'descriptor': 'sic_wyckoff'},
    ('vacancy-substitutional', 'ZPL'): {'model': 'kernel', 'scaling': 'unscaled', 'descriptor': 'sic_no_wyckoff'},
    ('interstitial', 'ZPL'): {'model': 'kernel', 'scaling': 'unscaled', 'descriptor': 'sic_no_wyckoff'}
}

plt.close("all")
n = 0
labelsize = 120
for defect_type in defect_types:
    continue
    for target_variable in target_variables:
        n = n + 1
        best = best_targets[(defect_type, target_variable)]
        toml_path = f"../../results/{best["descriptor"]}/{defect_type}/{target_variable}/results.toml"
        toml_data = toml.load(toml_path)
        toml_scatter_path = f"../../results/{best["descriptor"]}/{defect_type}/{target_variable}/results_scatter.toml"
        toml_scatter_data = toml.load(toml_scatter_path)

        try:
            fig, ax = plt.subplots(figsize=(24, 24))

            linewidth = 14
            ax.scatter(toml_scatter_data[f"scatter_true"], toml_scatter_data[f"scatter_{best["model"]}_pred"], marker="o", label="Predictions", alpha=0.2, s=300)
            base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
            straight_line = [min(toml_scatter_data["scatter_true"]), max(toml_scatter_data[f"scatter_{best["model"]}_pred"])]
            ax.plot(straight_line, straight_line, linewidth=linewidth, alpha=0.3, label="Perfect Predictions", color=base_color)
            base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
            ax.plot(
                [min(toml_scatter_data["scatter_true"]), max(toml_scatter_data["scatter_true"])],
                2 * [float(np.mean(toml_data["mean_target_variable"]))],
                alpha=0.4,
                linewidth=linewidth,
                label="Average Energy",
                color=base_color,
            )

            # plt.xlabel(f"True Energy")
            # plt.ylabel(f"Predicted Energy")
            # if n == 1:
                # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
            if n == 2:
                # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
                pass
            if n == 3:
                # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
                pass
            if n == 4:
                # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
                plt.ylim(-0.7, 1.3)
            # ax.set_box_aspect(1)
            plt.grid(True)


            plt.tick_params(axis='both', labelsize=labelsize, length=32, width=12)
            # plt.xticks()
            # plt.yticks()

            # if n == 1:
                # plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center')
            # ax.set_box_aspect(1)
            # fig.tight_layout()
            plt.subplots_adjust(right=.9, left=0.1, bottom=0.9)

            plt.savefig(f"../../results/scatter_{n}.pdf", dpi=300)
            plt.savefig(f"../../latex/rapport/figures/scatter_{n}.pdf", dpi=300)

            if n == 1:

                handles, labels = ax.get_legend_handles_labels()

                fig_legend = plt.figure(figsize=(6, 1))
                ax_legend = fig_legend.add_axes([0, 0, 1, 1])
                ax_legend.axis("off")

                legend = ax_legend.legend(
                    handles,
                    labels,
                    loc="center",
                    bbox_to_anchor=(0.5, 0.5),
                    ncol=3,
                    frameon=False,
                    fontsize=18,
                    handlelength=1.4,
                    handletextpad=0.5,
                    columnspacing=1.0,
                    borderaxespad=0.0,
                    markerscale=1.2,
                    alignment="center",
                )

                # thicker lines
                for h in legend.legend_handles:
                    if hasattr(h, "set_linewidth"):
                        h.set_linewidth(6)

                    if hasattr(h, "set_linewidths"):
                        h.set_linewidths(0)

                fig_legend.canvas.draw()

                # tightly crop figure to legend
                bbox = legend.get_window_extent().transformed(
                    fig_legend.dpi_scale_trans.inverted()
                )

                pad = 0.05
                fig_legend.set_size_inches(
                    bbox.width + pad,
                    bbox.height + pad,
                )

                fig_legend.savefig(f"../../latex/rapport/figures/scatter_legend.pdf", bbox_inches="tight", dpi=300,pad_inches=0)

            plt.close("all")

        except KeyError as error:
            print(n)
            print(error)
            plt.close("all")
