"""Paper Fig. 1: minimum, maximum, mean and standard deviation of every input parameter.
Output: results/figures/*_input_parameter_details_*.pdf.
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from common import FIGURES_DIR, ensure_dir, load_datasets
from input_parameters import FEATURES
import sys
from matplotlib.ticker import MaxNLocator
import numpy as np

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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{newtxtext}
        \usepackage{newtxmath}
    """,
    "font.size": 14,
})

ensure_dir(FIGURES_DIR)

fontsize=60
width = 8
scalings = [
    "unscaled",
    # "standard",
    # "normal"
]
first_bool = True
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for scaling in scalings:
    for descriptor in descriptors:

        features = FEATURES[descriptor]

        for defect_type in defect_types:

            data, data_single = load_datasets(descriptor, defect_type)
            data = data[features]
            data_single = data_single[features]

            if scaling == "unscaled":
                df = data
                df = df.dropna()
                df = df.loc[:, df.nunique() > 1]

                summary = pd.DataFrame({
                    "mean": df.mean(),
                    "max": df.max(),
                    "min": df.min(),
                    "std": df.std(ddof=1)
                })
                pdf_name = f"{FIGURES_DIR}/{descriptor}_{defect_type}_input_parameter_details_unscaled_double.pdf"

                df = data_single
                df = df.dropna()
                df = df.loc[:, df.nunique() > 1]
                mean_values_single = df.mean()
                summary_single = pd.DataFrame({
                    "mean": df.mean(),
                    "max": df.max(),
                    "min": df.min(),
                    "std": df.std(ddof=1)
                })
                pdf_name_single = f"{FIGURES_DIR}/{descriptor}_{defect_type}_input_parameter_details_unscaled_single.pdf"

            x_pos = np.arange(len(summary))
            plt.bar(
                x_pos,
                summary["max"],
                color=colors[0]
            )
            plt.bar(
                x_pos,
                summary["mean"],
                color=colors[1]
            )
            plt.bar(
                x_pos,
                summary["min"],
                color=colors[2]
            )
            plt.bar(
                x_pos,
                summary["std"],
                color=colors[3],
                label="Standard Deviation",
                width=0.3
            )
            plt.xticks([])
            plt.tick_params(axis='y', labelsize=fontsize, length=14, width=width)
            ax = plt.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            plt.tight_layout()
            plt.savefig(pdf_name, dpi=300, bbox_inches='tight')
            plt.close()

            x_pos = np.arange(len(mean_values_single))
            plt.bar(
                x_pos,
                summary_single["max"],
                color=colors[0],
                label="Max."
            )
            plt.bar(
                x_pos,
                summary_single["mean"],
                color=colors[1],
                label="Average"
            )
            plt.bar(
                x_pos,
                summary_single["min"],
                color=colors[2],
                label="Min."
            )
            plt.bar(
                x_pos,
                summary_single["std"],
                color=colors[3],
                label="Standard Deviation",
                width=0.3
            )
            plt.xticks([])
            ax = plt.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            plt.tick_params(axis='y', labelsize=fontsize, length=14, width=width)
            plt.tight_layout()
            plt.savefig(pdf_name_single, dpi=300, bbox_inches='tight')

            if first_bool:
                first_bool = False




                handles, labels = ax.get_legend_handles_labels()

                # move 3rd item to front
                handles.insert(0, handles.pop(2))
                labels.insert(0, labels.pop(2))

                fig_legend = plt.figure(figsize=(8, 1))
                ax_legend = fig_legend.add_axes([0, 0, 1, 1])
                ax_legend.axis("off")

                legend = ax_legend.legend(
                    handles,
                    labels,
                    loc="center",
                    bbox_to_anchor=(0.5, 0.5),
                    ncol=4,
                    frameon=False,
                    fontsize=18,
                    handlelength=1.4,
                    handletextpad=0.5,
                    columnspacing=1.0,
                    labelspacing=0.2,
                    borderaxespad=0.0,
                    borderpad=0.1,
                    markerscale=1.2,
                    alignment="center",
                )

                # style legend handles
                for h in legend.legend_handles:

                    # thicker lines
                    if hasattr(h, "set_linewidth"):
                        h.set_linewidth(6)

                    # clean scatter markers
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
                    bbox.width + pad,
                    bbox.height + pad,
                )





                fig_legend.savefig(f"{FIGURES_DIR}/input_parameter_details_legend.pdf", bbox_inches="tight", dpi=300,pad_inches=0)

            plt.close()
