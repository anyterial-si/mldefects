import os
import glob
# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

python_folder = os.path.join(os.getcwd(), "..")
database_folder = os.path.join(python_folder, "database")
sys.path.append(database_folder)

threshold = 0.31  # the cutoff

low_cmap = plt.cm.coolwarm

# low_cmap = plt.cm.coolwarm_r
# low_cmap = plt.cm.RdBu
# low_cmap = plt.cm.RdBu_r
# low_cmap = plt.cm.seismic
# low_cmap = plt.cm.seismic_r
# low_cmap = plt.cm.bwr
# low_cmap = plt.cm.bwr_r
# low_cmap = plt.cm.PiYG
# low_cmap = plt.cm.PiYG_r
# low_cmap = plt.cm.PRGn
# low_cmap = plt.cm.PRGn_r
# low_cmap = plt.cm.BrBG
# low_cmap = plt.cm.BrBG_r
# low_cmap = plt.cm.PuOr
# low_cmap = plt.cm.PuOr_r
# low_cmap = plt.cm.viridis
# low_cmap = plt.cm.inferno
# low_cmap = plt.cm.magma
# low_cmap = plt.cm.plasma
# low_cmap = plt.cm.cividis
# low_cmap = plt.cm.jet
# low_cmap = plt.cm.cool
# low_cmap = plt.cm.winter

# Create custom colormap
# We'll take 256 colors from low_cmap and add one solid color for values above threshold
n_colors = 256
colors = low_cmap(np.linspace(0, 1, n_colors))
colors = np.vstack((colors[0:int(threshold*n_colors)], np.array([[1, 0, 0, 1]])))  # red above threshold

custom_cmap = ListedColormap(colors)

# Define boundaries so everything above threshold maps to the last color
bounds = np.linspace(0, threshold, int(threshold*n_colors))
bounds = np.append(bounds, 1.01)  # ensure everything above threshold is red
norm = BoundaryNorm(bounds, custom_cmap.N)

for stat_test in ["covariance_pearson_unscaled", "covariance_standard", "covariance_normal"]:
    for descriptor in descriptors:

        if descriptor == "sic_no_wyckoff":
            from input_parameters import atom_distance_charge_spin_input_parameters as features  # type: ignore
        elif descriptor == "sic_wyckoff":
            from input_parameters import sic_input_parameters as features  # type: ignore
        elif descriptor == "sic_wyckoff_letter":
            from input_parameters import sic_wyckoff_letter_input_parameters as features  # type: ignore
        elif descriptor == "sic_wyckoff_nw_data":
            from input_parameters import sic_input_parameters as features  # type: ignore
        elif descriptor == "sic_wyckoff_letter_nw_data":
            from input_parameters import sic_wyckoff_letter_input_parameters as features  # type: ignore
        material_folder = os.path.join(python_folder, "..", f"results/{descriptor}")

        for defect_type in defect_types:


            defect_folder = os.path.join(material_folder, defect_type)
            csv_files = glob.glob(os.path.join(defect_folder, "*.csv"))
            double_index = next(i for i, s in enumerate(csv_files) if s.endswith("double.csv"))
            data = pd.read_csv(csv_files[double_index])
            data_single = pd.read_csv(csv_files[double_index - 1])
            data = data[features]
            data_single = data_single[features]
            all_data = pd.concat([data, data_single], ignore_index=True)

            if stat_test == "covariance_pearson_unscaled":
                df = all_data.copy()
                df = df.dropna()
                df = df.loc[:, df.nunique() > 1]

                corr = df.cov(ddof=1)
                pdf_name = f"../../latex/rapport/figures/{descriptor}_{defect_type}_heatmap_covariance_unscaled.pdf"

                corr_pearson = df.corr("pearson")
                pdf_name_pearson = f"../../latex/rapport/figures/{descriptor}_{defect_type}_heatmap_pearson_coefficient.pdf"

            elif stat_test == "covariance_standard":
                scaler = StandardScaler()
                scaler.fit(all_data)
                all_data_scaled = scaler.transform(all_data)
                all_data_scaled = pd.DataFrame(all_data_scaled, columns=all_data.columns)

                df = all_data_scaled.copy()
                df = df.dropna()
                df = df.loc[:, df.nunique() > 1]
                corr = df.cov(ddof=1)
                pdf_name = f"../../latex/rapport/figures/{descriptor}_{defect_type}_heatmap_covariance_standardized.pdf"

            elif stat_test == "covariance_normal":
                scaler = MinMaxScaler()
                scaler.fit(all_data)
                all_data_scaled = scaler.transform(all_data)
                all_data_scaled = pd.DataFrame(all_data_scaled, columns=all_data.columns)

                df = all_data_scaled.copy()
                df = df.dropna()
                df = df.loc[:, df.nunique() > 1]
                corr = df.cov(ddof=1)
                pdf_name = f"../../latex/rapport/figures/{descriptor}_{defect_type}_heatmap_covariance_normalized.pdf"

            plt.imshow(corr, cmap=custom_cmap, norm=norm)
            # plt.pcolormesh(corr, cmap=custom_cmap, norm=norm, linewidths=0, edgecolors='face')
            plt.axis('off')
            # sns.heatmap(corr, cmap=custom_cmap, norm=norm, annot=False, cbar=False, linewidths=0, linecolor='none')
            # plt.xticks([])
            # plt.yticks([])
            plt.savefig(pdf_name, dpi=300, bbox_inches='tight')
            plt.close()

            if stat_test == "covariance_pearson_unscaled":
                plt.imshow(corr_pearson, cmap=custom_cmap, norm=norm)
                # plt.pcolormesh(corr_pearson, cmap=custom_cmap, norm=norm, linewidths=0, edgecolors='face')
                plt.axis('off')
                # sns.heatmap(corr_pearson, cmap=custom_cmap, norm=norm, annot=False, cbar=False, linewidths=0, linecolor='none')
                # plt.xticks([])
                # plt.yticks([])
                plt.savefig(pdf_name_pearson, dpi=300, bbox_inches='tight')
                plt.close()
