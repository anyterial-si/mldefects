import os
import glob
import pandas as pd
import sys

descriptors = [
    # "sic_no_wyckoff",
    # "sic_wyckoff",
    # "sic_wyckoff_letter",
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

python_folder = os.path.join(os.getcwd(), "..")
database_folder = os.path.join(python_folder, "database")
sys.path.append(database_folder)

for target_variable in target_variables:
    for descriptor in descriptors:
        material_folder = os.path.join(python_folder, "..", f"results/{descriptor}")

        for defect_type in defect_types:


            defect_folder = os.path.join(material_folder, defect_type)
            csv_files = glob.glob(os.path.join(defect_folder, "*.csv"))
            double_index = next(i for i, s in enumerate(csv_files) if s.endswith("double.csv"))
            data = pd.read_csv(csv_files[double_index])
            data_single = pd.read_csv(csv_files[double_index - 1])

            print(f"{descriptor}, {defect_type}, {target_variable}: single: {data_single[target_variable].mean():.6f}, double: {data[target_variable].mean():.6f}")
            print(f"{descriptor}, {defect_type}: single: {len(data_single)}, double: {len(data)}")
