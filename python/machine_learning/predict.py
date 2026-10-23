import os
import sys
import pandas as pd
import joblib
from init_ml import features

defect_types = ["vacancy-substitutional", "interstitial"]
materials = ["sic_no_wyckoff", "sic_wyckoff_letter", "sic_wyckoff"]
target_variables = ["FE", "ZPL"]

python_folder = os.path.join(os.getcwd(), "..")
database_folder = os.path.join(python_folder, "database")
sys.path.append(database_folder)

for extend_data_method in ["", "_cluster"]:
    for defect_type in defect_types:
        for material in materials:

            if material == "sic_wyckoff":
                from input_parameters import sic_input_parameters as features  # type: ignore
            elif material == "sic_wyckoff_letter":
                from input_parameters import sic_wyckoff_letter_input_parameters as features  # type: ignore
            elif material == "sic_no_wyckoff":
                from input_parameters import atom_distance_charge_spin_input_parameters as features  # type: ignore

            csv_dir = os.path.join(f"./../database/extend_data{extend_data_method}/double_{defect_type}_{material}.csv")
            data = pd.read_csv(csv_dir)
            test_data = data[features]

            test_X = test_data[features]

            for target_variable in target_variables:

                model = joblib.load(f"./saved_models/{material}_{defect_type}_{target_variable}_model.joblib")

                if os.path.exists(f"./saved_models/{material}_{defect_type}_{target_variable}_scaler_X.joblib"):
                    scaling = "scaled"

                    scaler_X = joblib.load(f"./saved_models/{material}_{defect_type}_{target_variable}_scaler_X.joblib")
                    scaler_y = joblib.load(f"./saved_models/{material}_{defect_type}_{target_variable}_scaler_y.joblib")

                    test_X_scaled = scaler_X.transform(test_X)
                    test_X_scaled = pd.DataFrame(test_X_scaled, columns=test_X.columns)

                else:
                    scaling = "unscaled"
                    test_X_scaled = test_data[features]

                y_pred_test_scaled = model.predict(test_X_scaled)

                if scaling == "unscaled":
                    y_pred_test = y_pred_test_scaled
                else:
                    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

                data[f"pred_{target_variable}"] = y_pred_test

            data = data.sort_values(by=f"pred_FE")
            data.to_csv(csv_dir, index=False)
