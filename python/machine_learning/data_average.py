import toml

from itertools import combinations
import numpy as np
from sklearn.model_selection import KFold

from init_ml import (
    N,
    n,
    data,
    data_single,
    pd,
    target_variable,
    toml_path,
    cross_validation,
)

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

folds = list(kf.split(data))
fold_indices = [test_idx for _, test_idx in folds]

mean_target_variable = []
mean_target_variable_train = []
mean_target_variable_test = []

if cross_validation == "varied":
    data_percentage_k_limit = 10
else:
    data_percentage_k_limit = 11
nr_data_points_list = [0] * (data_percentage_k_limit - 1)

for data_percentage_k in range(1, data_percentage_k_limit):

    mean_target_variable_fold = []
    mean_target_variable_train_fold = []
    mean_target_variable_test_fold = []

    if cross_validation == "varied":
        folds = list(kf.split(data))
        fold_indices = [test_idx for _, test_idx in folds]

        combos_iter = combinations(range(n_splits), data_percentage_k)
        combos_list = list(combos_iter)
    else:
        fraction_data = data.sample(frac=data_percentage_k/10, random_state=42)
        folds = list(kf.split(fraction_data))
        fold_indices = [test_idx for _, test_idx in folds]

        combos_iter = combinations(range(n_splits), 9)
        combos_list = list(combos_iter)

    train_length_data_percentage = []
    for train_fold_ids in combos_list:
        train_index = np.concatenate([fold_indices[i] for i in train_fold_ids])
        test_index = np.concatenate([fold_indices[i] for i in range(n_splits) if i not in train_fold_ids])

        train_length_data_percentage.append(len(train_index))

        if cross_validation == "varied":
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]
        else:
            train_data = fraction_data.iloc[train_index]
            test_data = fraction_data.iloc[test_index]
        train_data = pd.concat([train_data, data_single], ignore_index=True)

        ml_data = pd.concat([train_data, test_data])
        mean_target_variable_fold.append(ml_data[target_variable].mean())
        mean_target_variable_train_fold.append(train_data[target_variable].mean())
        mean_target_variable_test_fold.append(test_data[target_variable].mean())

    nr_data_points_list[data_percentage_k-1] = np.mean(train_length_data_percentage)

    mean_target_variable.append(np.mean(mean_target_variable_fold))
    mean_target_variable_train.append(np.mean(mean_target_variable_train_fold))
    mean_target_variable_test.append(np.mean(mean_target_variable_test_fold))

mean_target_variable = [float(val) for val in mean_target_variable]
mean_target_variable_train = [float(val) for val in mean_target_variable_train]
mean_target_variable_test = [float(val) for val in mean_target_variable_test]
nr_data_points_list = [float(val) for val in nr_data_points_list]

toml_data = toml.load(toml_path)
toml_data = toml.load(toml_path)
toml_data["N"] = N
toml_data["n"] = n

if cross_validation == "varied":
    toml_data["mean_target_variable"] = mean_target_variable
    toml_data["mean_target_variable_train"] = mean_target_variable_train
    toml_data["mean_target_variable_test"] = mean_target_variable_test
    toml_data["nr_data_points"] = nr_data_points_list

else:
    toml_data["mean_target_variable_10_fold"] = mean_target_variable
    toml_data["mean_target_variable_train_10_fold"] = mean_target_variable_train
    toml_data["mean_target_variable_test_10_fold"] = mean_target_variable_test
    toml_data["nr_data_points_10_fold"] = nr_data_points_list

with open(toml_path, "w", encoding="utf-8") as file:
    toml.dump(toml_data, file)
