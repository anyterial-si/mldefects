"""Average number of training points and mean target value per learning-curve point.

Writes N, n, nr_data_points and mean_target_variable{,_train,_test} to results.toml
(used for the x axis of the learning curves in plot.py). Run before plot.py.
"""

from itertools import combinations

import numpy as np
import toml
from sklearn.model_selection import KFold

from common import CV_RANDOM_STATE, N_SPLITS
from init_ml import N, n, data, data_single, pd, target_variable, toml_path

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
fold_indices = [test_idx for _, test_idx in kf.split(data)]

mean_target_variable = []
mean_target_variable_train = []
mean_target_variable_test = []
nr_data_points_list = []

for data_percentage_k in range(1, N_SPLITS):
    mean_target_variable_fold = []
    mean_target_variable_train_fold = []
    mean_target_variable_test_fold = []
    train_length_data_percentage = []

    for train_fold_ids in combinations(range(N_SPLITS), data_percentage_k):
        train_index = np.concatenate([fold_indices[i] for i in train_fold_ids])
        test_index = np.concatenate([fold_indices[i] for i in range(N_SPLITS) if i not in train_fold_ids])

        train_length_data_percentage.append(len(train_index))

        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_data = pd.concat([train_data, data_single], ignore_index=True)

        ml_data = pd.concat([train_data, test_data])
        mean_target_variable_fold.append(ml_data[target_variable].mean())
        mean_target_variable_train_fold.append(train_data[target_variable].mean())
        mean_target_variable_test_fold.append(test_data[target_variable].mean())

    nr_data_points_list.append(float(np.mean(train_length_data_percentage)))
    mean_target_variable.append(float(np.mean(mean_target_variable_fold)))
    mean_target_variable_train.append(float(np.mean(mean_target_variable_train_fold)))
    mean_target_variable_test.append(float(np.mean(mean_target_variable_test_fold)))

toml_data = toml.load(toml_path) if toml_path.exists() else {}
toml_data["N"] = N
toml_data["n"] = n
toml_data["mean_target_variable"] = mean_target_variable
toml_data["mean_target_variable_train"] = mean_target_variable_train
toml_data["mean_target_variable_test"] = mean_target_variable_test
toml_data["nr_data_points"] = nr_data_points_list

with open(toml_path, "w", encoding="utf-8") as file:
    toml.dump(toml_data, file)
