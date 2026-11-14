"""Per-fold model weights for the 10-fold cross-validation behind Table 3, Table 4 and Fig. 4.

Every Table 3 cell (descriptor x defect type x target x model x scaling) is trained on each of
the 10 cross-validation training sets with the hyperparameters Table 3 reports, and the
fitted weights are saved. The tables can then be regenerated from the weights alone:

  python cv_models.py train --descriptor sic_no_wyckoff --defect_type vacancy-substitutional \\
                            --target_variable FE --model neural [--scaling standard] [--fold 3]
  python cv_models.py evaluate          # weights -> results/<case>/cv_scores.toml
  python make_tables.py --from-weights  # Table 3 and Table 4 from cv_scores.toml

Weights are written to results/<descriptor>/<defect_type>/<target>/weights/{model}{suffix}_fold{f}.npz
(suffix as in the optimization files: "" standard, "_unscaled", "_normal"). Fold f uses the same
numbering as the other scripts: the f-th combination of 9 training folds, i.e. KFold split 11 - f
is the test fold. Each file holds the test-set row indices, the scaler parameters, the weights,
the hyperparameters, the MAEs and the software versions it was trained with.

Hyperparameters: ridge and kernel ridge from best_params in optim_results_{linear,kernel}{suffix}.toml;
MLP from the configuration with the lowest mean MAE in optim_results_neural_clean.db (the
selection make_tables.py uses). Kernel ridge weights are the dual coefficients; the training
inputs they multiply are rebuilt from data/ and the stored indices.

`evaluate` predicts with NumPy only (plus sklearn's rbf_kernel), not with pickled estimators,
so the files do not depend on the scikit-learn version.
"""

import argparse
import datetime
import json
import platform
import sqlite3
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import toml
from sklearn.model_selection import KFold

from common import (CV_RANDOM_STATE, DEFECT_TYPES, DESCRIPTORS, N_SPLITS, RESULTS_DIR, SCALING_SUFFIX,
                    TARGET_VARIABLES, ensure_dir, load_datasets, target_folder)
from input_parameters import FEATURES

MODELS = {"ridge": ("unscaled", "standard", "normal"),
          "kernel": ("unscaled", "standard", "normal"),
          "neural": ("standard",)}


def cells():
    """All Table 3 cells as (descriptor, defect_type, target_variable, model, scaling)."""
    for descriptor in DESCRIPTORS:
        for defect_type in DEFECT_TYPES:
            for target_variable in TARGET_VARIABLES:
                for model, scalings in MODELS.items():
                    for scaling in scalings:
                        yield descriptor, defect_type, target_variable, model, scaling


def cell_name(model, scaling):
    return f"{model}{SCALING_SUFFIX[scaling]}"


def weights_path(descriptor, defect_type, target_variable, model, scaling, fold, name=None):
    return (target_folder(descriptor, defect_type, target_variable) / "weights"
            / f"{name or cell_name(model, scaling)}_fold{fold}.npz")


def curve_weights_path(descriptor, defect_type, target_variable, model, scaling, d, fold):
    """Weights of one learning-curve fit (ml.py --save_weights): d tenths of the data, fold-th combination."""
    return (target_folder(descriptor, defect_type, target_variable) / "weights" / "curves"
            / f"{cell_name(model, scaling)}_d{d}_f{fold}.npz")


def hyperparameters(descriptor, defect_type, target_variable, model, scaling):
    folder = target_folder(descriptor, defect_type, target_variable)
    if model in ("ridge", "kernel"):
        name = {"ridge": "linear", "kernel": "kernel"}[model]
        best = toml.load(folder / f"optim_results_{name}{SCALING_SUFFIX[scaling]}.toml")["best_params"]
        return {k: float(v) for k, v in best.items()}
    conn = sqlite3.connect(folder / "optim_results_neural_clean.db")
    ids = [row[0] for row in conn.execute("SELECT id FROM hyperparameters")]
    means = [np.mean([r[0] for r in conn.execute("SELECT fold_mae FROM maes WHERE hyperparameter_id = ?", (i,))])
             for i in ids]
    row = conn.execute("SELECT hidden_layer_sizes, activation, solver, alpha, max_iter, tol, max_fun, random_state "
                       "FROM hyperparameters WHERE id = ?", (ids[int(np.argmin(means))],)).fetchone()
    conn.close()
    return {"hidden_layer_sizes": json.loads(row[0]), "activation": row[1], "solver": row[2], "alpha": row[3],
            "max_iter": row[4], "tol": row[5], "max_fun": row[6], "random_state": row[7]}


def split(data, fold):
    """Row positions (in the double-defect data) of training and test set for fold 1..10."""
    fold_indices = [test for _, test in KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE).split(data)]
    train_ids = list(combinations(range(N_SPLITS), N_SPLITS - 1))[fold - 1]
    train_index = np.concatenate([fold_indices[i] for i in train_ids])
    test_index = np.concatenate([fold_indices[i] for i in range(N_SPLITS) if i not in train_ids])
    return train_index, test_index


def frames(data, data_single, train_index, test_index):
    return pd.concat([data.iloc[train_index], data_single], ignore_index=True), data.iloc[test_index]


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.ravel(y_true) - np.ravel(y_pred))))


def environment():
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "machine": platform.machine(),
            "threadpools": [{k: p.get(k) for k in ("internal_api", "version", "num_threads")} for p in threadpool_info()],
            "date": datetime.date.today().isoformat()}


def weight_arrays(model, estimator, scaling, scaler_X=None, scaler_y=None):
    """The arrays saved for a fitted estimator: scaler parameters and model weights."""
    arrays = {}
    if scaling != "unscaled":
        for name, scaler in (("x", scaler_X), ("y", scaler_y)):
            arrays[f"{name}_scale"] = scaler.scale_
            arrays[f"{name}_shift"] = scaler.mean_ if scaling == "standard" else scaler.min_
    if model == "ridge":
        arrays["coef"], arrays["intercept"] = np.ravel(estimator.coef_), np.ravel(estimator.intercept_)
    elif model == "kernel":
        arrays["dual_coef"] = np.ravel(estimator.dual_coef_)
    else:
        for i, (w, b) in enumerate(zip(estimator.coefs_, estimator.intercepts_)):
            arrays[f"W{i}"], arrays[f"b{i}"] = w, b
    return arrays


def save_weights_file(path, meta, arrays, test_index, estimator):
    """Write one weight file; meta gets the environment and, for MLPs, n_iter and n_layers."""
    meta = dict(meta, environment=environment())
    if meta["model"] == "neural":
        meta["n_iter"] = int(estimator.n_iter_)
        meta["n_layers"] = len(estimator.coefs_)
    np.savez_compressed(ensure_dir(path.parent) / path.name, meta=json.dumps(meta), test_index=test_index, **arrays)


def train(descriptor, defect_type, target_variable, model, scaling, folds, params=None, name=None):
    """Fit the cell on the given folds exactly as optim_*.py / ml.py do, and save the weights.

    params, name: train another MLP configuration than the Table 3 one (e.g. those compared in the
    Supplementary Material) and save it as weights/{name}_fold{f}.npz."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    features = FEATURES[descriptor]
    data, data_single = load_datasets(descriptor, defect_type)
    params = params or hyperparameters(descriptor, defect_type, target_variable, model, scaling)

    for fold in folds:
        train_index, test_index = split(data, fold)
        train_data, test_data = frames(data, data_single, train_index, test_index)
        train_X, train_y = train_data[features], train_data[[target_variable]]
        test_X, test_y = test_data[features], test_data[[target_variable]]
        scaler_X = scaler_y = None

        if scaling == "unscaled":
            train_X_scaled, train_y_scaled, test_X_scaled = train_X, train_data[target_variable], test_X
        else:
            scaler_X, scaler_y = (StandardScaler(), StandardScaler()) if scaling == "standard" else (MinMaxScaler(), MinMaxScaler())
            scaler_X.fit(train_X)
            scaler_y.fit(train_y.values.reshape(-1, 1))
            train_X_scaled = pd.DataFrame(scaler_X.transform(train_X), columns=train_X.columns)
            train_y_scaled = pd.DataFrame(scaler_y.transform(train_y.values.reshape(-1, 1)), columns=train_y.columns)
            test_X_scaled = pd.DataFrame(scaler_X.transform(test_X), columns=test_X.columns)

        if model == "ridge":
            estimator = Ridge(alpha=params["alpha"]).fit(train_X_scaled, train_y_scaled)
        elif model == "kernel":
            estimator = KernelRidge(alpha=params["alpha"], gamma=params["gamma"], kernel="rbf").fit(train_X_scaled, train_y_scaled)
        else:
            estimator = MLPRegressor(**params).fit(train_X_scaled, train_y_scaled.to_numpy().ravel())
        arrays = weight_arrays(model, estimator, scaling, scaler_X, scaler_y)

        def unscale(y):
            return y if scaling == "unscaled" else scaler_y.inverse_transform(np.reshape(y, (-1, 1)))

        meta = {"descriptor": descriptor, "defect_type": defect_type, "target_variable": target_variable,
                "model": model, "scaling": scaling, "fold": fold, "hyperparameters": params,
                "mae_train": mae(train_y, unscale(estimator.predict(train_X_scaled))),
                "mae_test": mae(test_y, unscale(estimator.predict(test_X_scaled))),
                "n_train": len(train_data), "n_test": len(test_data)}
        path = weights_path(descriptor, defect_type, target_variable, model, scaling, fold, name)
        save_weights_file(path, meta, arrays, test_index, estimator)
        print(f"{path.relative_to(RESULTS_DIR)}: test MAE {meta['mae_test']:.4f}, train MAE {meta['mae_train']:.4f}", flush=True)


ACTIVATIONS = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0), "identity": lambda x: x,
               "logistic": lambda x: 1 / (1 + np.exp(-x))}


def predict(weights, meta, X, X_train=None):
    """Predictions in eV from saved weights. X, X_train: unscaled input parameters (arrays)."""
    scaling, model = meta["scaling"], meta["model"]

    def scale_x(A):
        A = np.array(A, dtype=float)
        if scaling == "standard":
            return (A - weights["x_shift"]) / weights["x_scale"]
        if scaling == "normal":
            return A * weights["x_scale"] + weights["x_shift"]
        return A

    A = scale_x(X)
    if model == "ridge":
        y = A @ weights["coef"] + weights["intercept"][0]
    elif model == "kernel":
        from sklearn.metrics.pairwise import rbf_kernel
        B = scale_x(X_train)
        gamma = meta["hyperparameters"]["gamma"]
        y = np.concatenate([rbf_kernel(A[i:i + 2000], B, gamma=gamma) @ weights["dual_coef"]
                            for i in range(0, len(A), 2000)])
    else:
        activation = ACTIVATIONS[meta["hyperparameters"]["activation"]]
        n_layers = meta["n_layers"]
        for i in range(n_layers):
            A = A @ weights[f"W{i}"] + weights[f"b{i}"]
            if i < n_layers - 1:
                A = activation(A)
        y = A.ravel()
    if scaling == "standard":
        y = y * weights["y_scale"][0] + weights["y_shift"][0]
    elif scaling == "normal":
        y = (y - weights["y_shift"][0]) / weights["y_scale"][0]
    return np.ravel(y)


def fold_predictions(descriptor, defect_type, target_variable, model, scaling, fold):
    """(true, predicted) test-set values of one fold, computed from its weight file."""
    path = weights_path(descriptor, defect_type, target_variable, model, scaling, fold)
    with np.load(path) as stored:
        weights = dict(stored)
    meta = json.loads(str(weights.pop("meta")))
    features = FEATURES[descriptor]
    data, data_single = load_datasets(descriptor, defect_type)
    train_data, test_data = frames(data, data_single, *split(data, fold))
    pred = predict(weights, meta, test_data[features].to_numpy(), train_data[features].to_numpy())
    return test_data[target_variable].to_numpy(), pred


def matching_fold(descriptor, defect_type, target_variable, true_values):
    """The fold whose test set has exactly these target values (in order), or None."""
    data, _ = load_datasets(descriptor, defect_type, columns=[target_variable])
    for fold in range(1, N_SPLITS + 1):
        test = data[target_variable].to_numpy()[split(data, fold)[1]]
        if len(test) == len(true_values) and np.allclose(test, true_values, rtol=0, atol=1e-12):
            return fold
    return None


def published_score(descriptor, defect_type, target_variable, model, scaling):
    """The Table 3 value as stored in the optimization results (mean test MAE of the 10 folds)."""
    folder = target_folder(descriptor, defect_type, target_variable)
    suffix = SCALING_SUFFIX[scaling]
    if model == "ridge":
        return float(min(toml.load(folder / f"optim_results_linear{suffix}.toml")["scores"]["mae"]))
    if model == "kernel":
        optim = toml.load(folder / f"optim_results_kernel{suffix}.toml")
        return float(min(optim["scores"][str(optim["best_params"]["gamma"])]))
    conn = sqlite3.connect(folder / "optim_results_neural_clean.db")
    ids = [row[0] for row in conn.execute("SELECT id FROM hyperparameters")]
    score = min(np.mean([r[0] for r in conn.execute("SELECT fold_mae FROM maes WHERE hyperparameter_id = ?", (i,))])
                for i in ids)
    conn.close()
    return float(score)


def evaluate():
    """Recompute every fold's MAEs from the weight files and write results/<case>/cv_scores.toml."""
    by_case = {}
    for cell in cells():
        by_case.setdefault(cell[:3], []).append(cell[3:])

    n_complete = n_missing = 0
    print(f"{'cell':62} {'weights':>8} {'stored':>8} {'diff':>9}")
    for (descriptor, defect_type, target_variable), model_scalings in by_case.items():
        features = FEATURES[descriptor]
        data = data_single = None
        scores = {}
        for model, scaling in model_scalings:
            paths = [weights_path(descriptor, defect_type, target_variable, model, scaling, f) for f in range(1, N_SPLITS + 1)]
            if not all(p.exists() for p in paths):
                n_missing += 1
                continue
            if data is None:
                data, data_single = load_datasets(descriptor, defect_type)
            test_maes, train_maes, metas = [], [], []
            for fold, path in enumerate(paths, start=1):
                with np.load(path) as stored:
                    weights = dict(stored)
                meta = json.loads(str(weights.pop("meta")))
                train_index, test_index = split(data, fold)
                if not np.array_equal(test_index, weights["test_index"]):
                    sys.exit(f"{path}: stored test indices differ from the cross-validation split of data/")
                train_data, test_data = frames(data, data_single, train_index, test_index)
                X_train = train_data[features].to_numpy()
                test_mae = mae(test_data[target_variable], predict(weights, meta, test_data[features].to_numpy(), X_train))
                train_mae = mae(train_data[target_variable], predict(weights, meta, X_train, X_train))
                if abs(test_mae - meta["mae_test"]) > 1e-6 or abs(train_mae - meta["mae_train"]) > 1e-6:  # roundoff in near-singular kernel ridge fits
                    sys.exit(f"{path}: MAE from the weights differs from the MAE recorded at training time")
                test_maes.append(test_mae)
                train_maes.append(train_mae)
                metas.append(meta)
            name = cell_name(model, scaling)
            scores[name] = {"test": test_maes, "train": train_maes,
                            "mean_test": float(np.mean(test_maes)), "mean_train": float(np.mean(train_maes)),
                            "hyperparameters": metas[0]["hyperparameters"],
                            "environment": metas[0]["environment"]}
            if model == "neural":
                scores[name]["n_iter"] = [m["n_iter"] for m in metas]
            stored = published_score(descriptor, defect_type, target_variable, model, scaling)
            n_complete += 1
            label = f"{descriptor}/{defect_type}/{target_variable}/{name}"
            print(f"{label:62} {scores[name]['mean_test']:8.4f} {stored:8.4f} {scores[name]['mean_test'] - stored:+9.2e}")
        if scores:
            path = target_folder(descriptor, defect_type, target_variable) / "cv_scores.toml"
            existing = toml.load(path) if path.exists() else {}
            existing.update(scores)
            with open(path, "w", encoding="utf-8") as file:
                toml.dump(existing, file)
    print(f"{n_complete} complete cells, {n_missing} cells without (all) weight files")


_extra_cache = {}


def _extra_case(case):
    """Datasets, CV folds and learning curves of one case folder (cached per process)."""
    if case not in _extra_cache:
        descriptor, defect_type, _ = case.relative_to(RESULTS_DIR).parts
        data, data_single = load_datasets(descriptor, defect_type)
        fold_indices = [test for _, test in KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE).split(data)]
        curves = toml.load(case / "results.toml") if (case / "results.toml").exists() else {}
        _extra_cache[case] = data, data_single, fold_indices, curves
    return _extra_cache[case]


def _check_extra_file(path):
    """Recompute the MAEs of one non-Table-3 weight file; returns (path, meta, test MAE, train MAE,
    difference to results.toml or None) or raises ValueError on a mismatch."""
    case = path.parent.parent if path.parent.name == "curves" else path.parent
    case = case.parent
    descriptor, defect_type, target_variable = case.relative_to(RESULTS_DIR).parts
    features = FEATURES[descriptor]
    data, data_single, fold_indices, curves = _extra_case(case)
    with np.load(path) as stored:
        weights = dict(stored)
    meta = json.loads(str(weights.pop("meta")))
    if "train_folds" in meta:   # learning-curve fit
        train_ids = meta["train_folds"]
        train_index = np.concatenate([fold_indices[i] for i in train_ids])
        test_index = np.concatenate([fold_indices[i] for i in range(N_SPLITS) if i not in train_ids])
    else:
        train_index, test_index = split(data, meta["fold"])
    if not np.array_equal(test_index, weights["test_index"]):
        raise ValueError(f"{path}: stored test indices differ from the split of data/")
    train_data, test_data = frames(data, data_single, train_index, test_index)
    X_train = train_data[features].to_numpy()
    test_mae = mae(test_data[target_variable], predict(weights, meta, test_data[features].to_numpy(), X_train))
    train_mae = mae(train_data[target_variable], predict(weights, meta, X_train, X_train))
    if abs(test_mae - meta["mae_test"]) > 1e-6 or abs(train_mae - meta["mae_train"]) > 1e-6:
        raise ValueError(f"{path}: MAE from the weights differs from the MAE recorded at training time")
    worst = None
    if "train_folds" in meta:
        model = {"ridge": "linear"}.get(meta["model"], meta["model"])
        d, f = meta["train_data_percentage"], meta["fold"]
        worst = 0.0
        for kind, value in (("test", test_mae), ("train", train_mae)):
            stored = curves.get(f"MAE_percent_list_{model}_{kind}_{d}0", [])
            if len(stored) < f or abs(stored[f - 1] - value) > 1e-6:
                raise ValueError(f"{path}: {kind} MAE {value:.6f} differs from results.toml")
            worst = max(worst, abs(stored[f - 1] - value))
    return path, meta, test_mae, train_mae, worst


def evaluate_extra(jobs=1):
    """Check every other weight file (learning-curve fits from ml.py --save_weights, and named
    configurations from train --name) against the MAEs recorded at training time and, for the
    learning curves, against results.toml. Writes the named configurations' scores to cv_scores.toml.
    With jobs > 1 the files of each case are checked in that many processes."""
    from concurrent.futures import ProcessPoolExecutor
    table_names = {f"{cell_name(m, s)}_fold{f}.npz" for m, ss in MODELS.items() for s in ss for f in range(1, N_SPLITS + 1)}
    for folder in sorted(RESULTS_DIR.glob("*/*/*/weights")):
        paths = sorted(p for p in folder.rglob("*.npz") if p.name not in table_names)
        if not paths:
            continue
        case = folder.parent
        try:
            if jobs > 1:
                with ProcessPoolExecutor(jobs) as pool:
                    results = list(pool.map(_check_extra_file, paths, chunksize=4))
            else:
                results = [_check_extra_file(path) for path in paths]
        except ValueError as error:
            sys.exit(str(error))
        named, worst = {}, 0.0
        for path, meta, test_mae, train_mae, diff in results:
            if diff is not None:
                worst = max(worst, diff)
            else:
                name = path.stem.rsplit("_fold", 1)[0]
                entry = named.setdefault(name, {"test": [None] * N_SPLITS, "train": [None] * N_SPLITS,
                                                "hyperparameters": meta["hyperparameters"],
                                                "environment": meta["environment"]})
                entry["test"][meta["fold"] - 1], entry["train"][meta["fold"] - 1] = test_mae, train_mae
        n_curves = sum("curves" in p.parts for p in paths)
        if n_curves:
            print(f"{case.relative_to(RESULTS_DIR)}: {n_curves} learning-curve fits match results.toml (max diff {worst:.1e})", flush=True)
        for name, entry in named.items():
            if None in entry["test"]:
                print(f"{case.relative_to(RESULTS_DIR)}/{name}: incomplete ({entry['test'].count(None)} folds missing)")
                continue
            entry["mean_test"], entry["mean_train"] = float(np.mean(entry["test"])), float(np.mean(entry["train"]))
            path = case / "cv_scores.toml"
            existing = toml.load(path) if path.exists() else {}
            existing[name] = entry
            with open(path, "w", encoding="utf-8") as file:
                toml.dump(existing, file)
            print(f"{case.relative_to(RESULTS_DIR)}/{name}: mean test MAE {entry['mean_test']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--descriptor", choices=DESCRIPTORS, required=True)
    p.add_argument("--defect_type", choices=DEFECT_TYPES, required=True)
    p.add_argument("--target_variable", choices=TARGET_VARIABLES, required=True)
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--scaling", choices=["unscaled", "standard", "normal"], default="standard")
    p.add_argument("--fold", type=int, nargs="*", default=list(range(1, N_SPLITS + 1)))
    p.add_argument("--neural_params", type=json.loads, help="JSON dict of MLPRegressor parameters (default: Table 3)")
    p.add_argument("--name", help="file name prefix for --neural_params models: weights/{name}_fold{f}.npz")
    sub.add_parser("evaluate")
    p = sub.add_parser("evaluate-extra")
    p.add_argument("--jobs", type=int, default=1, help="number of processes (default 1)")
    args = parser.parse_args()
    if args.command == "train":
        if args.scaling not in MODELS[args.model]:
            sys.exit(f"Table 3 has no {args.model} model with {args.scaling} scaling")
        if (args.neural_params is None) != (args.name is None) or (args.name and args.model != "neural"):
            sys.exit("--neural_params and --name go together, with --model neural")
        train(args.descriptor, args.defect_type, args.target_variable, args.model, args.scaling, args.fold,
              args.neural_params, args.name)
    elif args.command == "evaluate":
        evaluate()
    else:
        evaluate_extra(args.jobs)
