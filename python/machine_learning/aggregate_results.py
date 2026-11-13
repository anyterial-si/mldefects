"""Collect per-job outputs into the result files read by make_tables.py and plot.py.

  kernel  : results/jobs/kernel/*.toml (optim_kernel.py)
            -> results/<case>/optim_results_kernel{,_unscaled,_normal}.toml
               (scores[gamma][alpha index], alphas, best_params = argmin over the grid)
  neural  : results/jobs/neural/*.toml (optim_neural.py)
            -> results/<case>/optim_results_neural{,_clean}.db (one MAE row per configuration and fold)
  curves  : results/<case>/{kernel,neural}_partial_results/*.toml (ml.py)
            -> results/<case>/results.toml, MAE_percent_list_{model}_{train,test}_{d}0

Usage: python aggregate_results.py [kernel] [neural] [curves]   (default: all three)
Existing grid points / database rows / curve entries that have no new job output are kept.
"""

import sqlite3
import sys
from math import comb

import numpy as np
import toml

from common import JOBS_DIR, N_SPLITS, RESULTS_DIR, target_folder


def aggregate_kernel():
    groups = {}
    for path in sorted((JOBS_DIR / "kernel").glob("*.toml")):
        job = toml.load(path)
        key = (job["material"], job["defect_type"], job["target_variable"], job["kernel_optim_target"])
        groups.setdefault(key, []).append(job)

    for (material, defect_type, target_variable, suffix), jobs in groups.items():
        optim_path = target_folder(material, defect_type, target_variable) / f"optim_results_kernel{suffix}.toml"
        optim = toml.load(optim_path) if optim_path.exists() else {}
        alphas = jobs[0]["alphas"]
        if optim.get("alphas", alphas) != alphas:
            sys.exit(f"{optim_path}: alpha grid differs from the job outputs")
        optim["alphas"] = alphas
        scores = optim.setdefault("scores", {})
        for job in jobs:
            row = scores.setdefault(str(float(job["gamma"])), [float("nan")] * len(alphas))
            row[job["kernel_optim_alpha_index"]] = job["alpha_mae"]

        best_mae, best_gamma, best_alpha = np.inf, None, None
        for gamma_string, maes in scores.items():
            maes = np.asarray(maes, dtype=float)
            if np.all(np.isnan(maes)):
                continue
            i = int(np.nanargmin(maes))
            if maes[i] < best_mae:
                best_mae, best_gamma, best_alpha = maes[i], float(gamma_string), alphas[i]
        optim["best_params"] = {"alpha": float(best_alpha), "gamma": best_gamma}

        with open(optim_path, "w", encoding="utf-8") as file:
            toml.dump(optim, file)
        print(f"{optim_path.relative_to(RESULTS_DIR)}: {len(jobs)} jobs, best MAE {best_mae:.4f} "
              f"(alpha={best_alpha:g}, gamma={best_gamma:g})")


HYPERPARAMETER_COLUMNS = ["hidden_layer_sizes", "activation", "solver", "alpha", "max_iter", "tol", "max_fun", "random_state"]


def aggregate_neural():
    groups = {}
    for path in sorted((JOBS_DIR / "neural").glob("*.toml")):
        job = toml.load(path)
        key = (job["material"], job["defect_type"], job["target_variable"], job.get("neural_target", ""))
        groups.setdefault(key, []).append(job)

    for (material, defect_type, target_variable, suffix), jobs in groups.items():
        db_path = target_folder(material, defect_type, target_variable) / f"optim_results_neural{suffix}.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS hyperparameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hidden_layer_sizes TEXT NOT NULL, activation TEXT NOT NULL, solver TEXT NOT NULL,
            alpha REAL, max_iter INTEGER, tol REAL, max_fun INTEGER, random_state INTEGER,
            UNIQUE (hidden_layer_sizes, activation, solver, alpha, max_iter, tol, max_fun, random_state))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS maes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fold_mae REAL, hyperparameter_id INTEGER,
            FOREIGN KEY (hyperparameter_id) REFERENCES hyperparameters(id) ON DELETE CASCADE)""")
        # The published databases have no fold column; new rows record the fold so that
        # repeated runs replace each other instead of adding extra "folds".
        if "fold" not in [row[1] for row in conn.execute("PRAGMA table_info(maes)")]:
            conn.execute("ALTER TABLE maes ADD COLUMN fold INTEGER")
        for job in jobs:
            values = [job.get(c) for c in HYPERPARAMETER_COLUMNS]
            conn.execute(f"INSERT OR IGNORE INTO hyperparameters ({', '.join(HYPERPARAMETER_COLUMNS)}) "
                         f"VALUES ({', '.join('?' * len(values))})", values)
            (hp_id,) = conn.execute("SELECT id FROM hyperparameters WHERE "
                                    + " AND ".join(f"{c} IS ?" for c in HYPERPARAMETER_COLUMNS), values).fetchone()
            conn.execute("DELETE FROM maes WHERE hyperparameter_id = ? AND fold = ?", (hp_id, job["fold"]))
            conn.execute("INSERT INTO maes (fold_mae, hyperparameter_id, fold) VALUES (?, ?, ?)",
                         (job["mae_fold"], hp_id, job["fold"]))
        conn.commit()
        conn.close()
        print(f"{db_path.relative_to(RESULTS_DIR)}: {len(jobs)} fold results")


def aggregate_curves():
    for partial_folder in sorted(RESULTS_DIR.glob("*/*/*/*_partial_results")):
        model_type = partial_folder.name.removesuffix("_partial_results")
        toml_path = partial_folder.parent / "results.toml"
        toml_data = toml.load(toml_path) if toml_path.exists() else {}
        n_found = 0
        for d in range(1, N_SPLITS):
            n_folds = comb(N_SPLITS, d)
            for kind in ("train", "test"):
                key = f"MAE_percent_list_{model_type}_{kind}_{d}0"
                if len(toml_data.get(key, [])) != n_folds:
                    toml_data[key] = [0.0] * n_folds  # 0.0 = not computed (as in the published files)
            for f in range(1, n_folds + 1):
                path = partial_folder / f"partial_results_d{d}_f{f}.toml"
                if path.exists():
                    partial = toml.load(path)
                    toml_data[f"MAE_percent_list_{model_type}_train_{d}0"][f - 1] = partial["mae_train"]
                    toml_data[f"MAE_percent_list_{model_type}_test_{d}0"][f - 1] = partial["mae_test"]
                    n_found += 1
        with open(toml_path, "w", encoding="utf-8") as file:
            toml.dump(toml_data, file)
        print(f"{toml_path.relative_to(RESULTS_DIR)}: {model_type} learning curve, {n_found}/1022 fits")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["kernel", "neural", "curves"]
    for step in steps:
        {"kernel": aggregate_kernel, "neural": aggregate_neural, "curves": aggregate_curves}[step]()
