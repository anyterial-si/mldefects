"""List the optim_neural.py jobs that re-run the published MLP hyperparameter search of one case.

Usage: python neural_search_jobs.py --material <descriptor> --defect_type <type> --target_variable <FE|ZPL>
           [--neural_target _clean]   > jobs.txt

Prints one optim_neural.py command per configuration recorded in optim_results_neural{target}.db
and fold. The published database does not record which folds were used; configurations with
fewer than 10 recorded scores (the four-fold screening stage of the search) get folds 1-4, the
others folds 1-10. Run the jobs (e.g. one per core or cluster job), then
`python aggregate_results.py neural`, `python neural_heatmaps.py` and `python make_tables.py`.
"""
import argparse
import json
import shlex
import sqlite3
from collections import Counter

from common import DEFECT_TYPES, DESCRIPTORS, target_folder

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--material", choices=DESCRIPTORS, required=True)
parser.add_argument("--defect_type", choices=DEFECT_TYPES, required=True)
parser.add_argument("--target_variable", choices=["FE", "ZPL"], required=True)
parser.add_argument("--neural_target", default="", choices=["", "_clean"])
args = parser.parse_args()

db = target_folder(args.material, args.defect_type, args.target_variable) / f"optim_results_neural{args.neural_target}.db"
connection = sqlite3.connect(db)
n_scores = Counter(row[0] for row in connection.execute("SELECT hyperparameter_id FROM maes"))
columns = ["hidden_layer_sizes", "activation", "solver", "alpha", "max_iter", "tol", "max_fun", "random_state"]
for row in connection.execute(f"SELECT id, {', '.join(columns)} FROM hyperparameters ORDER BY id"):
    params = dict(zip(columns, row[1:]))
    params["hidden_layer_sizes"] = json.loads(params["hidden_layer_sizes"])
    folds = range(1, 11) if n_scores[row[0]] >= 10 else range(1, 5)
    for fold in folds:
        print("python optim_neural.py", f"--material {args.material} --defect_type {args.defect_type}",
              f"--target_variable {args.target_variable} --scaling standard --fold {fold}",
              f"--neural_target '{args.neural_target}'" if args.neural_target else "",
              "--neural_params", shlex.quote(json.dumps(params)))
