"""Test run: regenerate paper Tables 2 and 3 and compare them with the published values.

    python test_run.py            # ~5 min with 8 cores
    python test_run.py --full     # also re-runs the 30 interstitial kernel ridge cells
                                    # (each ~10-30 CPU-min and ~6 GB RAM)

What it does (the repository's results/ folder is never modified):
  1. Table 2: dataset sizes and mean FE / ZPL from data/.
  2. Copies results/ to a temporary directory (MLDEFECTS_RESULTS_DIR) and, there,
     re-runs optim_linear.py at the published optimal alpha for all 60 ridge cells and
     optim_kernel.py at the published optimal (alpha, gamma) for the 30 vacancy-substitution
     kernel ridge cells, then aggregate_results.py kernel.
  3. Checks every recomputed MAE against the published optimization files (1e-6 eV).
  4. Runs make_tables.py on the temporary results and checks all 140 MAE cells of the
     generated Table 3(a) and Table 4 against the paper (to the 3 printed decimals).

The MLP cells are not re-trained (hours of CPU time, and not bit-reproducible across
numerical environments; see README): their MAEs are taken from cv_scores.toml (Table 3)
and results.toml (Table 4).
Exit status is 0 if all checks pass.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import toml

from common import (
    DEFECT_TYPES,
    DESCRIPTORS,
    RESULTS_DIR,
    SCALING_SUFFIX,
    SCALINGS,
    TARGET_VARIABLES,
    load_datasets,
)

SCRIPT_DIR = Path(__file__).resolve().parent

SHORT = {"sic_no_wyckoff": "A", "sic_wyckoff_letter": "WL", "sic_wyckoff": "WC",
         "sic_wyckoff_letter_nw_data": "WLC", "sic_wyckoff_nw_data": "WCC"}
ROW_NAMES = {"Atomic": "A", "Wyckoff Letter": "WL", "Wyckoff Coordinate": "WC",
             "Wyckoff Letter Compare": "WLC", "Wyckoff Coordinate Compare": "WCC"}

# Paper Table 2: (single N, mean FE, mean ZPL, double N, mean FE, mean ZPL)
PAPER_TABLE_2 = {
    ("A", "vacancy-substitutional"): (170, 5.86, 0.1951, 5136, 9.08, 0.3189),
    ("A", "interstitial"): (147, 11.32, 0.1780, 30502, 10.20, 0.2542),
    ("WL", "vacancy-substitutional"): (306, 5.81, 0.1796, 5834, 9.14, 0.3137),
    ("WL", "interstitial"): (248, 9.66, 0.1767, 32556, 10.27, 0.2549),
    ("WC", "vacancy-substitutional"): (306, 5.81, 0.1796, 5834, 9.14, 0.3137),
    ("WC", "interstitial"): (547, 8.72, 0.1380, 33499, 10.25, 0.2537),
}
for _d in ("WLC", "WCC"):
    for _t in DEFECT_TYPES:
        PAPER_TABLE_2[(_d, _t)] = PAPER_TABLE_2[("A", _t)]

# Paper Table 3(a): FE/ZPL MAE (eV) in the column order of TABLE_3_COLUMNS
TABLE_3_COLUMNS = ["unscaled_ridge", "unscaled_kernel", "standard_ridge", "standard_kernel",
                   "standard_mlp", "normal_ridge", "normal_kernel"]
PAPER_TABLE_3 = {
    ("vacancy-substitutional", "A"): "2.598/0.270 1.263/0.202 2.598/0.270 2.003/0.210 0.478/0.234 2.597/0.270 2.020/0.258",
    ("vacancy-substitutional", "WL"): "2.576/0.266 1.314/0.242 2.575/0.266 1.986/0.250 0.496/0.252 2.572/0.266 2.008/0.257",
    ("vacancy-substitutional", "WC"): "2.576/0.266 1.314/0.242 2.575/0.266 1.986/0.250 0.485/0.252 2.572/0.266 2.008/0.257",
    ("vacancy-substitutional", "WLC"): "2.595/0.270 1.314/0.248 2.594/0.270 2.004/0.259 0.528/0.262 2.592/0.270 2.021/0.262",
    ("vacancy-substitutional", "WCC"): "2.595/0.270 1.314/0.248 2.594/0.270 2.004/0.259 0.545/0.257 2.592/0.270 2.021/0.262",
    ("interstitial", "A"): "3.562/0.280 2.530/0.230 3.559/0.280 3.072/0.254 2.582/0.246 3.554/0.280 3.047/0.259",
    ("interstitial", "WL"): "3.462/0.281 2.420/0.232 3.460/0.281 2.851/0.259 1.523/0.247 3.461/0.281 2.880/0.265",
    ("interstitial", "WC"): "3.301/0.280 1.818/0.237 3.301/0.280 2.214/0.260 1.123/0.257 3.301/0.280 2.418/0.266",
    ("interstitial", "WLC"): "3.482/0.279 2.410/0.230 3.480/0.279 2.850/0.257 1.633/0.248 3.481/0.279 2.882/0.263",
    ("interstitial", "WCC"): "3.326/0.279 1.841/0.235 3.326/0.279 2.225/0.259 1.182/0.258 3.326/0.279 2.422/0.265",
}
PAPER_TABLE_4 = "2.563 1.088 0.379 0.266 0.166 0.171 3.300 1.540 0.820 0.279 0.223 0.235"


def paper_table_3():
    values = {}
    for (defect_type, descriptor), row in PAPER_TABLE_3.items():
        for column, cell in zip(TABLE_3_COLUMNS, row.split()):
            fe, zpl = map(float, cell.split("/"))
            values[(descriptor, defect_type, "FE", column)] = fe
            values[(descriptor, defect_type, "ZPL", column)] = zpl
    return values


class Checker:
    def __init__(self):
        self.failures = 0

    def check(self, ok, message):
        if not ok:
            self.failures += 1
            print(f"  FAIL {message}")
        return ok


def check_table_2(checker):
    print("1. Table 2 (data/)")
    n_ok = 0
    for descriptor in DESCRIPTORS:
        for defect_type in DEFECT_TYPES:
            data, single = load_datasets(descriptor, defect_type, columns=["FE", "ZPL"])
            got = (len(single), single.FE.mean(), single.ZPL.mean(), len(data), data.FE.mean(), data.ZPL.mean())
            ref = PAPER_TABLE_2[(SHORT[descriptor], defect_type)]
            tolerances = (0, 0.005, 0.00005, 0, 0.005, 0.00005)
            ok = all(abs(g - r) <= t + 1e-12 for g, r, t in zip(got, ref, tolerances))
            n_ok += checker.check(ok, f"{descriptor} {defect_type}: got {got}, paper {ref}")
    print(f"   {n_ok}/10 rows match the paper")


def run_jobs(commands, env, n_jobs):
    def run(command):
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        return command, result

    failed = 0
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for command, result in pool.map(run, commands):
            if result.returncode != 0:
                failed += 1
                print("  FAIL", " ".join(command[1:]), "\n", result.stderr[-2000:])
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="also re-run the interstitial kernel ridge cells")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="parallel jobs (default: min(8, CPUs))")
    parser.add_argument("--keep", action="store_true", help="keep the temporary results directory")
    args = parser.parse_args()

    start = time.time()
    checker = Checker()
    check_table_2(checker)

    tmp = Path(tempfile.mkdtemp(prefix="mldefects_test_"))
    tmp_results = tmp / "results"
    shutil.copytree(RESULTS_DIR, tmp_results, ignore=shutil.ignore_patterns("*_partial_results", "weights", "figures", "jobs", "tables"))
    env = dict(os.environ, MLDEFECTS_RESULTS_DIR=str(tmp_results), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    python = sys.executable

    commands, expected = [], []
    for descriptor in DESCRIPTORS:
        for defect_type in DEFECT_TYPES:
            for target_variable in TARGET_VARIABLES:
                folder = RESULTS_DIR / descriptor / defect_type / target_variable
                case = ["--material", descriptor, "--defect_type", defect_type, "--target_variable", target_variable]
                for scaling in SCALINGS:
                    suffix = SCALING_SUFFIX[scaling]
                    linear = toml.load(folder / f"optim_results_linear{suffix}.toml")
                    alpha = linear["best_params"]["alpha"]
                    commands.append([python, str(SCRIPT_DIR / "optim_linear.py"), *case, "--scaling", scaling,
                                     "--alphas_argparse", repr([alpha])])
                    expected.append(("linear", descriptor, defect_type, target_variable, scaling,
                                     None, None, min(linear["scores"]["mae"])))
                    if defect_type == "interstitial" and not args.full:
                        continue
                    kernel = toml.load(folder / f"optim_results_kernel{suffix}.toml")
                    gamma, k_alpha = kernel["best_params"]["gamma"], kernel["best_params"]["alpha"]
                    index = min(range(len(kernel["alphas"])), key=lambda i: abs(kernel["alphas"][i] - k_alpha))
                    commands.append([python, str(SCRIPT_DIR / "optim_kernel.py"), *case, "--scaling", scaling,
                                     "--kernel_optim_target", suffix, "--gamma", repr(gamma),
                                     "--kernel_optim_alpha_index", str(index)])
                    expected.append(("kernel", descriptor, defect_type, target_variable, scaling,
                                     str(gamma), index, kernel["scores"][str(gamma)][index]))

    # kernel jobs first: they are the slow ones
    order = sorted(range(len(commands)), key=lambda i: expected[i][0] != "kernel")
    print(f"2. Re-running {len(commands)} optimization jobs at the published optima ({args.jobs} parallel) in {tmp_results}")
    checker.failures += run_jobs([commands[i] for i in order], env, args.jobs)
    subprocess.run([python, str(SCRIPT_DIR / "aggregate_results.py"), "kernel"], env=env, check=True, capture_output=True)

    print("3. Recomputed MAE vs published optimization results")
    worst = 0.0
    for model, descriptor, defect_type, target_variable, scaling, gamma, index, published in expected:
        folder = tmp_results / descriptor / defect_type / target_variable
        optim = toml.load(folder / f"optim_results_{model}{SCALING_SUFFIX[scaling]}.toml")
        if model == "linear":
            recomputed = optim["scores"]["mae"][0]
        else:
            recomputed = optim["scores"][gamma][index]
        worst = max(worst, abs(recomputed - published))
        checker.check(abs(recomputed - published) < 1e-6,
                      f"{model} {descriptor} {defect_type} {target_variable} {scaling}: {recomputed} vs {published}")
    print(f"   {len(expected)} cells, max |difference| = {worst:.1e} eV")

    print("4. Table 3(a) regenerated with make_tables.py vs paper")
    subprocess.run([python, str(SCRIPT_DIR / "make_tables.py")], env=env, check=True, capture_output=True)
    lines = (tmp_results / "tables" / "all_results_table.tex").read_text().splitlines()
    paper = paper_table_3()
    n_cells, defect_type = 0, None
    for i, line in enumerate(lines):
        if "Vacancy-Substitution" in line:
            defect_type = "vacancy-substitutional"
        elif "multirow" in line and "Interstitial" in line:
            defect_type = "interstitial"
        match = re.match(r"^& (" + "|".join(ROW_NAMES) + r")\s*$", line)
        if match and defect_type and n_cells < 140:
            numbers = [float(x) for x in re.findall(r"(?<![\w{^])\d+\.\d{3}", " ".join(lines[i + 1:i + 4]))]
            for k, column in enumerate(TABLE_3_COLUMNS):
                for j, target_variable in enumerate(TARGET_VARIABLES):
                    reference = paper[(ROW_NAMES[match.group(1)], defect_type, target_variable, column)]
                    n_cells += 1
                    checker.check(abs(numbers[2 * k + j] - reference) < 1e-9,
                                  f"Table 3 {match.group(1)} {defect_type} {target_variable} {column}: "
                                  f"{numbers[2 * k + j]} vs paper {reference}")
    checker.check(n_cells == 140, f"parsed {n_cells} Table 3 cells, expected 140")
    print(f"   {n_cells} cells compared")

    projections = re.findall(r"\d+\.\d{3}", (tmp_results / "tables" / "projections.tex").read_text().split("\\midrule", 1)[1])
    checker.check(projections[:12] == PAPER_TABLE_4.split(),
                  f"Table 4 {' '.join(projections[:12])} vs paper {PAPER_TABLE_4}")
    print(f"   Table 4: {' '.join(projections[:12])}")

    if args.keep:
        print(f"Temporary results kept in {tmp}")
    else:
        shutil.rmtree(tmp)
    status = "PASSED" if checker.failures == 0 else f"FAILED ({checker.failures} failures)"
    print(f"Test run {status} in {time.time() - start:.0f} s")
    sys.exit(1 if checker.failures else 0)


if __name__ == "__main__":
    main()
