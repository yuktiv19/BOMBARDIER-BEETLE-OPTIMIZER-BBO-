"""
CEC-2017 Experiment Runner
==========================
Runs all three BBO variants on every CEC-2017 benchmark function and saves
results to CSV files in results/csv/.

Algorithms compared:
  1. BBO-Exact       — algorithms/bbo.py            (Standard CPU BBO)
  2. IMPROVE-1       — algorithms/bbo_improve_1.py  (Improved version 1)

Parallelism:
  - Trial-level: multiple trials run simultaneously via multiprocessing.Pool
  - Worker count auto-selected: 75% cores if ≤ 8, else 50% (see hardware.py)

Output files:
  results/csv/bbo_exact_cec2017.csv
  results/csv/improve_1_cec2017.csv

Each CSV: Function, Best, Mean, Worst, Std, iter_0 … iter_N columns

Usage:
  python experiments/run_cec2017.py                # full run (30 trials, parallel)
  python experiments/run_cec2017.py --smoke        # 1 function, 3 trials
  python experiments/run_cec2017.py --no-parallel  # serial fallback
"""

import sys
import os
import warnings
import argparse
import contextlib
import io
import numpy as np
import pandas as pd
import multiprocessing

# Suppress warnings that come from opfunu's internals (not our code)
warnings.filterwarnings("ignore", category=UserWarning,   module="opfunu")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="opfunu")

import opfunu

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.bbo import BBO as BBOExact
from algorithms.bbo_improve_1 import NADBBO as Improve1
from utils.hardware     import get_worker_count, print_hardware_summary

#  Parameters

DIMS     = 10
POP_SIZE = 30
MAX_ITER = 500
TRIALS   = 30
LB, UB   = -100, 100

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "csv")

#  CEC-2017 function discovery                                         #

def get_cec2017_classes():
    """Return list of CEC-2017 function classes, sorted by name."""
    all_funcs = opfunu.get_all_cec_functions()
    cec2017   = [cls for cls in all_funcs if "2017" in cls.__name__]
    return sorted(cec2017, key=lambda c: c.__name__)

#  Worker function — runs ONE trial for ONE function                   #
# Designed to be picklable (top-level function) for multiprocessing.  #

def _run_one_trial(args):
    """
    args = (algo_tag, func_class_name, trial_seed)
    Returns (trial_seed, best_score, convergence_curve)
    """
    algo_tag, func_class_name, trial_seed = args

    # Re-import inside worker (spawn context has fresh state)
    import sys, os, warnings
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))

    warnings.filterwarnings("ignore", category=UserWarning,   module="opfunu")
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="opfunu")

    import numpy as np
    import opfunu
    import contextlib, io

    from algorithms.bbo import BBO as BBOExact
    from algorithms.bbo_improve_1 import NADBBO as Improve1

    np.random.seed(trial_seed)

    # Reconstruct CEC-2017 evaluator by name
    all_funcs = opfunu.get_all_cec_functions()
    cls_map   = {c.__name__: c for c in all_funcs if "2017" in c.__name__}
    cls       = cls_map[func_class_name]
    evaluator = cls(ndim=DIMS).evaluate

    # Build optimizer
    if algo_tag == "BBO-Exact":
        optimizer = BBOExact(
            obj_func=evaluator, dims=DIMS, pop_size=POP_SIZE,
            max_iter=MAX_ITER, lb=LB, ub=UB
        )
    else:  # IMPROVE-1
        optimizer = Improve1(
            obj_func=evaluator, dims=DIMS, pop_size=POP_SIZE,
            max_iter=MAX_ITER, lb=LB, ub=UB
        )

    with contextlib.redirect_stdout(io.StringIO()):
        _, best_score, curve = optimizer.run()

    # NaN guard — replace any NaN best with a large finite value
    if not np.isfinite(best_score):
        best_score = 1e18
    curve = [v if np.isfinite(v) else 1e18 for v in curve]

    return trial_seed, best_score, curve


#  Run one algorithm on all functions

def _load_checkpoint(csv_path):
    """Load existing CSV and return set of completed function names."""
    if not os.path.exists(csv_path):
        return pd.DataFrame(), set()
    try:
        df = pd.read_csv(csv_path)
        # Only keep rows that have real results (not just smoke test NaN rows)
        done = set(df["Function"].tolist()) if "Function" in df.columns else set()
        return df, done
    except Exception:
        return pd.DataFrame(), set()

def run_algorithm(algo_tag, cec_classes, n_trials, n_workers, parallel, out_path):
    """
    Run algo_tag on every function in cec_classes for n_trials.
    Skips functions already present in out_path (checkpoint resume).
    Returns a DataFrame with stats + mean convergence curve per row.
    """
    existing_df, done_funcs = _load_checkpoint(out_path)

    # If all functions already done, skip entirely
    all_func_names = {cls.__name__ for cls in cec_classes}
    if all_func_names <= done_funcs:
        print(f"  [CHECKPOINT] All {len(cec_classes)} functions already complete — skipping.")
        return existing_df

    rows = list(existing_df.to_dict("records")) if not existing_df.empty else []

    for cls in cec_classes:
        func_name = cls.__name__

        if func_name in done_funcs:
            print(f"  [CHECKPOINT] {func_name} — already done, skipping.")
            continue
        func_name = cls.__name__
        print(f"\n[{algo_tag}] {func_name}", flush=True)

        # Build list of (algo_tag, func_name, seed) for each trial
        work_items = [
            (algo_tag, func_name, trial)
            for trial in range(n_trials)
        ]

        if parallel and n_workers > 1:
            # Spawn context — required for CUDA safety
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                results = pool.map(_run_one_trial, work_items)
        else:
            results = [_run_one_trial(item) for item in work_items]

        trial_scores = []
        trial_curves = []
        for seed, score, curve in sorted(results, key=lambda x: x[0]):
            trial_scores.append(score)
            trial_curves.append(curve)
            print(f"  Trial {seed+1:>2}/{n_trials}: {score:.6e}", flush=True)

        scores = np.array(trial_scores)
        row = {
            "Function": func_name,
            "Best":     float(np.min(scores)),
            "Mean":     float(np.mean(scores)),
            "Worst":    float(np.max(scores)),
            "Std":      float(np.std(scores)),
        }

        mean_curve = np.mean(trial_curves, axis=0)
        for idx, val in enumerate(mean_curve):
            row[f"iter_{idx}"] = val

        rows.append(row)
        print(f"  → Best={row['Best']:.4e}  Mean={row['Mean']:.4e}  Std={row['Std']:.4e}",
              flush=True)

        # Incremental save after each function — power-cut safe
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)

#  Entry point          

def main():
    parser = argparse.ArgumentParser(description="CEC-2017 BBO Experiment Runner")
    parser.add_argument("--smoke",       action="store_true",
                        help="Smoke test: 1 function, 3 trials only")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable multiprocessing (serial mode)")
    parser.add_argument("--algo",        type=str, choices=["exact", "improve-1", "all"], default="all",
                        help="Which algorithm to run: 'exact', 'improve-1', or 'all' (default)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Hardware detection
    print_hardware_summary()
    n_workers, total_cores, pct = get_worker_count()

    parallel = not args.no_parallel

    cec_classes = get_cec2017_classes()
    n_trials    = TRIALS

    if args.smoke:
        cec_classes = cec_classes[:1]
        n_trials    = 3
        print("=== SMOKE TEST MODE: 1 function, 3 trials ===")

    print(f"\nCEC-2017 functions : {len(cec_classes)}")
    print(f"Trials / function  : {n_trials}")
    print(f"Iterations         : {MAX_ITER}")
    print(f"Population size    : {POP_SIZE}")
    print(f"Dimensions         : {DIMS}")
    print(f"Parallel workers   : {n_workers if parallel else 1} (parallel={parallel})")
    print("=" * 60)

    ALL_ALGORITHMS = [
        ("BBO-Exact",      "bbo_exact_cec2017.csv"),
        ("IMPROVE-1",      "improve_1_cec2017.csv"),
    ]

    if args.algo == "exact":
        algorithms_to_run = [ALL_ALGORITHMS[0]]
    elif args.algo == "improve-1":
        algorithms_to_run = [ALL_ALGORITHMS[1]]
    else:
        algorithms_to_run = ALL_ALGORITHMS

    for algo_tag, csv_name in algorithms_to_run:
        out_path = os.path.join(OUTPUT_DIR, csv_name)
        print(f"\n{'='*60}")
        print(f" Algorithm : {algo_tag}")
        print(f" Output    : {out_path}")
        print(f"{'='*60}")

        df = run_algorithm(
            algo_tag, cec_classes, n_trials,
            n_workers, parallel,
            out_path
        )

        print(f"\n  Saved → {out_path} ({len(df)} functions)")

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
