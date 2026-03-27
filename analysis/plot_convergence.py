"""
Convergence Curve Plotter
=========================
Reads mean convergence data from the three CSV result files and produces:
  - One PNG per CEC-2017 function showing all 3 algorithms on the same axes
  - One grid PNG showing all functions together (like paper's convergence figures)

Usage:
  python analysis/plot_convergence.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving to file
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#  Config    

CSV_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "csv")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

DATASETS = [
    ("BBO-Simplified", "bbo_simplified_cec2017.csv", "dimgrey",  "--"),
    ("BBO-Exact",      "bbo_exact_cec2017.csv",      "steelblue", "-"),
    ("NAD-BBO",        "nad_bbo_cec2017.csv",        "crimson",   "-"),
]

#  Helpers   

def load_convergence(csv_path):
    """
    Return a dict: function_name → 1-D numpy array of mean convergence values.
    Convergence columns are named iter_0, iter_1, ...
    """
    df = pd.read_csv(csv_path)
    result = {}
    iter_cols = [c for c in df.columns if c.startswith("iter_")]
    for _, row in df.iterrows():
        result[row["Function"]] = row[iter_cols].values.astype(float)
    return result


def check_csvs_exist():
    missing = []
    for _, fname, _, _ in DATASETS:
        path = os.path.join(CSV_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
    return missing


#  Per-function convergence plot

def plot_function(func_name, data_per_algo, out_path):
    """Plot convergence curves for one function, save to out_path."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, curve, color, ls in data_per_algo:
        ax.plot(curve, label=label, color=color, linestyle=ls, linewidth=1.8)

    ax.set_title(func_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


#  Grid convergence plot (all functions)

def plot_grid(func_names, all_data, out_path):
    """Plot all functions in a grid layout, like the paper's figures."""
    n = len(func_names)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2))
    axes = axes.flatten()

    for idx, func_name in enumerate(func_names):
        ax = axes[idx]
        for label, data_dict, color, ls in all_data:
            curve = data_dict.get(func_name)
            if curve is not None:
                ax.plot(curve, label=label, color=color, linestyle=ls, linewidth=1.5)
        ax.set_title(func_name, fontsize=9, fontweight="bold")
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Convergence Curves — BBO-Simplified vs BBO-Exact vs NAD-BBO",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid saved → {out_path}")


#  Main

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    missing = check_csvs_exist()
    if missing:
        print("ERROR: Missing result CSVs — run experiments/run_cec2017.py first:")
        for f in missing:
            print(f"  {f}")
        return

    # Load all data
    all_data = []
    for label, fname, color, ls in DATASETS:
        path = os.path.join(CSV_DIR, fname)
        conv = load_convergence(path)
        all_data.append((label, conv, color, ls))
        print(f"Loaded {len(conv)} functions from {fname}")

    # Common function names (intersection across all datasets)
    func_names = sorted(
        set(all_data[0][1].keys())
        .intersection(all_data[1][1].keys())
        .intersection(all_data[2][1].keys())
    )
    print(f"Plotting convergence for {len(func_names)} functions...")

    # Individual plots
    for func_name in func_names:
        data_per_algo = [
            (label, data_dict[func_name], color, ls)
            for label, data_dict, color, ls in all_data
            if func_name in data_dict
        ]
        out_path = os.path.join(PLOT_DIR, f"convergence_{func_name}.png")
        plot_function(func_name, data_per_algo, out_path)

    print(f"  Individual plots saved to {PLOT_DIR}/convergence_*.png")

    # Grid plot
    grid_path = os.path.join(PLOT_DIR, "convergence_grid.png")
    plot_grid(func_names, all_data, grid_path)

    print("Done.")

if __name__ == "__main__":
    main()
