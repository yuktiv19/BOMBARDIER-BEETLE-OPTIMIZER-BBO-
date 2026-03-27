"""
Comparison Bar Chart Plotter
============================
Reads the three result CSVs and produces:
  - Grouped bar chart of Mean scores per function (all 3 algorithms)
  - Mean Rank bar chart (like paper's Fig. 10)

Usage:
  python analysis/plot_comparison.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Config    

CSV_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "csv")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

ALGO_FILES = [
    ("BBO-Simplified", "bbo_simplified_cec2017.csv", "dimgrey"),
    ("BBO-Exact",      "bbo_exact_cec2017.csv",      "steelblue"),
    ("NAD-BBO",        "nad_bbo_cec2017.csv",        "crimson"),
]

# Load stats

def load_stats(csv_path):
    """Return DataFrame with Function, Best, Mean, Worst, Std columns."""
    df = pd.read_csv(csv_path)
    stat_cols = ["Function", "Best", "Mean", "Worst", "Std"]
    return df[stat_cols].copy()

def load_all():
    frames = {}
    for name, fname, _ in ALGO_FILES:
        path = os.path.join(CSV_DIR, fname)
        if not os.path.exists(path):
            print(f"Missing: {fname} — skipping")
            continue
        frames[name] = load_stats(path)
    return frames

# Mean scores grouped bar chart

def plot_mean_comparison(frames):
    algos = list(frames.keys())
    colors = {name: col for name, _, col in ALGO_FILES}

    # Align on common functions
    common_funcs = sorted(
        set.intersection(*[set(frames[a]["Function"]) for a in algos])
    )
    n = len(common_funcs)
    x = np.arange(n)
    width = 0.25
    offsets = np.linspace(-(len(algos)-1)/2, (len(algos)-1)/2, len(algos)) * width

    fig, ax = plt.subplots(figsize=(max(14, n * 0.7), 6))

    for idx, algo in enumerate(algos):
        df = frames[algo].set_index("Function")
        means = [df.loc[f, "Mean"] for f in common_funcs]
        ax.bar(x + offsets[idx], means, width, label=algo,
               color=colors[algo], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(common_funcs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Fitness (log scale)", fontsize=11)
    ax.set_title("Mean Fitness Comparison — BBO-Simplified vs BBO-Exact vs NAD-BBO",
                 fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "mean_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

# Mean Rank bar chart (like paper's Fig. 10)

def compute_mean_ranks(frames):
    """
    For each function, rank all algorithms by Mean score (lower = better rank 1).
    Returns dict: algo_name → mean rank across all functions.
    """
    algos = list(frames.keys())
    common_funcs = sorted(
        set.intersection(*[set(frames[a]["Function"]) for a in algos])
    )

    rank_records = []
    for func in common_funcs:
        scores = {}
        for algo in algos:
            df = frames[algo].set_index("Function")
            scores[algo] = df.loc[func, "Mean"]

        # Rank: 1 = best (lowest mean)
        sorted_algos = sorted(scores, key=scores.get)
        ranks = {algo: sorted_algos.index(algo) + 1 for algo in algos}
        rank_records.append(ranks)

    mean_ranks = {algo: np.mean([r[algo] for r in rank_records]) for algo in algos}
    return mean_ranks


def plot_mean_rank(frames):
    mean_ranks = compute_mean_ranks(frames)
    colors     = {name: col for name, _, col in ALGO_FILES}

    algos = list(mean_ranks.keys())
    ranks = [mean_ranks[a] for a in algos]
    bar_colors = [colors.get(a, "grey") for a in algos]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(algos, ranks, color=bar_colors, edgecolor="white", width=0.5)

    # Label each bar with its rank value
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rank:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Total Mean Rank (lower = better)", fontsize=11)
    ax.set_title("Mean Rank — BBO-Simplified vs BBO-Exact vs NAD-BBO",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(ranks) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "mean_rank_bar.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")
    return mean_ranks

# Main      

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    frames = load_all()

    if not frames:
        print("No result CSVs found. Run experiments/run_cec2017.py first.")
        return

    print(f"Loaded results for: {list(frames.keys())}")

    print("Generating mean comparison bar chart...")
    plot_mean_comparison(frames)

    print("Generating mean rank bar chart...")
    mean_ranks = plot_mean_rank(frames)
    for algo, rank in mean_ranks.items():
        print(f"  {algo:20s}: Mean Rank = {rank:.3f}")

    print("Done.")

if __name__ == "__main__":
    main()
