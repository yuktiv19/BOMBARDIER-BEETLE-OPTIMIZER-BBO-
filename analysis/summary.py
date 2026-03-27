"""
Results Summary
===============
Reads the three result CSVs and prints a formatted terminal table showing:
  - Function-by-function winner
  - Overall win counts
  - Total mean rank per algorithm
  - Improvement narrative

Usage:
  python analysis/summary.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#  Config

CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "csv")

ALGO_FILES = [
    ("BBO-Simplified", "bbo_simplified_cec2017.csv"),
    ("BBO-Exact",      "bbo_exact_cec2017.csv"),
    ("NAD-BBO",        "nad_bbo_cec2017.csv"),
]

#  Load    

def load_all():
    frames = {}
    for name, fname in ALGO_FILES:
        path = os.path.join(CSV_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            stat_cols = ["Function", "Best", "Mean", "Worst", "Std"]
            frames[name] = df[stat_cols].set_index("Function")
    return frames

#  Rank computation

def compute_ranks(frames):
    algos = list(frames.keys())
    common_funcs = sorted(
        set.intersection(*[set(frames[a].index) for a in algos])
    )

    rank_records = {}
    for func in common_funcs:
        scores = {a: frames[a].loc[func, "Mean"] for a in algos}
        sorted_algos = sorted(scores, key=scores.get)
        rank_records[func] = {a: sorted_algos.index(a) + 1 for a in algos}

    return rank_records, common_funcs, algos

#  Print table

def print_summary(frames):
    if not frames:
        print("No result CSVs found. Run experiments/run_cec2017.py first.")
        return

    rank_records, common_funcs, algos = compute_ranks(frames)

    # Column widths
    fw = 12   # function name
    vw = 14   # value columns
    ww = 14   # winner column

    sep = "+" + "-"*(fw+2) + "+" + ("+"+"-"*(vw+2)) * len(algos) + "+" + "-"*(ww+2) + "+"
    hdr_fmt  = "| {:<{fw}} |" + " {:>{vw}} |" * len(algos) + " {:<{ww}} |"
    row_fmt  = "| {:<{fw}} |" + " {:>{vw}} |" * len(algos) + " {:<{ww}} |"

    print("\n" + "=" * (fw + len(algos)*(vw+3) + ww + 10))
    print("  BBO Results Summary — Mean Fitness per CEC-2017 Function")
    print("=" * (fw + len(algos)*(vw+3) + ww + 10))
    print(sep)
    header_vals = [a[:vw] for a in algos]
    print(hdr_fmt.format("Function", *header_vals, "Winner", fw=fw, vw=vw, ww=ww))
    print(sep)

    win_count = {a: 0 for a in algos}

    for func in common_funcs:
        means = {a: frames[a].loc[func, "Mean"] for a in algos}
        best_algo = min(means, key=means.get)

        # Check for ties (within 1% of best)
        best_val = means[best_algo]
        tied = [a for a in algos if abs(means[a] - best_val) / (abs(best_val) + 1e-300) < 0.01]

        if len(tied) > 1:
            winner_str = "Tie"
        else:
            winner_str = best_algo + " ✓"
            win_count[best_algo] += 1

        mean_strs = [f"{means[a]:.3e}" for a in algos]
        print(row_fmt.format(func, *mean_strs, winner_str, fw=fw, vw=vw, ww=ww))

    print(sep)

    # Mean rank row
    mean_ranks = {}
    for algo in algos:
        ranks = [rank_records[f][algo] for f in common_funcs]
        mean_ranks[algo] = np.mean(ranks)

    rank_strs = [f"rank={mean_ranks[a]:.2f}" for a in algos]
    print(row_fmt.format("MEAN RANK", *rank_strs, "", fw=fw, vw=vw, ww=ww))
    print(sep)

    # Win counts
    print()
    print("  Win counts (by lowest mean fitness):")
    for algo in algos:
        print(f"    {algo:20s}: {win_count[algo]:>3} / {len(common_funcs)} functions")

    # Mean ranks
    print()
    print("  Total Mean Rank (lower = better):")
    for algo in sorted(algos, key=lambda a: mean_ranks[a]):
        print(f"    {algo:20s}: {mean_ranks[algo]:.3f}")

    # Narrative
    best_algo = min(mean_ranks, key=mean_ranks.get)
    print()
    print("  Progression:")
    for algo in algos:
        tag = " ← BEST" if algo == best_algo else ""
        print(f"    {algo:20s}  mean rank = {mean_ranks[algo]:.3f}{tag}")

    if "NAD-BBO" in mean_ranks and "BBO-Exact" in mean_ranks:
        improvement = (mean_ranks["BBO-Exact"] - mean_ranks["NAD-BBO"]) / mean_ranks["BBO-Exact"] * 100
        direction = "improvement" if improvement > 0 else "regression"
        print(f"\n  NAD-BBO vs BBO-Exact: {abs(improvement):.1f}% mean rank {direction}")

    if "NAD-BBO" in mean_ranks and "BBO-Simplified" in mean_ranks:
        improvement = (mean_ranks["BBO-Simplified"] - mean_ranks["NAD-BBO"]) / mean_ranks["BBO-Simplified"] * 100
        direction = "improvement" if improvement > 0 else "regression"
        print(f"  NAD-BBO vs BBO-Simplified: {abs(improvement):.1f}% mean rank {direction}")

    print()

def main():
    frames = load_all()
    print_summary(frames)

if __name__ == "__main__":
    main()
