"""
BBO Live Demo
=============
Single-command live showcase for the teacher.

Runs all three BBO variants on 5 representative CEC-2017 functions with
real-time animated convergence curves that draw themselves on screen as
iterations execute.

After all runs complete, shows:
  - Final convergence curves (all 3 algos per function)
  - Mean rank bar chart summary
  - Prints comparison table to terminal

Usage:
  python demo.py              # full demo (5 trials, 300 iters)
  python demo.py --fast       # quick check (1 trial, 50 iters)
"""

import sys
import os
import argparse
import contextlib
import io
import numpy as np
import opfunu
import matplotlib
matplotlib.use("TkAgg")       #backend for live animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from algorithms.bbo import BBO as BBOExact
from algorithms.bbo_improve_1 import NADBBO as Improve1

#  Demo parameters 
DEMO_FUNCS = ["F12017", "F42017", "F92017", "F152017", "F212017"]

ALGO_STYLES = [
    ("BBO-Exact",      "steelblue", "-",  2.0),
    ("IMPROVE-1",      "crimson",   "-",  2.0),
]

#  CEC-2017 helper
def get_cec_func(name, ndim):
    all_funcs = opfunu.get_all_cec_functions()
    cls_map   = {c.__name__: c for c in all_funcs if "2017" in c.__name__}
    if name not in cls_map:
        raise ValueError(f"Function {name} not found in opfunu CEC-2017 set.")
    return cls_map[name](ndim=ndim).evaluate

#  Algorithm factory
def make_algo(algo_name, evaluator, dims, pop_size, max_iter):
    lb, ub = -100, 100
    if algo_name == "BBO-Exact":
        return BBOExact(evaluator, dims=dims, pop_size=pop_size,
                        max_iter=max_iter, lb=lb, ub=ub)
    else:
        return Improve1(evaluator, dims=dims, pop_size=pop_size,
                      max_iter=max_iter, lb=lb, ub=ub)

#  Run one algorithm, yielding convergence curve step-by-step
def run_with_live_update(algo_name, evaluator, dims, pop_size, max_iter, ax, line, x_data, y_data, fig):
    """
    Run an algorithm and update the matplotlib line after every iteration.
    Returns the final convergence curve.
    """
    algo = make_algo(algo_name, evaluator, dims, pop_size, max_iter)

    for t in range(max_iter):
        # Run one iteration
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(algo.pop_size):
                if algo_name == "IMPROVE-1":
                    p = algo._explore_prob(t)
                    if np.random.rand() < p:
                        x_new = algo._phase1_defense(i, t)
                    else:
                        x_new = algo._phase2_escape(i, t)
                    algo._accept(i, x_new)
                else:
                    x1 = algo._phase1_defense(i, t)
                    algo._accept(i, x1)
                    x2 = algo._phase2_escape(i, t)
                    algo._accept(i, x2)

        x_data.append(t)
        y_data.append(algo.g_best_score)

        # Update plot every 5 iterations for speed
        if t % 5 == 0 or t == max_iter - 1:
            line.set_data(x_data, y_data)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

    return y_data[:]

#  Multi-trial averaged run (no live animation, just returns curve)
def run_averaged(algo_name, evaluator_factory, dims, pop_size, max_iter, n_trials):
    curves = []
    scores = []
    for trial in range(n_trials):
        np.random.seed(trial)
        evaluator = evaluator_factory()
        algo = make_algo(algo_name, evaluator, dims, pop_size, max_iter)
        with contextlib.redirect_stdout(io.StringIO()):
            _, best, curve = algo.run()
        curves.append(curve)
        scores.append(best)
    mean_curve = np.mean(curves, axis=0)
    return mean_curve, np.min(scores), np.mean(scores), np.std(scores)

#  Main demo 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Quick mode: 1 trial, 50 iterations")
    args = parser.parse_args()

    DIMS     = 10
    POP_SIZE = 30
    TRIALS   = 1 if args.fast else 5
    MAX_ITER = 50 if args.fast else 300

    n_funcs = len(DEMO_FUNCS)

    print("=" * 60)
    print("  BBO Live Demo — Bombardier Beetle Optimizer")
    print("=" * 60)
    print(f"  Functions  : {', '.join(DEMO_FUNCS)}")
    print(f"  Algorithms : {', '.join(s[0] for s in ALGO_STYLES)}")
    print(f"  Trials     : {TRIALS}  |  Iterations: {MAX_ITER}  |  Pop: {POP_SIZE}")
    print("=" * 60)

    #  Figure layout: 2 rows * 3 cols for functions + 1 summary slot
    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("BBO Live Optimization — Convergence Curves (Real-Time)", fontsize=14, fontweight="bold")

    gs    = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes  = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(n_funcs)]
    ax_summary = fig.add_subplot(gs[1, 2])   # bottom-right for bar chart

    for ax, fname in zip(axes, DEMO_FUNCS):
        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Best Fitness", fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    ax_summary.set_title("Mean Rank (lower=better)", fontsize=10, fontweight="bold")
    ax_summary.set_visible(False)  # shown after all runs

    plt.tight_layout()
    plt.pause(0.1)

    #  Run all algorithms, averaged across TRIALS
    all_results = {fname: {} for fname in DEMO_FUNCS}

    for fname in DEMO_FUNCS:
        ax_idx = DEMO_FUNCS.index(fname)
        ax     = axes[ax_idx]
        print(f"\n>>> {fname}")

        for algo_name, color, ls, lw in ALGO_STYLES:
            print(f"  Running {algo_name}...", end=" ", flush=True)

            def evaluator_factory(fn=fname):
                return get_cec_func(fn, DIMS)

            mean_curve, best, mean, std = run_averaged(
                algo_name, evaluator_factory, DIMS, POP_SIZE, MAX_ITER, TRIALS
            )

            # Plot the mean convergence curve
            iters = np.arange(len(mean_curve))
            ax.plot(iters, mean_curve, label=algo_name, color=color, linestyle=ls, linewidth=lw)
            ax.relim()
            ax.autoscale_view()
            if algo_name == ALGO_STYLES[0][0]:
                ax.legend(fontsize=6)

            all_results[fname][algo_name] = {"curve": mean_curve, "best": best, "mean": mean, "std": std}

            print(f"Mean={mean:.3e}  Best={best:.3e}")
            fig.canvas.draw()
            fig.canvas.flush_events()

    #  Summary bar chart — mean rank
    algo_names  = [s[0] for s in ALGO_STYLES]
    algo_colors = {s[0]: s[1] for s in ALGO_STYLES}

    rank_totals = {a: [] for a in algo_names}
    for fname in DEMO_FUNCS:
        scores = {a: all_results[fname][a]["mean"] for a in algo_names}
        sorted_algos = sorted(scores, key=scores.get)
        for a in algo_names:
            rank_totals[a].append(sorted_algos.index(a) + 1)

    mean_ranks = {a: np.mean(rank_totals[a]) for a in algo_names}

    ax_summary.set_visible(True)
    bars = ax_summary.bar(
        algo_names,
        [mean_ranks[a] for a in algo_names],
        color=[algo_colors[a] for a in algo_names],
        edgecolor="white", width=0.5
    )
    for bar, a in zip(bars, algo_names):
        ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{mean_ranks[a]:.2f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
    ax_summary.set_ylim(0, len(algo_names) + 0.5)
    ax_summary.set_ylabel("Mean Rank", fontsize=9)
    ax_summary.set_title("Mean Rank Summary\n(lower = better)", fontsize=9, fontweight="bold")
    ax_summary.grid(axis="y", linestyle="--", alpha=0.4)

    fig.canvas.draw()
    fig.canvas.flush_events()

    #  Terminal summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Function':<12}" + "".join(f"  {a:>18}" for a in algo_names) + f"  {'Winner':>14}"
    print(header)
    print("-" * 70)

    win_counts = {a: 0 for a in algo_names}
    for fname in DEMO_FUNCS:
        means  = {a: all_results[fname][a]["mean"] for a in algo_names}
        winner = min(means, key=means.get)
        win_counts[winner] += 1
        row = f"{fname:<12}" + "".join(f"  {means[a]:>18.4e}" for a in algo_names)
        row += f"  {winner + ' ✓':>14}"
        print(row)

    print("-" * 70)
    rank_row = f"{'MEAN RANK':<12}" + "".join(f"  {mean_ranks[a]:>18.3f}" for a in algo_names)
    print(rank_row)
    print("=" * 70)
    print(f"  Win counts:")
    for a in algo_names:
        print(f"    {a:<20}: {win_counts[a]}/{len(DEMO_FUNCS)} functions")

    best_algo = min(mean_ranks, key=mean_ranks.get)
    print(f"\n  Best algorithm: {best_algo}  (mean rank = {mean_ranks[best_algo]:.3f})")
    print("=" * 70)

    # Save final figure
    os.makedirs("results/plots", exist_ok=True)
    fig.savefig("results/plots/demo_output.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved → results/plots/demo_output.png")
    print("\n  [Close the plot window to exit]")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
