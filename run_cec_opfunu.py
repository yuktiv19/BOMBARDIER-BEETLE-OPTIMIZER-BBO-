import opfunu
import inspect
import numpy as np
import pandas as pd
from src.bbo import BombardierBeetleOptimizer


def _instantiate_cec_class(cls, ndim):
    """Try to instantiate a CEC function class given a candidate signature."""
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        # prefer keyword arg names that indicate dimension
        for key in ("ndim", "n_dim", "dimension", "dims", "dim"):
            if key in params:
                return cls(**{key: ndim})
        # fallback: try positional if first param isn't 'self'
        if len(params) == 1:
            return cls(ndim)
        # try no-arg construction
        return cls()
    except Exception:
        # give one last try via positional
        try:
            return cls(ndim)
        except Exception:
            return None


def run_cec_experiment():
    # 1. Setup Parameters (Matching Page 15 of the paper)
    DIMENSIONS = 10
    ITERATIONS = 500
    POP_SIZE = 30
    TRIALS = 30  # Standard for statistical significance

    # 2. Discover CEC-based functions from opfunu and select 2017 set
    funcs = opfunu.get_cec_based_functions()

    cec2017 = []
    for it in funcs:
        try:
            if isinstance(it, tuple) and len(it) == 2:
                name, cls = it
            else:
                cls = it
                name = getattr(cls, "__name__", str(cls))

            module_name = getattr(cls, "__module__", "")
            if "2017" in module_name.lower() or "2017" in name.lower() or "cec2017" in module_name.lower():
                cec2017.append((name, cls))
        except Exception:
            continue

    if not cec2017:
        print("No CEC2017 functions detected in opfunu. Aborting.")
        return

    final_stats = []

    for name, cls in cec2017:
        print(f"\n>>> Starting CEC 2017 {name} ({cls})...")
        problem_inst = _instantiate_cec_class(cls, DIMENSIONS)
        if problem_inst is None:
            print(f"  Skipping {name}: could not instantiate class {cls}")
            continue

        # derive evaluator
        if hasattr(problem_inst, "evaluate"):
            evaluator = problem_inst.evaluate
        elif callable(problem_inst):
            evaluator = problem_inst
        else:
            print(f"  Skipping {name}: no callable evaluator found on instance")
            continue

        trial_scores = []
        for t in range(TRIALS):
            np.random.seed(t)  # reproducible

            bbo = BombardierBeetleOptimizer(
                obj_func=evaluator,
                dims=DIMENSIONS,
                iterations=ITERATIONS,
                pop_size=POP_SIZE,
                lb=-100,
                ub=100,
            )

            _, best_score, _ = bbo.run()
            trial_scores.append(best_score)
            print(f"  Trial {t+1}: Best Score = {best_score:.4e}")

        final_stats.append({
            "CEC_Function": name,
            "Best_Score": np.min(trial_scores),
            "Mean_Score": np.mean(trial_scores),
            "Std_Dev": np.std(trial_scores),
        })

    # 3. Save to CSV for your Paper
    df = pd.DataFrame(final_stats)
    df.to_csv("research_results_cec.csv", index=False)
    print("\n" + "=" * 40)
    print("EXPERIMENT COMPLETE: Results saved to 'research_results_cec.csv'")
    print("=" * 40)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_cec_experiment()