import inspect
import opfunu
import numpy as np
from src.bbo import BombardierBeetleOptimizer


def _instantiate_cec_class(cls, ndim):
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        for key in ("ndim", "n_dim", "dimension", "dims", "dim"):
            if key in params:
                return cls(**{key: ndim})
        if len(params) == 1:
            return cls(ndim)
        return cls()
    except Exception:
        try:
            return cls(ndim)
        except Exception:
            return None


def smoke_test():
    funcs = opfunu.get_cec_based_functions()
    cec2017 = []
    for item in funcs:
        # opfunu may return either (name, cls) or just cls entries
        if isinstance(item, tuple) and len(item) >= 2:
            name, cls = item[0], item[1]
        else:
            cls = item
            name = getattr(cls, '__name__', str(cls))
        module_name = getattr(cls, "__module__", "")
        if "2017" in module_name or "2017" in name or "cec2017" in module_name:
            cec2017.append((name, cls))

    if not cec2017:
        print("No CEC2017 functions found. Exiting smoke test.")
        return

    name, cls = cec2017[0]
    print(f"Selected function: {name} -> {cls}")

    DIMENSIONS = 10
    ITERATIONS = 50
    POP_SIZE = 20
    TRIALS = 3

    problem_inst = _instantiate_cec_class(cls, DIMENSIONS)
    if problem_inst is None:
        print(f"Could not instantiate {cls}.")
        return

    evaluator = getattr(problem_inst, 'evaluate', None)
    if evaluator is None and callable(problem_inst):
        evaluator = problem_inst
    if evaluator is None:
        print("No evaluator found on problem instance.")
        return

    for t in range(TRIALS):
        np.random.seed(t)
        bbo = BombardierBeetleOptimizer(
            obj_func=evaluator,
            dims=DIMENSIONS,
            iterations=ITERATIONS,
            pop_size=POP_SIZE,
            lb=-100,
            ub=100,
        )
        _, best_score, _ = bbo.run()
        print(f"Smoke Trial {t+1}/{TRIALS}: Best Score = {best_score:.4e}")


if __name__ == '__main__':
    smoke_test()
