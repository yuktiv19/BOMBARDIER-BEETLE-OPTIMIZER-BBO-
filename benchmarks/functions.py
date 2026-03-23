import numpy as np

def sphere(x):
    """Goal: 0.0 at [0,0,0...]. Tests basic convergence."""
    return np.sum(x**2)

def rastrigin(x):
    """Goal: 0.0. Tests the beetle's ability to escape 'local optima' (traps)."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """Goal: 0.0. Tests how well the beetle follows narrow, curved paths."""
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)