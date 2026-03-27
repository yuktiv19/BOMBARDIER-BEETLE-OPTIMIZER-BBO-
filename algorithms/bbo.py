"""
Bombardier Beetle Optimizer — Exact Implementation
===================================================
Faithful implementation of the algorithm described in:
  Shehadeh et al., "Bombardier Beetle Optimizer: A Novel Bio-Inspired
  Algorithm for Global Optimization"

Algorithm structure (Algorithm 1 from paper):
  Phase 1 — Defense mechanism (Exploration):
      Circle Intersection Area (CIA)  [Eq. 30]
      Chemical Reaction (CR)
      Chaos spray                      [Eq. 33]
      Position update                  [Eq. 31]
      Greedy selection                 [Eq. 32]

  Phase 2 — Escape via insect lift (Exploitation):
      Newton's lift equation           [Eq. 34]
      Position update
      Greedy selection                 [Eq. 32]
"""

import numpy as np


class BBO:
    """Exact Bombardier Beetle Optimizer from Shehadeh et al."""

    def __init__(self, obj_func, dims, pop_size=30, max_iter=500, lb=-100, ub=100):
        self.obj_func = obj_func
        self.dims     = dims
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb       = np.full(dims, lb, dtype=float)
        self.ub       = np.full(dims, ub, dtype=float)

        # Population — Eq. 29
        self.population = self._init_population()
        self.fitness    = np.array([obj_func(ind) for ind in self.population])

        # Global best
        best_idx           = np.argmin(self.fitness)
        self.g_best_pos    = self.population[best_idx].copy()
        self.g_best_score  = self.fitness[best_idx]

        # Each beetle has its own chaos variable (sinusoidal map seed)
        self.chaos_state = np.random.uniform(0.1, 0.9, pop_size)


    #  Initialisation           
    def _init_population(self):
        """Eq. 29: x_id = lb_d + rand(0,1) * (ub_d - lb_d)"""
        r = np.random.uniform(0, 1, (self.pop_size, self.dims))
        return self.lb + r * (self.ub - self.lb)

    #  CIA — Circle Intersection Area  [Eq. 30]
    def _cia(self, R, r, d):
        """
        Compute the intersection area of two circles.
          R — radius of beetle's circle
          r — radius of predator's circle
          d — distance between their centres
        Three cases from the paper:
          1. d >= R+r  → no intersection → CIA = 0
          2. d <= |R-r| → one fully inside → CIA = π·min(R,r)²
          3. otherwise  → partial overlap  → Eq. 30
        """
        if d >= R + r:
            return 0.0
        if d <= abs(R - r):
            return np.pi * min(R, r) ** 2

        # Eq. 30 — partial intersection
        # Guard against floating-point domain errors in arccos
        arg1 = np.clip((d**2 + r**2 - R**2) / (2 * d * r), -1.0, 1.0)
        arg2 = np.clip((d**2 + R**2 - r**2) / (2 * d * R), -1.0, 1.0)

        term1 = r**2 * np.arccos(arg1)
        term2 = R**2 * np.arccos(arg2)
        # Heron-style term — must stay non-negative; clamp for safety
        heron = (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)
        term3 = 0.5 * np.sqrt(max(heron, 0.0))

        return term1 + term2 - term3

    #  Chaos map — sinusoidal
    def _chaos_map(self, x):
        """Sinusoidal chaos map: x_{n+1} = sin(π · x_n)"""
        val = np.sin(np.pi * x)
        # Keep strictly in (0,1) to avoid degenerate spray values
        return np.clip(abs(val), 1e-6, 1.0 - 1e-6)

    #  Spray — Eq. 33    

    def _spray(self, chaos_val, t):
        """
        Eq. 33: Spray = chaos * 2.7^(100^(t / Max_Iter))
        Controls how far the new position moves from the beetle.
        Grows with t → steps shrink → exploitation tightens.
        """
        ratio    = t / max(self.max_iter - 1, 1)
        exponent = 100 ** ratio
        # Cap exponent to avoid overflow (2.7^1e8 → inf)
        exponent = min(exponent, 700)
        spray    = chaos_val * (2.7 ** exponent)
        # Spray must be > 0; if chaos_val → 0 use safe floor
        return max(spray, 1e-10)

    #  Phase 1 — Defense / Exploration  [Eq. 31, 33]
    def _phase1_defense(self, i, t):
        """
        Exploration phase: beetle sprays chemicals toward predator.
        Returns a candidate new position vector.
        """
        # Select a random predator (any beetle except self)
        candidates = [j for j in range(self.pop_size) if j != i]
        pred_idx   = np.random.choice(candidates)
        predator   = self.population[pred_idx]

        # BBO parameters — all random per Table 2
        R = np.random.uniform(0, 1)
        r = np.random.uniform(0, 1) 
        d = np.random.uniform(0, 1)

        cia = self._cia(R, r, d)

        # Chemical Reaction components
        hot_vapor = 100.0   # boiling point of water
        O2        = np.random.uniform(0, 1)
        p_benzo   = np.random.uniform(0, 1)
        CR        = hot_vapor * O2 * p_benzo

        # Update chaos state for this beetle
        self.chaos_state[i] = self._chaos_map(self.chaos_state[i])
        spray = self._spray(self.chaos_state[i], t)

        # Position update — Eq. 31
        x_new = (self.population[i] + predator * cia * CR * self.population[i]) / spray
        return x_new

    #  Phase 2 — Escape / Exploitation  [Eq. 34]  
    def _phase2_escape(self, i, t):
        """
        Exploitation phase: beetle flies away using Newton's lift.
        Returns a candidate new position vector.
        """
        LC  = np.random.uniform(0, 1)   # lift coefficient
        rho = np.random.uniform(0, 1)   # air density
        V   = np.random.uniform(0, 1)   # wing velocity
        A   = np.random.uniform(0, 1)   # wing area

        # Eq. 34
        L = LC * 0.5 * rho * V**2 * A

        # Position update — shrinks as t grows (exploitation tightens)
        x_new = L * (self.ub - self.lb) / (t + 1)
        return x_new

    #  Greedy selection — Eq. 32      
    def _accept(self, i, x_new):
        """Accept x_new for beetle i if it is strictly better."""
        x_new   = np.clip(x_new, self.lb, self.ub)
        f_new   = self.obj_func(x_new)
        if f_new < self.fitness[i]:
            self.population[i] = x_new
            self.fitness[i]    = f_new
            if f_new < self.g_best_score:
                self.g_best_score = f_new
                self.g_best_pos   = x_new.copy()

    #  Main loop  
    def run(self):
        """
        Execute BBO for max_iter iterations.
        Returns:
            g_best_pos       — best position found
            g_best_score     — best fitness value found
            convergence_curve — list of best score after each iteration
        """
        convergence_curve = []

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Phase 1 — exploration (defense spray)
                x1 = self._phase1_defense(i, t)
                self._accept(i, x1)

                # Phase 2 — exploitation (escape lift)
                x2 = self._phase2_escape(i, t)
                self._accept(i, x2)

            convergence_curve.append(self.g_best_score)

        return self.g_best_pos, self.g_best_score, convergence_curve
