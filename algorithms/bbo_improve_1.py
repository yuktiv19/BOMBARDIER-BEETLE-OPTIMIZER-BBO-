"""
NAD-BBO — Nonlinear Adaptive Decay Bombardier Beetle Optimizer
===============================================================
Improvement over the exact BBO (algorithms/bbo.py) using two changes:

  Change 1 — Nonlinear Spray Decay:
      Original BBO uses a linear ratio t/T inside the spray exponent.
      NAD-BBO replaces it with sin(π/2 · t/T) — a slow start, fast finish
      curve that keeps exploration active longer in early iterations and
      commits to exploitation more sharply near convergence.

      Original : ratio = t / T
                 Spray = chaos * 2.7^(100^ratio)

      NAD-BBO  : ratio = sin(π/2 · t/T)          ← nonlinear
                 Spray = chaos * 2.7^(100^ratio)

  Change 2 — Adaptive Phase-Switching Probability:
      Original BBO always runs BOTH Phase 1 and Phase 2 every iteration.
      NAD-BBO selects ONE phase per beetle per iteration based on a
      cosine-squared probability:

          p_explore(t) = cos²(π/2 · t/T)   → ~1.0 early, ~0.0 late

      So early iterations are mostly exploration (Phase 1) and late
      iterations are mostly exploitation (Phase 2).

Everything else — CIA formula, chemical reaction, Newton's lift, greedy
selection — is identical to the exact BBO.
"""

import numpy as np
from algorithms.bbo import BBO


class NADBBO(BBO):
    """
    Nonlinear Adaptive Decay BBO.
    Inherits all mechanics from BBO; overrides only spray and run().
    """

    #  Change 1 — Nonlinear spray decay
    def _nonlinear_ratio(self, t):
        """
        sin(π/2 · t/T) maps [0, T] → [0, 1] with a slow start.
        At t=0   → ratio ≈ 0   (spray small → large exploration steps)
        At t=T/2 → ratio ≈ 0.7 (midpoint)
        At t=T   → ratio = 1   (spray maximal → tight exploitation)
        """
        return np.sin(np.pi / 2 * t / max(self.max_iter - 1, 1))

    def _spray(self, chaos_val, t):
        """
        Override: use nonlinear ratio instead of linear t/T.
        Eq. 33 (modified): Spray = chaos * 2.7^(100^nonlinear_ratio)
        """
        ratio    = self._nonlinear_ratio(t)
        exponent = 100 ** ratio
        exponent = min(exponent, 700)   # overflow guard
        spray    = chaos_val * (2.7 ** exponent)
        return max(spray, 1e-10)

    #  Change 2 — Adaptive phase-switching probability
    def _explore_prob(self, t):
        """
        cos²(π/2 · t/T) maps [0, T] → [1.0, 0.0].
        At t=0 → p ≈ 1.0  (almost always Phase 1 / exploration)
        At t=T → p ≈ 0.0  (almost always Phase 2 / exploitation)
        """
        return np.cos(np.pi / 2 * t / max(self.max_iter - 1, 1)) ** 2

    #  Overridden main loop
    def run(self):
        """
        NAD-BBO main loop.
        Instead of running both phases every iteration, a single phase is
        chosen per beetle based on the adaptive probability p_explore(t).
        """
        convergence_curve = []

        for t in range(self.max_iter):
            p = self._explore_prob(t)      # probability of choosing Phase 1

            for i in range(self.pop_size):
                if np.random.rand() < p:
                    # Phase 1 — exploration (spray)
                    x_new = self._phase1_defense(i, t)
                else:
                    # Phase 2 — exploitation (lift)
                    x_new = self._phase2_escape(i, t)

                self._accept(i, x_new)

            convergence_curve.append(self.g_best_score)

        return self.g_best_pos, self.g_best_score, convergence_curve
