# Product Requirements Document
# BBO Improvement Showcase

**Date:** 2026-03-27
**Status:** v2 — Updated based on full project review
**Goal:** Implement the exact BBO algorithm from the paper, improve it with Nonlinear Adaptive Decay (Option B), benchmark both on CEC-2017 via opfunu, and produce a clear visual showcase.

---

## 1. What We Are Building

Three algorithms, one comparison:

| Algorithm | File | Description |
|-----------|------|-------------|
| **BBO (Simplified)** | `src/bbo.py` | Existing implementation — kept as-is. The quick approximation we already have. |
| **BBO (Exact)** | `algorithms/bbo.py` | True implementation of Shehadeh et al.'s paper — CIA formula, chaos maps, chemical reaction, Newton's lift. This is our **paper-faithful baseline**. |
| **NAD-BBO** | `algorithms/bbo_improved.py` | Nonlinear Adaptive Decay BBO — same as BBO Exact but with nonlinear decay schedule and adaptive phase-switching ratio. This is our **proposed improvement**. |

The current `src/bbo.py` (Simplified) is kept for reference and historical comparison. It is missing:
- Circle Intersection Area (CIA) formula — Eq. 30
- Chaos maps (sinusoidal, Chebyshev, tent, etc.) — Eq. 33
- Chemical Reaction (CR) components — hot vapor (100) + O₂ (rand) + p-benzoquinone (rand)
- `SelectedPredator` concept
- Newton's insect lift equation — Eq. 34
- Correct position update formula: `x_PBB = (x + SelectedPredator × CIA × CR × x) / Spray`

---

## 2. Exact BBO Algorithm (Baseline)

### Initialization (Eq. 29)
```
x_id = lb_d + rand(0,1) × (ub_d - lb_d)
```

### Phase 1 — Defense Mechanism (Exploration)
1. Pick a random `SelectedPredator` from population (not beetle i itself)
2. Calculate CIA = Circle Intersection Area between beetle i (radius R) and predator (radius r), distance d:
   - If d > R + r → CIA = 0 (no overlap)
   - If d ≤ |R - r| → CIA = π × min(R,r)²  (fully inside)
   - Otherwise → Use full Eq. 30 formula
3. Chemical Reaction: `CR = hot_vapor × O2 × p_benzoquinone`
   - hot_vapor = 100, O2 = rand(0,1), p_benzoquinone = rand(0,1)
4. Chaos value: sinusoidal map → `chaos_{n+1} = sin(π × chaos_n)`
5. Spray = `chaos × 2.7^(100^(t / Max_Iter))`    (Eq. 33)
6. New position: `x_new = (x_id + SelectedPredator × CIA × CR × x_id) / Spray`    (Eq. 31)
7. Accept if better (Eq. 32), clip to bounds

### Phase 2 — Escape via Insect Lift (Exploitation)
1. Lift: `L = LC × 0.5 × ρ × V² × A`    (Eq. 34)
   - LC, ρ, V, A = rand(0,1) each
2. New position: `x_new = L × (ub_d - lb_d) / t`
3. Accept if better (Eq. 32), clip to bounds

### Best Tracking
- Record global best after each full iteration

---

## 3. Improvement Strategy — NAD-BBO (Option B: Nonlinear Adaptive Decay)

### Motivation
The exact BBO has two implicit decay mechanisms:
- Phase 1: Spray grows as `2.7^(100^(t/T))` — decay is governed by **linear** t/T ratio
- Phase 2: Position shrinks as `L × (ub-lb) / t` — shrinkage tied to **linear** iteration counter

Both use linear time ratios, which means the algorithm transitions from exploration to exploitation at a constant rate. This is suboptimal:
- **Early iterations** should stay exploratory longer (linear decay is too aggressive)
- **Late iterations** should commit to exploitation faster (linear decay is too slow near end)

### What We Change

**Change 1 — Nonlinear Spray Decay:**
Replace linear `t/T` in the spray exponent with a **sine-based nonlinear ratio**:
```
# Original (linear):
ratio = t / Max_Iter
Spray = chaos × 2.7^(100^ratio)

# NAD-BBO (nonlinear):
ratio = sin(π/2 × t / Max_Iter)   ← slow start, fast finish
Spray = chaos × 2.7^(100^ratio)
```
Effect: Early iterations have lower spray values → larger exploration steps. The spray "explosion" happens later and more sharply.

**Change 2 — Adaptive Phase-Switching Probability:**
The original BBO runs BOTH Phase 1 and Phase 2 for every beetle every iteration. We replace this with an adaptive probability that determines which phase runs:
```
p_explore(t) = cos²(π/2 × t / Max_Iter)   ← starts ~1.0, ends ~0.0

If rand() < p_explore(t):  → Run Phase 1 (exploration spray)
Else:                       → Run Phase 2 (exploitation lift)
```
Effect: Early iterations are ~100% exploration. Late iterations are ~100% exploitation. The transition follows a smooth cosine curve — no abrupt switching.

### Why This Is a Valid Academic Improvement
- Both changes are motivated by the **exploration-exploitation tradeoff** — a foundational concept in metaheuristics
- Nonlinear scheduling is used in well-known algorithms (LSHADE, CMA-ES, SCA) and consistently outperforms linear scheduling
- The change is **minimal** — same BBO mechanics, same CIA formula, same physics — only the scheduling is different
- Easy to explain: "We made the beetle stay curious longer, then commit to the best solution faster"

---

## 4. Repository Structure (Cleaned Up)

Current repo is messy — files scattered at root level. New structure:

```
BOMBARDIER-BEETLE-OPTIMIZER-BBO-/
│
├── algorithms/                  ← All algorithm implementations
│   ├── __init__.py
│   ├── bbo.py                   ← Exact BBO from paper
│   └── bbo_improved.py          ← NAD-BBO (our improvement)
│
├── benchmarks/                  ← Benchmark functions
│   ├── __init__.py
│   └── functions.py             ← Custom: sphere, rastrigin, rosenbrock (kept, not used in main experiments)
│
├── experiments/                 ← Experiment runners
│   ├── __init__.py
│   ├── run_cec2017.py           ← Main: CEC-2017 via opfunu, 30 trials, both algos → CSV
│   └── run_custom.py            ← Optional: custom benchmark quick test
│
├── results/                     ← All outputs (gitignored large files)
│   ├── csv/
│   │   ├── bbo_cec2017.csv      ← BBO Exact results
│   │   └── nad_bbo_cec2017.csv  ← NAD-BBO results
│   └── plots/                   ← Generated PNG charts
│
├── analysis/                    ← Post-experiment analysis scripts
│   ├── plot_convergence.py      ← Convergence curves (BBO vs NAD-BBO per function)
│   ├── plot_comparison.py       ← Bar charts + mean rank chart
│   └── summary.py              ← Terminal summary: which algo won, by how much
│
├── demo.py                      ← Single command live showcase (real-time plots)
├── requirements.txt
├── .agent/context/              ← Research documents (PDFs, PRD, PLAN)
└── README.md
```

**Files to retire (moved/replaced):**
| Old File | Action |
|----------|--------|
| `src/bbo.py` | **Kept as-is** — Simplified BBO, used as 3rd comparison point |
| `run_cec_opfunu.py` | Replaced by `experiments/run_cec2017.py` |
| `run_cec_smoke.py` | Retired (was just a test) |
| `experiments.py` | Replaced by `experiments/run_custom.py` |
| `main.py` | Retired (replaced by `demo.py`) |
| `tools/list_cec.py` | Retired (utility no longer needed) |
| `research_results.csv` | Moved to `results/csv/` |
| `research_results_cec.csv` | Moved to `results/csv/bbo_simplified_cec.csv` (results from simplified BBO) |
| `bbo_convergence.png` | Moved to `results/plots/` |

---

## 5. Benchmarking — CEC-2017 via opfunu

**Why CEC-2017:** Same benchmark suite used in the original BBO paper. Using opfunu ensures identical function definitions to the paper.

**Experiment Parameters (matching paper's Table 2):**
- Population size: 30
- Dimensions: 10
- Iterations: 1000 (paper used 1000; we may reduce to 500 for demo speed)
- Trials: 30 (statistical significance)
- Bounds: [-100, 100] for most functions

**Functions:** All available CEC-2017 functions from opfunu (F1–F29 based on what's available)

**Output per algorithm per function:**
- Best (minimum fitness across 30 trials)
- Mean (average fitness)
- Worst (maximum fitness)
- Std Dev (standard deviation)

**Ranking:** Mean rank computed per function (lower fitness = better rank), summed across all functions — like paper's Table 3-5 and Fig. 10.

---

## 6. Results Pipeline

```
Step 1: experiments/run_cec2017.py
        → results/csv/bbo_simplified_cec2017.csv   (Simplified BBO — already done, reuse or re-run)
        → results/csv/bbo_exact_cec2017.csv         (Exact BBO)
        → results/csv/nad_bbo_cec2017.csv           (NAD-BBO)

Step 2: analysis/plot_convergence.py
        → results/plots/convergence_F1.png ... convergence_FN.png
        (3 lines per plot: Simplified BBO vs Exact BBO vs NAD-BBO)

Step 3: analysis/plot_comparison.py
        → results/plots/mean_rank_bar.png
        → results/plots/best_score_comparison.png
        (bar charts: all 3 algorithms on Best, Mean, Std)

Step 4: analysis/summary.py
        → Terminal table: function-by-function winner across all 3
        → Overall: "NAD-BBO wins on X/N functions"
        → Mean rank: Simplified = A.A | Exact = B.B | NAD-BBO = C.C
        → Narrative: shows progression from simplified → exact → improved
```

---

## 7. Live Demo (Showcase Script)

`demo.py` — runs everything live in front of the teacher:

1. Picks 5 representative CEC-2017 functions (fast to run)
2. Runs BBO Exact on all 5 (5 trials for speed), printing live trial scores
3. Runs NAD-BBO on same 5 functions, same seeds
4. Shows **real-time convergence plot** (matplotlib interactive) — curve draws itself as iterations run
5. After both complete: shows final comparison bar chart
6. Prints summary table to terminal

---

## 8. Success Criteria

- [ ] Simplified BBO (`src/bbo.py`) kept unchanged and included in comparison
- [ ] Exact BBO fully implements: CIA, chaos map, CR, Newton lift, Eq. 31, Eq. 32
- [ ] NAD-BBO changes only the decay schedule and phase probability — nothing else
- [ ] All 3 algorithms run on all CEC-2017 functions, 30 trials, output to CSV
- [ ] Convergence plots show clear progression: Simplified → Exact → NAD-BBO
- [ ] NAD-BBO achieves lowest total mean rank of the three
- [ ] `python demo.py` works end-to-end, shows real-time plot, finishes under 5 minutes

---

## 9. Out of Scope

- GWO or any other algorithm comparison (just BBO vs NAD-BBO)
- Real-world engineering application problems
- Multi-objective or binary variants
- Comparison against the paper's results (different implementation language — paper uses MATLAB)
