# Bombardier Beetle Optimizer (BBO)

A Python implementation of the **Bombardier Beetle Optimizer**, a bio-inspired metaheuristic algorithm based on the unique defense mechanism of the bombardier beetle.

## 📌 Overview
The BBO algorithm simulates the beetle's ability to store chemical precursors and release an explosive, boiling spray to deter predators. In optimization terms, this provides a dynamic balance between **Exploration** (the spray) and **Exploitation** (the aim).

## 🚀 Features
* **Exploration Phase:** High-velocity random search simulating the "chemical spray."
* **Exploitation Phase:** Fine-tuned local search simulating "directional aiming."
* **Benchmark Support:** Tested against standard CEC-2017 functions.

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/BOMBARDIER-BEETLE-OPTIMIZER-BBO-.git
cd BOMBARDIER-BEETLE-OPTIMIZER-BBO-

# Install dependencies (numpy, opfunu, matplotlib, pandas)
pip install -r requirements.txt
```

## 🎮 How to Run

### 1. Live Visual Demo
Runs a side-by-side animated comparison between the Exact BBO and your latest improvement.
```bash
python demo.py
```

### 2. Full Experiment Runner (CEC-2017)
Runs 30 independent trials across 29 benchmark functions with parallel processing.
```bash
# Run everything (Exact vs Improve-1)
python experiments/run_cec2017.py --algo all

# Run ONLY the research-grade Exact baseline
python experiments/run_cec2017.py --algo exact

# Run ONLY your improved version
python experiments/run_cec2017.py --algo improve-1

# Quick smoke test (1 function, 3 trials)
python experiments/run_cec2017.py --algo improve-1 --smoke
```

## 📈 Incremental Improvement System

This repository follows a strict version tracking system for developing new algorithm variants.

### How to create `IMPROVE-2`:
1. **Create the file:**
   Copy the latest version: `cp algorithms/bbo_improve_1.py algorithms/bbo_improve_2.py`
2. **Register in runner:**
   Open `run_cec2017.py` and `demo.py` and add your new version to the `choices` and algorithm lists.
3. **Execute:**
   ```bash
   python experiments/run_cec2017.py --algo improve-2
   ```

## 📁 Repository Structure
* `algorithms/`: The mathematical implementations (Exact, Improve-1, etc.).
* `experiments/`: Data-heavy benchmarking scripts.
* `results/`: CSV data outputs and statistical plots.
* `utils/`: Hardware detection and utility helpers.