"""
Hardware Detection Utility
==========================
Detects available CPU cores and recommends the optimal
number of parallel workers for the experiment runner.

Rules:
  - Always use 75% of logical cores.
  - Always leave at least 1 core free for the OS.
"""

import multiprocessing

def get_worker_count():
    total     = multiprocessing.cpu_count()
    n_workers = max(1, int(total * 0.75))
    return n_workers, total, 75

#  Summary print 
def print_hardware_summary():
    n_workers, total, pct = get_worker_count()

    print("=" * 50)
    print("  Hardware Configuration")
    print("=" * 50)
    print(f"  CPU logical cores : {total}")
    print(f"  Workers selected  : {n_workers}  ({pct}% of {total} cores)")
    print("=" * 50)

if __name__ == "__main__":
    print_hardware_summary()
