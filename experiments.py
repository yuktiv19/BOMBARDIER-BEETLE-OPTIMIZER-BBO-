import numpy as np
import pandas as pd
from src.bbo import BombardierBeetleOptimizer
from benchmarks.functions import sphere, rastrigin, rosenbrock

def run_statistical_test(func, func_name, trials=30):
    print(f"--- Testing {func_name} for {trials} trials ---")
    results = []
    
    for i in range(trials):
        # We use a different random seed each time for true randomness
        np.random.seed(i) 
        bbo = BombardierBeetleOptimizer(obj_func=func, dims=10, iterations=100, pop_size=30)
        _, best_score, _ = bbo.run()
        results.append(best_score)
        print(f"Trial {i+1}/{trials}: {best_score:.4e}")

    return {
        "Function": func_name,
        "Best": np.min(results),
        "Worst": np.max(results),
        "Mean": np.mean(results),
        "Std Dev": np.std(results)
    }

# Run tests on all functions
test_benchmarks = [
    (sphere, "Sphere"),
    (rastrigin, "Rastrigin"),
    (rosenbrock, "Rosenbrock")
]

final_data = []
for func, name in test_benchmarks:
    stats = run_statistical_test(func, name)
    final_data.append(stats)

# Create a professional table
df = pd.DataFrame(final_data)
print("\n--- FINAL RESEARCH TABLE ---")
print(df.to_string(index=False))

# Save to CSV for your Excel/Word document
df.to_csv("research_results.csv", index=False)
print("\nResults saved to 'research_results.csv'")