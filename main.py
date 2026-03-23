import sys
import os
# This line helps Python find your 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bbo import BombardierBeetleOptimizer
from benchmarks.functions import rastrigin, sphere
import matplotlib.pyplot as plt

def main():
    print("--- Starting Bombardier Beetle Optimization ---")
    
    # Configuration
    DIMENSIONS = 10
    ITERATIONS = 250
    POP_SIZE = 40
    
    # Initialize Optimizer (Let's test on Rastrigin)
    bbo = BombardierBeetleOptimizer(
        obj_func=rastrigin, 
        dims=DIMENSIONS, 
        iterations=ITERATIONS, 
        pop_size=POP_SIZE
    )
    
    # Run
    best_pos, best_score, curve = bbo.run()
    
    print("\n--- Optimization Complete ---")
    print(f"Best Score Found: {best_score}")
    
    # Plotting for Research Paper
    plt.figure(figsize=(10, 5))
    plt.plot(curve, color='red', linewidth=2, label='BBO Fitness')
    plt.title("Convergence Curve: BBO on Rastrigin Function")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (Lower is Better)")
    plt.yscale('log') # Log scale is standard for optimization research
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Save the plot for your paper
    plt.savefig("bbo_convergence.png")
    print("Graph saved as 'bbo_convergence.png'")
    plt.show()

if __name__ == "__main__":
    main()