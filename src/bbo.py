import numpy as np

class BombardierBeetleOptimizer:
    def __init__(self, obj_func, dims, pop_size=50, iterations=100, lb=-10, ub=10):
        self.obj_func = obj_func
        self.dims = dims
        self.pop_size = pop_size
        self.max_iter = iterations
        self.lb = np.array([lb] * dims)
        self.ub = np.array([ub] * dims)
        
        # Initialize Population
        self.population = np.random.uniform(lb, ub, (pop_size, dims))
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # Track the Best
        idx = np.argmin(self.fitness)
        self.g_best_pos = self.population[idx].copy()
        self.g_best_score = self.fitness[idx]

    def update(self, t):
        # alpha controls the 'explosive' force of the spray
        # It reduces over time to allow for fine-tuning (Exploitation)
        alpha = 2.0 * (1 - (t / self.max_iter)) 
        
        for i in range(self.pop_size):
            r = np.random.rand()
            
            if r < 0.5:
                # PHASE 1: EXPLOSIVE SPRAY (Exploration)
                # Beetle sprays chemicals in random directions
                step = np.random.uniform(-1, 1, self.dims) * (self.ub - self.lb) * (alpha / 10)
                new_pos = self.population[i] + step
            else:
                # PHASE 2: PRECISION AIM (Exploitation)
                # Beetle aims the spray toward the best known target
                phi = np.random.uniform(-1, 1, self.dims)
                new_pos = self.g_best_pos + phi * np.abs(self.g_best_pos - self.population[i])
            
            # Keep within bounds
            new_pos = np.clip(new_pos, self.lb, self.ub)
            
            # Selection: Only move if the new spot is better
            new_fit = self.obj_func(new_pos)
            if new_fit < self.fitness[i]:
                self.population[i] = new_pos
                self.fitness[i] = new_fit
                
                if new_fit < self.g_best_score:
                    self.g_best_score = new_fit
                    self.g_best_pos = new_pos.copy()

    def run(self):
        convergence_curve = []
        for t in range(self.max_iter):
            self.update(t)
            convergence_curve.append(self.g_best_score)
            if t % 10 == 0:
                print(f"Iteration {t}: Best Score = {self.g_best_score:.4e}")
        return self.g_best_pos, self.g_best_score, convergence_curve