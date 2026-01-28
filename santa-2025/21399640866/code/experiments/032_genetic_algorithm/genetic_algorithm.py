"""
Genetic Algorithm for Tree Packing Optimization
Focus on N=2-20 (highest per-N scores)
"""
import numpy as np
from numba import njit
import math
import random
from shapely import Polygon
from shapely.affinity import rotate, translate
import pandas as pd
import json
import time

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlaps(xs, ys, angles):
    """Check if any trees overlap."""
    n = len(xs)
    if n <= 1:
        return False
    polygons = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    area = polygons[i].intersection(polygons[j]).area
                    if area > 1e-12:
                        return True
    return False

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    """Compute bounding box score."""
    n = len(xs)
    V = len(tx)
    mnx = 1e300
    mny = 1e300
    mxx = -1e300
    mxy = -1e300
    
    for i in range(n):
        r = angles[i] * math.pi / 180.0
        c = math.cos(r)
        s = math.sin(r)
        xi = xs[i]
        yi = ys[i]
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def strip(v):
    return float(str(v).replace("s", ""))

class Individual:
    """Represents a single solution (configuration for N trees)."""
    def __init__(self, n, xs=None, ys=None, angles=None):
        self.n = n
        if xs is None:
            # Random initialization
            self.xs = np.random.uniform(-2, 2, n)
            self.ys = np.random.uniform(-2, 2, n)
            self.angles = np.random.uniform(0, 360, n)
        else:
            self.xs = xs.copy()
            self.ys = ys.copy()
            self.angles = angles.copy()
        self.fitness = None
        self.valid = None
    
    def evaluate(self):
        """Compute fitness (lower is better)."""
        if self.fitness is None:
            self.valid = not check_overlaps(list(self.xs), list(self.ys), list(self.angles))
            if self.valid:
                self.fitness = compute_bbox_score(self.xs, self.ys, self.angles, TX, TY)
            else:
                self.fitness = 1e10  # Penalty for invalid solutions
        return self.fitness
    
    def mutate(self, mutation_rate=0.1, position_std=0.05, angle_std=5.0):
        """Apply mutation to the individual."""
        for i in range(self.n):
            if random.random() < mutation_rate:
                self.xs[i] += np.random.normal(0, position_std)
                self.ys[i] += np.random.normal(0, position_std)
                self.angles[i] += np.random.normal(0, angle_std)
                self.angles[i] = self.angles[i] % 360
        self.fitness = None
        self.valid = None
    
    def copy(self):
        return Individual(self.n, self.xs, self.ys, self.angles)

def crossover(parent1, parent2):
    """Single-point crossover."""
    n = parent1.n
    point = random.randint(1, n-1) if n > 1 else 1
    
    child1 = Individual(n)
    child2 = Individual(n)
    
    child1.xs = np.concatenate([parent1.xs[:point], parent2.xs[point:]])
    child1.ys = np.concatenate([parent1.ys[:point], parent2.ys[point:]])
    child1.angles = np.concatenate([parent1.angles[:point], parent2.angles[point:]])
    
    child2.xs = np.concatenate([parent2.xs[:point], parent1.xs[point:]])
    child2.ys = np.concatenate([parent2.ys[:point], parent1.ys[point:]])
    child2.angles = np.concatenate([parent2.angles[:point], parent1.angles[point:]])
    
    return child1, child2

def tournament_select(population, tournament_size=3):
    """Tournament selection."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda x: x.evaluate())

def genetic_algorithm(n, baseline_xs, baseline_ys, baseline_angles, 
                      pop_size=50, generations=200, mutation_rate=0.2,
                      position_std=0.02, angle_std=2.0, elite_size=5):
    """Run genetic algorithm for a single N value."""
    
    # Initialize population with baseline and mutations of it
    population = []
    
    # Add baseline
    baseline = Individual(n, baseline_xs, baseline_ys, baseline_angles)
    population.append(baseline)
    
    # Add mutations of baseline
    for _ in range(pop_size - 1):
        ind = baseline.copy()
        ind.mutate(mutation_rate=0.5, position_std=position_std*2, angle_std=angle_std*2)
        population.append(ind)
    
    best_ever = baseline.copy()
    best_ever.evaluate()
    
    for gen in range(generations):
        # Evaluate all
        for ind in population:
            ind.evaluate()
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness)
        
        # Update best
        if population[0].valid and population[0].fitness < best_ever.fitness:
            best_ever = population[0].copy()
            best_ever.fitness = population[0].fitness
            best_ever.valid = True
        
        # Elitism - keep best individuals
        new_population = [ind.copy() for ind in population[:elite_size]]
        
        # Generate rest through selection and crossover
        while len(new_population) < pop_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            child1, child2 = crossover(parent1, parent2)
            child1.mutate(mutation_rate, position_std, angle_std)
            child2.mutate(mutation_rate, position_std, angle_std)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
    
    return best_ever

def main():
    print("=" * 70)
    print("Genetic Algorithm for Tree Packing Optimization")
    print("=" * 70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    # Focus on N=2-20 (highest per-N scores)
    target_ns = list(range(2, 21))
    
    improvements = []
    total_improvement = 0
    
    start_time = time.time()
    
    for n in target_ns:
        g = baseline_df[baseline_df['N'] == n]
        baseline_xs = np.array([strip(v) for v in g['x']])
        baseline_ys = np.array([strip(v) for v in g['y']])
        baseline_angles = np.array([strip(v) for v in g['deg']])
        
        baseline_score = compute_bbox_score(baseline_xs, baseline_ys, baseline_angles, TX, TY)
        
        # Run GA with different parameters based on N
        if n <= 5:
            # Small N: more exploration
            best = genetic_algorithm(n, baseline_xs, baseline_ys, baseline_angles,
                                    pop_size=100, generations=500, mutation_rate=0.3,
                                    position_std=0.05, angle_std=10.0, elite_size=10)
        elif n <= 10:
            # Medium N
            best = genetic_algorithm(n, baseline_xs, baseline_ys, baseline_angles,
                                    pop_size=80, generations=300, mutation_rate=0.25,
                                    position_std=0.03, angle_std=5.0, elite_size=8)
        else:
            # Larger N: more conservative
            best = genetic_algorithm(n, baseline_xs, baseline_ys, baseline_angles,
                                    pop_size=60, generations=200, mutation_rate=0.2,
                                    position_std=0.02, angle_std=3.0, elite_size=6)
        
        if best.valid and best.fitness < baseline_score - 0.0001:
            improvement = baseline_score - best.fitness
            improvements.append((n, improvement, best))
            total_improvement += improvement
            print(f"N={n:3d}: {baseline_score:.6f} -> {best.fitness:.6f} (+{improvement:.6f}) ✓")
        else:
            print(f"N={n:3d}: {baseline_score:.6f} (no improvement)")
    
    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.1f}s")
    print(f"Total improvements: {len(improvements)}")
    print(f"Total improvement: {total_improvement:.6f}")
    
    # Save results
    results = {
        'improvements': [(n, imp) for n, imp, _ in improvements],
        'total_improvement': total_improvement,
        'elapsed_time': elapsed,
        'target_ns': target_ns
    }
    
    with open('ga_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return improvements

if __name__ == "__main__":
    improvements = main()
