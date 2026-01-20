"""
Genetic Algorithm for tree packing optimization.
Uses crossover and mutation to explore diverse configurations.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
import random
import math
import time

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

def create_tree(x, y, deg):
    tree = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
    tree = affinity.translate(tree, x, y)
    return tree

def get_side(trees):
    if not trees:
        return 0
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for tree in trees:
        bounds = tree.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    return max(max_x - min_x, max_y - min_y)

def check_any_overlap(trees, tolerance=1e-9):
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True
    return False

def calculate_overlap_area(trees, tolerance=1e-9):
    total = 0
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    total += intersection.area
    return total

class Individual:
    def __init__(self, n):
        self.n = n
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.deg = np.zeros(n)
        self._fitness = None
        self._trees = None
    
    def get_trees(self):
        if self._trees is None:
            self._trees = [create_tree(self.x[i], self.y[i], self.deg[i]) for i in range(self.n)]
        return self._trees
    
    def invalidate(self):
        self._fitness = None
        self._trees = None
    
    def fitness(self):
        if self._fitness is None:
            trees = self.get_trees()
            side = get_side(trees)
            score = side**2 / self.n
            overlap = calculate_overlap_area(trees)
            # Fitness = score + heavy penalty for overlaps
            self._fitness = score + overlap * 10000
        return self._fitness
    
    def is_valid(self):
        return not check_any_overlap(self.get_trees())
    
    def copy(self):
        ind = Individual(self.n)
        ind.x = self.x.copy()
        ind.y = self.y.copy()
        ind.deg = self.deg.copy()
        return ind

def random_individual(n, spread=None):
    """Create a random individual."""
    if spread is None:
        spread = 0.4 * math.sqrt(n)
    
    ind = Individual(n)
    for i in range(n):
        ind.x[i] = random.uniform(-spread, spread)
        ind.y[i] = random.uniform(-spread, spread)
        ind.deg[i] = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
    return ind

def load_baseline_individual(n, df):
    """Load individual from baseline dataframe."""
    prefix = f"{n:03d}_"
    n_rows = df[df['id'].str.startswith(prefix)]
    
    ind = Individual(n)
    for idx, (_, row) in enumerate(n_rows.iterrows()):
        x = str(row['x']).replace('s', '')
        y = str(row['y']).replace('s', '')
        deg = str(row['deg']).replace('s', '')
        ind.x[idx] = float(x)
        ind.y[idx] = float(y)
        ind.deg[idx] = float(deg)
    return ind

def crossover(parent1, parent2):
    """Crossover two parents to create a child."""
    n = parent1.n
    child = Individual(n)
    
    # Uniform crossover
    for i in range(n):
        if random.random() < 0.5:
            child.x[i] = parent1.x[i]
            child.y[i] = parent1.y[i]
            child.deg[i] = parent1.deg[i]
        else:
            child.x[i] = parent2.x[i]
            child.y[i] = parent2.y[i]
            child.deg[i] = parent2.deg[i]
    
    return child

def mutate(ind, mutation_rate=0.1, mutation_strength=0.1):
    """Mutate an individual."""
    for i in range(ind.n):
        if random.random() < mutation_rate:
            # Position mutation
            ind.x[i] += random.gauss(0, mutation_strength)
            ind.y[i] += random.gauss(0, mutation_strength)
        
        if random.random() < mutation_rate:
            # Angle mutation
            if random.random() < 0.5:
                ind.deg[i] += random.gauss(0, 15)
            else:
                ind.deg[i] = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            ind.deg[i] %= 360
    
    ind.invalidate()
    return ind

def tournament_select(population, tournament_size=3):
    """Select individual using tournament selection."""
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: x.fitness())

def genetic_algorithm(n, population_size=50, generations=100, baseline_df=None):
    """Run genetic algorithm for N trees."""
    # Initialize population
    population = []
    
    # Add baseline if available
    if baseline_df is not None:
        baseline = load_baseline_individual(n, baseline_df)
        population.append(baseline)
        
        # Add mutated versions of baseline
        for _ in range(population_size // 4):
            mutated = baseline.copy()
            mutate(mutated, mutation_rate=0.3, mutation_strength=0.2)
            population.append(mutated)
    
    # Fill rest with random individuals
    while len(population) < population_size:
        population.append(random_individual(n))
    
    best = min(population, key=lambda x: x.fitness())
    best_valid = None
    best_valid_fitness = float('inf')
    
    if best.is_valid():
        best_valid = best.copy()
        best_valid_fitness = best.fitness()
    
    for gen in range(generations):
        # Create new population
        new_population = []
        
        # Elitism - keep best
        new_population.append(best.copy())
        
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            mutate(child, mutation_rate=0.15, mutation_strength=0.1)
            
            new_population.append(child)
        
        population = new_population
        
        # Update best
        current_best = min(population, key=lambda x: x.fitness())
        if current_best.fitness() < best.fitness():
            best = current_best.copy()
        
        # Track best valid
        for ind in population:
            if ind.is_valid() and ind.fitness() < best_valid_fitness:
                best_valid = ind.copy()
                best_valid_fitness = ind.fitness()
        
        if gen % 20 == 0:
            valid_count = sum(1 for ind in population if ind.is_valid())
            print(f"  Gen {gen}: best={best.fitness():.6f}, valid={valid_count}/{population_size}")
    
    return best_valid, best_valid_fitness if best_valid else float('inf')

def main():
    print("Genetic Algorithm Optimization")
    print("=" * 50)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    
    results = {}
    total_improvement = 0
    
    # Focus on small N values
    for n in range(2, 11):
        print(f"\nOptimizing N={n}...")
        start_time = time.time()
        
        # Get baseline score
        baseline_ind = load_baseline_individual(n, baseline_df)
        baseline_score = get_side(baseline_ind.get_trees())**2 / n
        
        # Run GA
        best, best_score = genetic_algorithm(n, population_size=60, generations=80, baseline_df=baseline_df)
        
        elapsed = time.time() - start_time
        
        if best is not None:
            improvement = baseline_score - best_score
            total_improvement += max(0, improvement)
            
            status = "✓ BETTER" if improvement > 0.0001 else "= same" if abs(improvement) < 0.0001 else "✗ worse"
            print(f"  N={n}: baseline={baseline_score:.6f}, GA={best_score:.6f}, diff={improvement:+.6f} {status} ({elapsed:.1f}s)")
            
            if improvement > 0.0001:
                results[n] = {
                    'x': best.x,
                    'y': best.y,
                    'deg': best.deg,
                    'score': best_score,
                    'improvement': improvement
                }
        else:
            print(f"  N={n}: No valid solution found ({elapsed:.1f}s)")
    
    print(f"\nTotal improvement: {total_improvement:+.6f}")
    
    return results

if __name__ == "__main__":
    results = main()
