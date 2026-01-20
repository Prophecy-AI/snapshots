"""
Simulated Annealing from Random Starts
Generate new configurations by starting from random placements and optimizing.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
from copy import deepcopy
import time
import random
import math

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])
BASE_TREE = Polygon(zip(TX, TY))

def create_tree(x, y, deg):
    """Create a tree polygon at position (x, y) with rotation deg."""
    tree = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
    tree = affinity.translate(tree, x, y)
    return tree

def get_side(trees):
    """Get the side length of the bounding box."""
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
    """Check if any trees overlap."""
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True
    return False

def calculate_overlap_area(trees, tolerance=1e-9):
    """Calculate total overlap area."""
    total = 0
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    total += intersection.area
    return total

class Configuration:
    def __init__(self, n):
        self.n = n
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.deg = np.zeros(n)
        self.trees = [None] * n
    
    def update_tree(self, i):
        self.trees[i] = create_tree(self.x[i], self.y[i], self.deg[i])
    
    def update_all(self):
        for i in range(self.n):
            self.update_tree(i)
    
    def get_side(self):
        return get_side(self.trees)
    
    def get_score(self):
        return self.get_side()**2 / self.n
    
    def has_overlap(self):
        return check_any_overlap(self.trees)
    
    def get_overlap_area(self):
        return calculate_overlap_area(self.trees)
    
    def copy(self):
        c = Configuration(self.n)
        c.x = self.x.copy()
        c.y = self.y.copy()
        c.deg = self.deg.copy()
        c.trees = [t for t in self.trees]
        return c

def random_configuration(n, spread=None):
    """Create a random configuration."""
    if spread is None:
        spread = 0.5 * math.sqrt(n)  # Rough estimate of needed space
    
    config = Configuration(n)
    for i in range(n):
        config.x[i] = random.uniform(-spread, spread)
        config.y[i] = random.uniform(-spread, spread)
        config.deg[i] = random.uniform(0, 360)
    config.update_all()
    return config

def simulated_annealing(config, iterations=5000, t_start=1.0, t_end=0.001):
    """Run simulated annealing on a configuration."""
    best = config.copy()
    best_fitness = best.get_score() + best.get_overlap_area() * 1000
    
    current = config.copy()
    current_fitness = best_fitness
    
    for i in range(iterations):
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (i / iterations)
        
        # Create neighbor
        neighbor = current.copy()
        tree_idx = random.randint(0, config.n - 1)
        
        move_type = random.random()
        if move_type < 0.4:
            # Small translation
            neighbor.x[tree_idx] += random.gauss(0, 0.1 * t)
            neighbor.y[tree_idx] += random.gauss(0, 0.1 * t)
        elif move_type < 0.7:
            # Small rotation
            neighbor.deg[tree_idx] += random.gauss(0, 30 * t)
            neighbor.deg[tree_idx] %= 360
        else:
            # Larger move
            neighbor.x[tree_idx] += random.gauss(0, 0.3)
            neighbor.y[tree_idx] += random.gauss(0, 0.3)
            neighbor.deg[tree_idx] = random.uniform(0, 360)
        
        neighbor.update_tree(tree_idx)
        
        # Calculate fitness (score + overlap penalty)
        neighbor_score = neighbor.get_score()
        neighbor_overlap = neighbor.get_overlap_area()
        neighbor_fitness = neighbor_score + neighbor_overlap * 1000
        
        # Accept or reject
        delta = neighbor_fitness - current_fitness
        if delta < 0 or random.random() < math.exp(-delta / t):
            current = neighbor
            current_fitness = neighbor_fitness
            
            if current_fitness < best_fitness:
                best = current.copy()
                best_fitness = current_fitness
    
    return best

def local_search(config, max_iter=1000):
    """Local search to remove overlaps and improve score."""
    best = config.copy()
    best_fitness = best.get_score() + best.get_overlap_area() * 1000
    
    for _ in range(max_iter):
        improved = False
        for i in range(config.n):
            # Try small moves in 8 directions
            for dx, dy in [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01),
                          (0.007, 0.007), (0.007, -0.007), (-0.007, 0.007), (-0.007, -0.007)]:
                neighbor = best.copy()
                neighbor.x[i] += dx
                neighbor.y[i] += dy
                neighbor.update_tree(i)
                
                neighbor_fitness = neighbor.get_score() + neighbor.get_overlap_area() * 1000
                if neighbor_fitness < best_fitness:
                    best = neighbor
                    best_fitness = neighbor_fitness
                    improved = True
            
            # Try small rotations
            for da in [5, -5, 10, -10]:
                neighbor = best.copy()
                neighbor.deg[i] = (neighbor.deg[i] + da) % 360
                neighbor.update_tree(i)
                
                neighbor_fitness = neighbor.get_score() + neighbor.get_overlap_area() * 1000
                if neighbor_fitness < best_fitness:
                    best = neighbor
                    best_fitness = neighbor_fitness
                    improved = True
        
        if not improved:
            break
    
    return best

def optimize_n(n, num_starts=20, sa_iterations=3000):
    """Optimize configuration for N trees using multiple random starts."""
    best = None
    best_score = float('inf')
    best_valid = None
    best_valid_score = float('inf')
    
    for start in range(num_starts):
        # Random initial configuration
        config = random_configuration(n)
        
        # Run SA
        config = simulated_annealing(config, iterations=sa_iterations)
        
        # Local search
        config = local_search(config, max_iter=200)
        
        score = config.get_score()
        has_overlap = config.has_overlap()
        
        if not has_overlap and score < best_valid_score:
            best_valid = config
            best_valid_score = score
        
        if score < best_score:
            best = config
            best_score = score
    
    return best_valid, best_valid_score

def main():
    print("SA from Random Starts Optimization")
    print("=" * 50)
    
    # Load baseline for comparison
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    
    results = {}
    total_improvement = 0
    
    # Focus on small N values (most room for improvement)
    for n in range(2, 16):
        print(f"\nOptimizing N={n}...")
        start_time = time.time()
        
        config, score = optimize_n(n, num_starts=30, sa_iterations=5000)
        
        elapsed = time.time() - start_time
        
        # Get baseline score
        prefix = f"{n:03d}_"
        n_rows = baseline_df[baseline_df['id'].str.startswith(prefix)]
        baseline_trees = []
        for _, row in n_rows.iterrows():
            x = float(str(row['x']).replace('s', ''))
            y = float(str(row['y']).replace('s', ''))
            deg = float(str(row['deg']).replace('s', ''))
            baseline_trees.append(create_tree(x, y, deg))
        baseline_side = get_side(baseline_trees)
        baseline_score = baseline_side**2 / n
        
        if config is not None:
            improvement = baseline_score - score
            total_improvement += improvement
            status = "✓ BETTER" if improvement > 0.0001 else "= same" if abs(improvement) < 0.0001 else "✗ worse"
            print(f"  N={n}: baseline={baseline_score:.6f}, new={score:.6f}, diff={improvement:+.6f} {status} ({elapsed:.1f}s)")
            
            results[n] = {
                'trees': [(config.x[i], config.y[i], config.deg[i]) for i in range(n)],
                'score': score
            }
        else:
            print(f"  N={n}: No valid configuration found ({elapsed:.1f}s)")
    
    print(f"\nTotal improvement for N=2-15: {total_improvement:+.6f}")
    
    return results

if __name__ == "__main__":
    results = main()
