"""
Genetic Algorithm for Tree Packing
A fundamentally different approach to escape the local optimum.
"""

import numpy as np
import random
from shapely import Polygon, affinity
import pandas as pd
from typing import List, Tuple
import json

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def get_tree_polygon(x: float, y: float, angle: float) -> Polygon:
    """Create tree polygon at position (x, y) with rotation angle."""
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def get_bbox_size(trees: List[Tuple[float, float, float]]) -> float:
    """Get the maximum of width and height of bounding box."""
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))

def check_overlap(poly1: Polygon, poly2: Polygon, tolerance: float = 1e-10) -> bool:
    """Check if two polygons overlap (beyond touching)."""
    if not poly1.intersects(poly2):
        return False
    if poly1.touches(poly2):
        return False
    intersection = poly1.intersection(poly2)
    return intersection.area > tolerance

def has_any_overlap(trees: List[Tuple[float, float, float]]) -> bool:
    """Check if any trees overlap."""
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if check_overlap(polygons[i], polygons[j]):
                return True
    return False

def repair_overlaps(trees: List[Tuple[float, float, float]], max_iterations: int = 100) -> List[Tuple[float, float, float]]:
    """Try to repair overlaps by moving trees apart."""
    trees = list(trees)
    for _ in range(max_iterations):
        polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
        moved = False
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if check_overlap(polygons[i], polygons[j]):
                    # Move tree j away from tree i
                    dx = trees[j][0] - trees[i][0]
                    dy = trees[j][1] - trees[i][1]
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-6
                    # Move by small amount
                    move_dist = 0.05
                    trees[j] = (trees[j][0] + dx/dist * move_dist, 
                               trees[j][1] + dy/dist * move_dist, 
                               trees[j][2])
                    moved = True
        if not moved:
            break
    return trees

def generate_random_config(n: int, spread: float = 3.0) -> List[Tuple[float, float, float]]:
    """Generate a random configuration of n trees."""
    trees = []
    for _ in range(n):
        x = random.uniform(-spread, spread)
        y = random.uniform(-spread, spread)
        angle = random.uniform(0, 360)
        trees.append((x, y, angle))
    return trees

def evaluate(trees: List[Tuple[float, float, float]], n: int) -> float:
    """Evaluate a configuration. Returns score (lower is better)."""
    if has_any_overlap(trees):
        return float('inf')  # Invalid solution
    bbox = get_bbox_size(trees)
    return bbox ** 2 / n

def crossover(parent1: List[Tuple[float, float, float]], 
              parent2: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Crossover two parents to create a child."""
    n = len(parent1)
    # Single-point crossover
    point = random.randint(1, n-1)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(trees: List[Tuple[float, float, float]], 
           mutation_rate: float = 0.1,
           position_std: float = 0.1,
           angle_std: float = 10.0) -> List[Tuple[float, float, float]]:
    """Mutate a configuration."""
    trees = list(trees)
    for i in range(len(trees)):
        if random.random() < mutation_rate:
            x, y, angle = trees[i]
            x += random.gauss(0, position_std)
            y += random.gauss(0, position_std)
            angle += random.gauss(0, angle_std)
            angle = angle % 360
            trees[i] = (x, y, angle)
    return trees

def genetic_algorithm(n: int, 
                      population_size: int = 50, 
                      generations: int = 100,
                      elite_size: int = 5) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Genetic Algorithm for tree packing.
    Returns best configuration and its score.
    """
    # Initialize population with diverse configurations
    population = []
    for _ in range(population_size):
        config = generate_random_config(n, spread=n * 0.3)
        config = repair_overlaps(config)
        population.append(config)
    
    best_ever = None
    best_score_ever = float('inf')
    
    for gen in range(generations):
        # Evaluate fitness
        fitness = []
        for config in population:
            score = evaluate(config, n)
            fitness.append((config, score))
        
        # Sort by fitness (lower is better)
        fitness.sort(key=lambda x: x[1])
        
        # Track best
        if fitness[0][1] < best_score_ever:
            best_score_ever = fitness[0][1]
            best_ever = fitness[0][0]
            if gen % 20 == 0:
                print(f"  Gen {gen}: Best score = {best_score_ever:.6f}")
        
        # Selection: keep elite and select parents
        elite = [f[0] for f in fitness[:elite_size]]
        
        # Create new population
        new_population = elite.copy()
        
        # Generate children through crossover
        while len(new_population) < population_size:
            # Tournament selection
            tournament = random.sample(fitness[:population_size//2], 2)
            parent1 = min(tournament, key=lambda x: x[1])[0]
            tournament = random.sample(fitness[:population_size//2], 2)
            parent2 = min(tournament, key=lambda x: x[1])[0]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            # Repair overlaps
            child = repair_overlaps(child)
            
            new_population.append(child)
        
        population = new_population
    
    return best_ever, best_score_ever

def load_baseline_scores():
    """Load baseline per-N scores from exp_022."""
    df = pd.read_csv('/home/code/experiments/022_extended_cpp_optimization/optimized.csv')
    
    def parse_value(s):
        return float(s[1:]) if s.startswith('s') else float(s)
    
    scores = {}
    for n in range(1, 201):
        n_str = f"{n:03d}_"
        n_data = df[df['id'].str.startswith(n_str)]
        if len(n_data) == 0:
            continue
        
        trees = []
        for _, row in n_data.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            trees.append((x, y, deg))
        
        bbox = get_bbox_size(trees)
        scores[n] = bbox ** 2 / n
    
    return scores

if __name__ == "__main__":
    print("=== Genetic Algorithm for Tree Packing ===")
    print("Testing on small N values first...")
    
    # Load baseline scores
    baseline_scores = load_baseline_scores()
    
    # Test on small N values
    test_n_values = [10, 15, 20, 25, 30]
    results = {}
    
    for n in test_n_values:
        print(f"\nN={n}:")
        print(f"  Baseline score: {baseline_scores[n]:.6f}")
        
        # Run GA
        best_config, best_score = genetic_algorithm(
            n, 
            population_size=100, 
            generations=200,
            elite_size=10
        )
        
        results[n] = {
            'baseline': baseline_scores[n],
            'ga_score': best_score,
            'improvement': baseline_scores[n] - best_score
        }
        
        print(f"  GA score: {best_score:.6f}")
        print(f"  Improvement: {results[n]['improvement']:+.6f}")
    
    # Summary
    print("\n=== Summary ===")
    total_improvement = 0
    for n in test_n_values:
        imp = results[n]['improvement']
        total_improvement += imp
        status = "✅ IMPROVED" if imp > 0.0001 else "❌ No improvement"
        print(f"N={n}: {status} ({imp:+.6f})")
    
    print(f"\nTotal improvement on test N values: {total_improvement:+.6f}")
    
    # Save results
    with open('/home/code/experiments/043_genetic_algorithm/small_n_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to small_n_results.json")
