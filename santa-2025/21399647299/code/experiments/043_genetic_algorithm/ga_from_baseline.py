"""
Hybrid GA: Start from baseline and apply mutations to explore neighborhood.
"""

import numpy as np
import random
from shapely import Polygon, affinity
import pandas as pd
from typing import List, Tuple
import json

TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def get_bbox_size(trees):
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))

def check_overlap(poly1, poly2, tolerance=1e-10):
    if not poly1.intersects(poly2):
        return False
    if poly1.touches(poly2):
        return False
    intersection = poly1.intersection(poly2)
    return intersection.area > tolerance

def has_any_overlap(trees):
    polygons = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if check_overlap(polygons[i], polygons[j]):
                return True
    return False

def evaluate(trees, n):
    if has_any_overlap(trees):
        return float('inf')
    bbox = get_bbox_size(trees)
    return bbox ** 2 / n

def mutate_config(trees, position_std=0.01, angle_std=1.0, mutation_rate=0.3):
    """Apply small mutations to a configuration."""
    trees = list(trees)
    for i in range(len(trees)):
        if random.random() < mutation_rate:
            x, y, angle = trees[i]
            x += random.gauss(0, position_std)
            y += random.gauss(0, position_std)
            angle += random.gauss(0, angle_std)
            trees[i] = (x, y, angle)
    return trees

def load_baseline_config(n):
    """Load baseline configuration for N from exp_022."""
    df = pd.read_csv('/home/code/experiments/022_extended_cpp_optimization/optimized.csv')
    
    def parse_value(s):
        return float(s[1:]) if s.startswith('s') else float(s)
    
    n_str = f"{n:03d}_"
    n_data = df[df['id'].str.startswith(n_str)]
    
    trees = []
    for _, row in n_data.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        trees.append((x, y, deg))
    
    return trees

def hybrid_ga(n, population_size=50, generations=500, elite_size=5):
    """
    Hybrid GA: Start from baseline and explore neighborhood.
    """
    # Load baseline
    baseline = load_baseline_config(n)
    baseline_score = evaluate(baseline, n)
    
    # Initialize population with mutations of baseline
    population = [baseline]
    for _ in range(population_size - 1):
        mutated = mutate_config(baseline, position_std=0.05, angle_std=5.0, mutation_rate=0.5)
        population.append(mutated)
    
    best_ever = baseline
    best_score_ever = baseline_score
    
    for gen in range(generations):
        # Evaluate
        fitness = [(config, evaluate(config, n)) for config in population]
        fitness.sort(key=lambda x: x[1])
        
        # Track best
        if fitness[0][1] < best_score_ever:
            best_score_ever = fitness[0][1]
            best_ever = fitness[0][0]
            print(f"  Gen {gen}: NEW BEST = {best_score_ever:.8f} (improvement: {baseline_score - best_score_ever:+.8f})")
        
        # Selection
        elite = [f[0] for f in fitness[:elite_size]]
        
        # Create new population
        new_population = elite.copy()
        
        while len(new_population) < population_size:
            # Select parent from top half
            parent = random.choice([f[0] for f in fitness[:population_size//2]])
            
            # Mutate with varying intensity
            intensity = random.choice([0.001, 0.005, 0.01, 0.02, 0.05])
            child = mutate_config(parent, position_std=intensity, angle_std=intensity*100, mutation_rate=0.3)
            new_population.append(child)
        
        population = new_population
    
    return best_ever, best_score_ever, baseline_score

if __name__ == "__main__":
    print("=== Hybrid GA: Starting from Baseline ===")
    
    test_n_values = [10, 20, 30, 50, 100]
    results = {}
    
    for n in test_n_values:
        print(f"\nN={n}:")
        best_config, best_score, baseline_score = hybrid_ga(n, population_size=100, generations=300)
        
        improvement = baseline_score - best_score
        results[n] = {
            'baseline': baseline_score,
            'best_score': best_score,
            'improvement': improvement
        }
        
        print(f"  Baseline: {baseline_score:.8f}")
        print(f"  Best: {best_score:.8f}")
        print(f"  Improvement: {improvement:+.8f}")
    
    # Summary
    print("\n=== Summary ===")
    total_improvement = 0
    for n in test_n_values:
        imp = results[n]['improvement']
        total_improvement += imp
        status = "✅ IMPROVED" if imp > 1e-8 else "❌ No improvement"
        print(f"N={n}: {status} ({imp:+.8f})")
    
    print(f"\nTotal improvement: {total_improvement:+.8f}")
    
    with open('hybrid_ga_results.json', 'w') as f:
        json.dump(results, f, indent=2)
