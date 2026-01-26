"""
Genetic Algorithm for Christmas Tree Packing

Key differences from SA:
1. Maintains POPULATION of solutions (not just one)
2. CROSSOVER combines good features from different solutions
3. MUTATION explores new regions
4. SELECTION keeps diverse solutions, not just the best

The goal is to explore MULTIPLE basins of attraction simultaneously.
"""

import numpy as np
from numba import njit
import pandas as pd
import time
import json
import random

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)

@njit
def rotate_vertices(tx, ty, angle_deg):
    angle_rad = angle_deg * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rx = tx * cos_a - ty * sin_a
    ry = tx * sin_a + ty * cos_a
    return rx, ry

@njit
def get_tree_vertices(x, y, angle):
    rx, ry = rotate_vertices(TX, TY, angle)
    return rx + x, ry + y

@njit
def compute_bbox(trees_x, trees_y, trees_angle, n):
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    
    for i in range(n):
        vx, vy = get_tree_vertices(trees_x[i], trees_y[i], trees_angle[i])
        for j in range(15):
            if vx[j] < min_x: min_x = vx[j]
            if vx[j] > max_x: max_x = vx[j]
            if vy[j] < min_y: min_y = vy[j]
            if vy[j] > max_y: max_y = vy[j]
    
    return max(max_x - min_x, max_y - min_y)

@njit
def point_in_polygon(px, py, poly_x, poly_y, n_vertices):
    inside = False
    j = n_vertices - 1
    for i in range(n_vertices):
        if ((poly_y[i] > py) != (poly_y[j] > py)) and \
           (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside

@njit
def segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    d1 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    d2 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3)
    d3 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    d4 = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

@njit
def polygons_overlap(vx1, vy1, vx2, vy2, n1, n2):
    for i in range(n1):
        if point_in_polygon(vx1[i], vy1[i], vx2, vy2, n2):
            return True
    for i in range(n2):
        if point_in_polygon(vx2[i], vy2[i], vx1, vy1, n1):
            return True
    for i in range(n1):
        i_next = (i + 1) % n1
        for j in range(n2):
            j_next = (j + 1) % n2
            if segments_intersect(vx1[i], vy1[i], vx1[i_next], vy1[i_next],
                                  vx2[j], vy2[j], vx2[j_next], vy2[j_next]):
                return True
    return False

@njit
def has_any_overlap(trees_x, trees_y, trees_angle, n):
    """Check if any trees overlap."""
    for i in range(n):
        vx1, vy1 = get_tree_vertices(trees_x[i], trees_y[i], trees_angle[i])
        for j in range(i + 1, n):
            vx2, vy2 = get_tree_vertices(trees_x[j], trees_y[j], trees_angle[j])
            
            # Quick bounding box check
            if max(vx1) < min(vx2) or max(vx2) < min(vx1):
                continue
            if max(vy1) < min(vy2) or max(vy2) < min(vy1):
                continue
            
            if polygons_overlap(vx1, vy1, vx2, vy2, 15, 15):
                return True
    return False

@njit
def check_single_tree_overlap(trees_x, trees_y, trees_angle, n, tree_idx):
    """Check if a single tree overlaps with any other."""
    vx1, vy1 = get_tree_vertices(trees_x[tree_idx], trees_y[tree_idx], trees_angle[tree_idx])
    
    for i in range(n):
        if i == tree_idx:
            continue
        vx2, vy2 = get_tree_vertices(trees_x[i], trees_y[i], trees_angle[i])
        
        if max(vx1) < min(vx2) or max(vx2) < min(vx1):
            continue
        if max(vy1) < min(vy2) or max(vy2) < min(vy1):
            continue
        
        if polygons_overlap(vx1, vy1, vx2, vy2, 15, 15):
            return True
    return False

def generate_random_valid_config(n, max_attempts=10000):
    """Generate a random valid configuration."""
    # Estimate area needed
    estimated_side = np.sqrt(n * 1.0)  # Rough estimate
    
    trees_x = np.zeros(n, dtype=np.float64)
    trees_y = np.zeros(n, dtype=np.float64)
    trees_angle = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            x = random.uniform(-estimated_side, estimated_side)
            y = random.uniform(-estimated_side, estimated_side)
            angle = random.uniform(0, 360)
            
            trees_x[i] = x
            trees_y[i] = y
            trees_angle[i] = angle
            
            if i == 0 or not check_single_tree_overlap(trees_x, trees_y, trees_angle, i + 1, i):
                placed = True
            
            attempts += 1
        
        if not placed:
            # Expand search area
            estimated_side *= 1.5
            attempts = 0
            while not placed and attempts < max_attempts:
                x = random.uniform(-estimated_side, estimated_side)
                y = random.uniform(-estimated_side, estimated_side)
                angle = random.uniform(0, 360)
                
                trees_x[i] = x
                trees_y[i] = y
                trees_angle[i] = angle
                
                if not check_single_tree_overlap(trees_x, trees_y, trees_angle, i + 1, i):
                    placed = True
                
                attempts += 1
    
    return trees_x, trees_y, trees_angle

def perturb_baseline(baseline_x, baseline_y, baseline_angle, n, scale=0.5):
    """Create a perturbed version of the baseline."""
    trees_x = baseline_x.copy()
    trees_y = baseline_y.copy()
    trees_angle = baseline_angle.copy()
    
    # Apply large random perturbations
    for i in range(n):
        trees_x[i] += random.uniform(-scale, scale)
        trees_y[i] += random.uniform(-scale, scale)
        trees_angle[i] += random.uniform(-90, 90)
    
    return trees_x, trees_y, trees_angle

def fix_overlaps(trees_x, trees_y, trees_angle, n, max_iterations=100):
    """Try to fix overlaps by moving overlapping trees."""
    for iteration in range(max_iterations):
        if not has_any_overlap(trees_x, trees_y, trees_angle, n):
            return trees_x, trees_y, trees_angle, True
        
        # Find and fix overlapping trees
        for i in range(n):
            if check_single_tree_overlap(trees_x, trees_y, trees_angle, n, i):
                # Move this tree slightly
                for _ in range(50):
                    dx = random.uniform(-0.1, 0.1)
                    dy = random.uniform(-0.1, 0.1)
                    da = random.uniform(-10, 10)
                    
                    old_x, old_y, old_a = trees_x[i], trees_y[i], trees_angle[i]
                    trees_x[i] += dx
                    trees_y[i] += dy
                    trees_angle[i] += da
                    
                    if not check_single_tree_overlap(trees_x, trees_y, trees_angle, n, i):
                        break
                    else:
                        trees_x[i], trees_y[i], trees_angle[i] = old_x, old_y, old_a
    
    return trees_x, trees_y, trees_angle, not has_any_overlap(trees_x, trees_y, trees_angle, n)

def crossover(parent1_x, parent1_y, parent1_angle, parent2_x, parent2_y, parent2_angle, n):
    """Combine tree positions from two parents."""
    child_x = np.zeros(n, dtype=np.float64)
    child_y = np.zeros(n, dtype=np.float64)
    child_angle = np.zeros(n, dtype=np.float64)
    
    # For each tree, randomly choose from parent1 or parent2
    for i in range(n):
        if random.random() < 0.5:
            child_x[i] = parent1_x[i]
            child_y[i] = parent1_y[i]
            child_angle[i] = parent1_angle[i]
        else:
            child_x[i] = parent2_x[i]
            child_y[i] = parent2_y[i]
            child_angle[i] = parent2_angle[i]
    
    return child_x, child_y, child_angle

def mutate(trees_x, trees_y, trees_angle, n, mutation_rate=0.1):
    """Apply random mutations."""
    for i in range(n):
        if random.random() < mutation_rate:
            trees_x[i] += random.uniform(-0.3, 0.3)
            trees_y[i] += random.uniform(-0.3, 0.3)
            trees_angle[i] += random.uniform(-45, 45)
    
    return trees_x, trees_y, trees_angle

def genetic_algorithm(n, baseline_x, baseline_y, baseline_angle, pop_size=30, generations=50, verbose=False):
    """
    Genetic Algorithm for tree packing.
    
    Key: Maintains diverse population to explore multiple basins.
    """
    baseline_side = compute_bbox(baseline_x, baseline_y, baseline_angle, n)
    baseline_score = (baseline_side ** 2) / n
    
    # Initialize population
    population = []
    scores = []
    
    # Add baseline
    population.append((baseline_x.copy(), baseline_y.copy(), baseline_angle.copy()))
    scores.append(baseline_score)
    
    # Add perturbed baselines
    for _ in range(pop_size // 2):
        px, py, pa = perturb_baseline(baseline_x, baseline_y, baseline_angle, n, scale=0.3)
        px, py, pa, valid = fix_overlaps(px, py, pa, n)
        if valid:
            side = compute_bbox(px, py, pa, n)
            score = (side ** 2) / n
            population.append((px, py, pa))
            scores.append(score)
    
    # Add random configurations
    for _ in range(pop_size - len(population)):
        try:
            rx, ry, ra = generate_random_valid_config(n)
            side = compute_bbox(rx, ry, ra, n)
            score = (side ** 2) / n
            population.append((rx, ry, ra))
            scores.append(score)
        except:
            pass
    
    if verbose:
        print(f"  Initial population size: {len(population)}")
        print(f"  Best initial score: {min(scores):.6f} (baseline: {baseline_score:.6f})")
    
    best_score = min(scores)
    best_idx = scores.index(best_score)
    best_solution = population[best_idx]
    
    # Evolution
    for gen in range(generations):
        # Selection: keep top 50%
        sorted_indices = np.argsort(scores)
        keep_size = max(len(population) // 2, 2)
        survivors = [population[i] for i in sorted_indices[:keep_size]]
        survivor_scores = [scores[i] for i in sorted_indices[:keep_size]]
        
        # Crossover and mutation
        new_population = list(survivors)
        new_scores = list(survivor_scores)
        
        while len(new_population) < pop_size:
            # Select two parents
            p1_idx = random.randint(0, len(survivors) - 1)
            p2_idx = random.randint(0, len(survivors) - 1)
            
            p1_x, p1_y, p1_a = survivors[p1_idx]
            p2_x, p2_y, p2_a = survivors[p2_idx]
            
            # Crossover
            child_x, child_y, child_a = crossover(p1_x, p1_y, p1_a, p2_x, p2_y, p2_a, n)
            
            # Mutation
            child_x, child_y, child_a = mutate(child_x, child_y, child_a, n, mutation_rate=0.1)
            
            # Fix overlaps
            child_x, child_y, child_a, valid = fix_overlaps(child_x, child_y, child_a, n)
            
            if valid:
                side = compute_bbox(child_x, child_y, child_a, n)
                score = (side ** 2) / n
                new_population.append((child_x, child_y, child_a))
                new_scores.append(score)
        
        population = new_population
        scores = new_scores
        
        # Track best
        gen_best_score = min(scores)
        if gen_best_score < best_score:
            best_score = gen_best_score
            best_idx = scores.index(best_score)
            best_solution = population[best_idx]
            if verbose:
                print(f"  Gen {gen}: NEW BEST {best_score:.8f}")
        elif verbose and gen % 10 == 0:
            print(f"  Gen {gen}: best={best_score:.8f}, pop_size={len(population)}")
    
    improvement = baseline_score - best_score
    return best_solution, best_score, improvement

def load_baseline(csv_path):
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for n in range(1, 201):
        n_df = df[df['id'].str.startswith(f'{n:03d}_')]
        trees_x = np.zeros(n, dtype=np.float64)
        trees_y = np.zeros(n, dtype=np.float64)
        trees_angle = np.zeros(n, dtype=np.float64)
        
        for idx, (_, row) in enumerate(n_df.iterrows()):
            trees_x[idx] = float(str(row['x']).replace('s', ''))
            trees_y[idx] = float(str(row['y']).replace('s', ''))
            trees_angle[idx] = float(str(row['deg']).replace('s', ''))
        
        solutions[n] = (trees_x, trees_y, trees_angle)
    
    return solutions

def test_genetic_algorithm(baseline_solutions, test_ns=[10, 20, 30], pop_size=30, generations=50):
    print("=" * 60)
    print("TESTING GENETIC ALGORITHM")
    print(f"Parameters: pop_size={pop_size}, generations={generations}")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        trees_x, trees_y, trees_angle = baseline_solutions[n]
        baseline_side = compute_bbox(trees_x, trees_y, trees_angle, n)
        baseline_score = (baseline_side ** 2) / n
        
        start_time = time.time()
        best_solution, best_score, improvement = genetic_algorithm(
            n, trees_x, trees_y, trees_angle,
            pop_size=pop_size, generations=generations, verbose=True
        )
        elapsed = time.time() - start_time
        
        results[n] = {
            'baseline_score': float(baseline_score),
            'best_score': float(best_score),
            'improvement': float(improvement),
            'time': elapsed
        }
        
        print(f"\n  RESULTS for N={n}:")
        print(f"    Baseline: {baseline_score:.8f}")
        print(f"    GA Best: {best_score:.8f}")
        print(f"    Improvement: {improvement:.10f}")
        print(f"    Time: {elapsed:.2f}s")
        
        if improvement > 1e-8:
            print(f"    ✅ IMPROVED!")
        else:
            print(f"    ❌ No improvement")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_improvement = sum(r['improvement'] for r in results.values())
    improved_count = sum(1 for r in results.values() if r['improvement'] > 1e-8)
    
    print(f"Total improvement: {total_improvement:.10f}")
    print(f"N values improved: {improved_count}/{len(test_ns)}")
    
    return results

if __name__ == "__main__":
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Warm up Numba
    print("\nWarming up Numba JIT...")
    trees_x, trees_y, trees_angle = baseline_solutions[5]
    _ = compute_bbox(trees_x, trees_y, trees_angle, 5)
    _ = has_any_overlap(trees_x, trees_y, trees_angle, 5)
    print("JIT compilation complete.")
    
    # Test GA
    test_results = test_genetic_algorithm(baseline_solutions, test_ns=[10, 20, 30], pop_size=30, generations=50)
    
    with open('/home/code/experiments/013_genetic_algorithm/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to test_results.json")
