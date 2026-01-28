import numpy as np
import pandas as pd
from numba import jit
import math
import time
import random
import glob

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

@jit(nopython=True)
def get_tree_vertices_numba(x, y, deg):
    rad = deg * np.pi / 180.0
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    vertices_x = np.zeros(15)
    vertices_y = np.zeros(15)
    for i in range(15):
        vertices_x[i] = TX[i] * cos_a - TY[i] * sin_a + x
        vertices_y[i] = TX[i] * sin_a + TY[i] * cos_a + y
    return vertices_x, vertices_y

@jit(nopython=True)
def calculate_bounding_box(trees_x, trees_y, trees_deg, n):
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    
    for i in range(n):
        vx, vy = get_tree_vertices_numba(trees_x[i], trees_y[i], trees_deg[i])
        for j in range(15):
            if vx[j] < min_x: min_x = vx[j]
            if vx[j] > max_x: max_x = vx[j]
            if vy[j] < min_y: min_y = vy[j]
            if vy[j] > max_y: max_y = vy[j]
    
    return max(max_x - min_x, max_y - min_y)

@jit(nopython=True)
def point_in_polygon(px, py, poly_x, poly_y, n_vertices):
    inside = False
    j = n_vertices - 1
    for i in range(n_vertices):
        if ((poly_y[i] > py) != (poly_y[j] > py)) and \
           (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside

@jit(nopython=True)
def segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    return ccw(ax1, ay1, bx1, by1, bx2, by2) != ccw(ax2, ay2, bx1, by1, bx2, by2) and \
           ccw(ax1, ay1, ax2, ay2, bx1, by1) != ccw(ax1, ay1, ax2, ay2, bx2, by2)

@jit(nopython=True)
def polygons_overlap(vx1, vy1, vx2, vy2):
    for i in range(15):
        if point_in_polygon(vx1[i], vy1[i], vx2, vy2, 15):
            return True
    for i in range(15):
        if point_in_polygon(vx2[i], vy2[i], vx1, vy1, 15):
            return True
    for i in range(15):
        i_next = (i + 1) % 15
        for j in range(15):
            j_next = (j + 1) % 15
            if segments_intersect(vx1[i], vy1[i], vx1[i_next], vy1[i_next],
                                  vx2[j], vy2[j], vx2[j_next], vy2[j_next]):
                return True
    return False

@jit(nopython=True)
def check_any_overlap(trees_x, trees_y, trees_deg, n):
    for i in range(n):
        vx1, vy1 = get_tree_vertices_numba(trees_x[i], trees_y[i], trees_deg[i])
        for j in range(i+1, n):
            vx2, vy2 = get_tree_vertices_numba(trees_x[j], trees_y[j], trees_deg[j])
            if polygons_overlap(vx1, vy1, vx2, vy2):
                return True
    return False

def genetic_algorithm_for_n(n, baseline_x, baseline_y, baseline_deg, 
                            population_size=30, generations=100, mutation_rate=0.3):
    """
    Genetic algorithm for optimizing N trees.
    
    Key differences from SA:
    1. Maintains a POPULATION of solutions
    2. Uses CROSSOVER to combine good parts from different solutions
    3. Uses TOURNAMENT SELECTION to favor better solutions
    """
    # Initialize population with variations of baseline
    population = []
    for _ in range(population_size):
        # Create variation by small perturbations
        x = baseline_x.copy()
        y = baseline_y.copy()
        deg = baseline_deg.copy()
        
        # Random perturbation
        for i in range(n):
            x[i] += np.random.normal(0, 0.05)
            y[i] += np.random.normal(0, 0.05)
            deg[i] += np.random.normal(0, 5.0)
        
        # Only add if no overlaps
        if not check_any_overlap(x, y, deg, n):
            population.append((x.copy(), y.copy(), deg.copy()))
    
    # If population is too small, add copies of baseline
    while len(population) < population_size:
        population.append((baseline_x.copy(), baseline_y.copy(), baseline_deg.copy()))
    
    # Track best solution
    best_x = baseline_x.copy()
    best_y = baseline_y.copy()
    best_deg = baseline_deg.copy()
    best_side = calculate_bounding_box(best_x, best_y, best_deg, n)
    
    for gen in range(generations):
        # Evaluate fitness (lower side = better)
        fitness = []
        for x, y, deg in population:
            side = calculate_bounding_box(x, y, deg, n)
            fitness.append(side)
            if side < best_side:
                best_side = side
                best_x = x.copy()
                best_y = y.copy()
                best_deg = deg.copy()
        
        # Tournament selection
        new_population = []
        
        # Elitism: keep best solution
        best_idx = np.argmin(fitness)
        new_population.append(population[best_idx])
        
        while len(new_population) < population_size:
            # Tournament selection (size 3)
            candidates = random.sample(range(len(population)), min(3, len(population)))
            winner_idx = min(candidates, key=lambda i: fitness[i])
            parent1 = population[winner_idx]
            
            candidates = random.sample(range(len(population)), min(3, len(population)))
            winner_idx = min(candidates, key=lambda i: fitness[i])
            parent2 = population[winner_idx]
            
            # Crossover: for each tree, randomly pick from parent1 or parent2
            child_x = np.zeros(n)
            child_y = np.zeros(n)
            child_deg = np.zeros(n)
            
            for i in range(n):
                if random.random() < 0.5:
                    child_x[i] = parent1[0][i]
                    child_y[i] = parent1[1][i]
                    child_deg[i] = parent1[2][i]
                else:
                    child_x[i] = parent2[0][i]
                    child_y[i] = parent2[1][i]
                    child_deg[i] = parent2[2][i]
            
            # Mutation
            if random.random() < mutation_rate:
                idx = random.randint(0, n-1)
                child_x[idx] += np.random.normal(0, 0.02)
                child_y[idx] += np.random.normal(0, 0.02)
                child_deg[idx] += np.random.normal(0, 2.0)
            
            # Only add if no overlaps
            if not check_any_overlap(child_x, child_y, child_deg, n):
                new_population.append((child_x, child_y, child_deg))
            else:
                # If overlap, add a copy of parent1
                new_population.append(parent1)
        
        population = new_population
    
    return best_x, best_y, best_deg, best_side

def parse_coord(val):
    if isinstance(val, str) and val.startswith('s'):
        return float(val[1:])
    return float(val)

def parse_id(id_str):
    parts = str(id_str).split('_')
    n = int(parts[0])
    i = int(parts[1])
    return n, i

# Load exp_029 baseline
baseline_df = pd.read_csv('/home/code/experiments/029_final_ensemble_v2/submission.csv')
baseline_df['n'] = baseline_df['id'].apply(lambda x: parse_id(x)[0])
baseline_df['i'] = baseline_df['id'].apply(lambda x: parse_id(x)[1])
for col in ['x', 'y', 'deg']:
    baseline_df[col] = baseline_df[col].apply(parse_coord)

# Test genetic algorithm on small N values
print("Testing GENETIC ALGORITHM on small N values:")
print("(Starting from baseline solution)")

total_improvement = 0
for n in [10, 20, 30, 50]:
    n_df = baseline_df[baseline_df['n'] == n].sort_values('i')
    trees_x = np.array(n_df['x'].values)
    trees_y = np.array(n_df['y'].values)
    trees_deg = np.array(n_df['deg'].values)
    
    baseline_side = calculate_bounding_box(trees_x, trees_y, trees_deg, n)
    baseline_score = baseline_side**2 / n
    
    print(f"\nN={n}: baseline score = {baseline_score:.6f}")
    
    # Run genetic algorithm
    start_time = time.time()
    best_x, best_y, best_deg, best_side = genetic_algorithm_for_n(
        n, trees_x, trees_y, trees_deg,
        population_size=50, generations=200, mutation_rate=0.3
    )
    elapsed = time.time() - start_time
    
    new_score = best_side**2 / n
    improvement = baseline_score - new_score
    total_improvement += improvement
    status = "✅ BETTER" if improvement > 1e-9 else "❌ NO IMPROVEMENT"
    print(f"  GA score = {new_score:.6f} ({status}, diff={improvement:.9f})")
    print(f"  Time: {elapsed:.2f}s")

print(f"\nTotal improvement across tested N values: {total_improvement:.9f}")
