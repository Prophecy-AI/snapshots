"""
Generate Diverse Solutions Using Different Algorithms

Key insight: All our snapshots converge to the same local optimum.
To find better solutions, we need to generate from scratch using
fundamentally different algorithms.

Strategies:
1. Bottom-left heuristic with different angle sets
2. Spiral placement
3. Hexagonal tessellation
4. Physics-based simulation (repelling particles)
5. Greedy best-fit placement

For each strategy, generate multiple solutions with different seeds.
Then ensemble: pick best per-N across all generated solutions.
"""

import numpy as np
from numba import njit
import pandas as pd
import time
import json
import random
from decimal import Decimal, getcontext

getcontext().prec = 30

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

@njit
def has_any_overlap(trees_x, trees_y, trees_angle, n):
    """Check if any trees overlap."""
    for i in range(n):
        if check_single_tree_overlap(trees_x, trees_y, trees_angle, n, i):
            return True
    return False

def generate_spiral_solution(n, seed=0):
    """
    Place trees in a spiral pattern from center outward.
    Uses golden angle for even distribution.
    """
    np.random.seed(seed)
    
    trees_x = np.zeros(n, dtype=np.float64)
    trees_y = np.zeros(n, dtype=np.float64)
    trees_angle = np.zeros(n, dtype=np.float64)
    
    golden_angle = 137.5  # degrees
    radius_scale = 0.4 + np.random.random() * 0.2  # Vary the scale
    
    for i in range(n):
        theta = i * golden_angle * np.pi / 180
        r = np.sqrt(i + 1) * radius_scale
        
        trees_x[i] = r * np.cos(theta)
        trees_y[i] = r * np.sin(theta)
        trees_angle[i] = (theta * 180 / np.pi + 45 + np.random.uniform(-30, 30)) % 360
    
    return trees_x, trees_y, trees_angle

def generate_hexagonal_solution(n, seed=0):
    """
    Place trees in a hexagonal grid pattern.
    """
    np.random.seed(seed)
    
    trees_x = np.zeros(n, dtype=np.float64)
    trees_y = np.zeros(n, dtype=np.float64)
    trees_angle = np.zeros(n, dtype=np.float64)
    
    spacing = 0.7 + np.random.random() * 0.3  # Vary spacing
    angle_set = [0, 45, 90, 135, 180, 225, 270, 315]
    
    cols = int(np.ceil(np.sqrt(n * 1.2)))
    
    idx = 0
    row = 0
    while idx < n:
        x_offset = (row % 2) * spacing / 2
        for col in range(cols):
            if idx >= n:
                break
            
            trees_x[idx] = col * spacing + x_offset
            trees_y[idx] = row * spacing * 0.866  # sqrt(3)/2
            trees_angle[idx] = angle_set[(row + col) % len(angle_set)]
            idx += 1
        row += 1
    
    return trees_x, trees_y, trees_angle

def generate_bottom_left_solution(n, seed=0, angle_set=None):
    """
    Bottom-left heuristic: place trees one at a time in the position
    that minimizes bounding box increase.
    """
    np.random.seed(seed)
    
    if angle_set is None:
        angle_set = [0, 45, 90, 135, 180, 225, 270, 315]
    
    trees_x = np.zeros(n, dtype=np.float64)
    trees_y = np.zeros(n, dtype=np.float64)
    trees_angle = np.zeros(n, dtype=np.float64)
    
    # Place first tree at origin
    trees_x[0] = 0
    trees_y[0] = 0
    trees_angle[0] = angle_set[0]
    
    for i in range(1, n):
        best_pos = None
        best_score = float('inf')
        
        # Get current bounding box
        current_side = compute_bbox(trees_x, trees_y, trees_angle, i)
        
        # Try many positions
        for _ in range(500):
            x = np.random.uniform(-current_side - 1, current_side + 1)
            y = np.random.uniform(-current_side - 1, current_side + 1)
            angle = angle_set[np.random.randint(len(angle_set))]
            
            trees_x[i] = x
            trees_y[i] = y
            trees_angle[i] = angle
            
            if not check_single_tree_overlap(trees_x, trees_y, trees_angle, i + 1, i):
                new_side = compute_bbox(trees_x, trees_y, trees_angle, i + 1)
                if new_side < best_score:
                    best_score = new_side
                    best_pos = (x, y, angle)
        
        if best_pos is None:
            # Expand search area
            for _ in range(1000):
                x = np.random.uniform(-current_side * 2, current_side * 2)
                y = np.random.uniform(-current_side * 2, current_side * 2)
                angle = angle_set[np.random.randint(len(angle_set))]
                
                trees_x[i] = x
                trees_y[i] = y
                trees_angle[i] = angle
                
                if not check_single_tree_overlap(trees_x, trees_y, trees_angle, i + 1, i):
                    new_side = compute_bbox(trees_x, trees_y, trees_angle, i + 1)
                    if new_side < best_score:
                        best_score = new_side
                        best_pos = (x, y, angle)
        
        if best_pos:
            trees_x[i], trees_y[i], trees_angle[i] = best_pos
    
    return trees_x, trees_y, trees_angle

def generate_diverse_solutions(n, num_solutions=20):
    """
    Generate diverse solutions using different algorithms.
    """
    solutions = []
    
    # Strategy 1: Spiral with different seeds
    for seed in range(num_solutions // 4):
        try:
            x, y, a = generate_spiral_solution(n, seed)
            if not has_any_overlap(x, y, a, n):
                score = (compute_bbox(x, y, a, n) ** 2) / n
                solutions.append(('spiral', seed, score, x.copy(), y.copy(), a.copy()))
        except:
            pass
    
    # Strategy 2: Hexagonal with different seeds
    for seed in range(num_solutions // 4):
        try:
            x, y, a = generate_hexagonal_solution(n, seed)
            if not has_any_overlap(x, y, a, n):
                score = (compute_bbox(x, y, a, n) ** 2) / n
                solutions.append(('hexagonal', seed, score, x.copy(), y.copy(), a.copy()))
        except:
            pass
    
    # Strategy 3: Bottom-left with 45° angles
    for seed in range(num_solutions // 4):
        try:
            x, y, a = generate_bottom_left_solution(n, seed, [0, 45, 90, 135, 180, 225, 270, 315])
            if not has_any_overlap(x, y, a, n):
                score = (compute_bbox(x, y, a, n) ** 2) / n
                solutions.append(('bottom_left_45', seed, score, x.copy(), y.copy(), a.copy()))
        except:
            pass
    
    # Strategy 4: Bottom-left with 30° angles
    for seed in range(num_solutions // 4):
        try:
            x, y, a = generate_bottom_left_solution(n, seed, [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
            if not has_any_overlap(x, y, a, n):
                score = (compute_bbox(x, y, a, n) ** 2) / n
                solutions.append(('bottom_left_30', seed, score, x.copy(), y.copy(), a.copy()))
        except:
            pass
    
    return solutions

def load_baseline(csv_path):
    """Load baseline solution from CSV."""
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

def test_diverse_generation(baseline_solutions, test_ns=[10, 20, 30], num_solutions=20):
    """Test diverse solution generation on small N values."""
    print("=" * 60)
    print("TESTING DIVERSE SOLUTION GENERATION")
    print(f"Parameters: num_solutions={num_solutions}")
    print("=" * 60)
    
    results = {}
    
    for n in test_ns:
        print(f"\nN={n}:")
        baseline_x, baseline_y, baseline_angle = baseline_solutions[n]
        baseline_side = compute_bbox(baseline_x, baseline_y, baseline_angle, n)
        baseline_score = (baseline_side ** 2) / n
        
        start_time = time.time()
        solutions = generate_diverse_solutions(n, num_solutions)
        elapsed = time.time() - start_time
        
        print(f"  Generated {len(solutions)} valid solutions in {elapsed:.2f}s")
        
        if solutions:
            best_solution = min(solutions, key=lambda x: x[2])
            best_strategy, best_seed, best_score, best_x, best_y, best_angle = best_solution
            
            improvement = baseline_score - best_score
            
            print(f"  Baseline score: {baseline_score:.6f}")
            print(f"  Best generated: {best_score:.6f} ({best_strategy}, seed={best_seed})")
            print(f"  Improvement: {improvement:.8f} ({improvement/baseline_score*100:.4f}%)")
            
            # Show distribution of scores
            scores = [s[2] for s in solutions]
            print(f"  Score range: {min(scores):.6f} - {max(scores):.6f}")
            
            results[n] = {
                'baseline_score': float(baseline_score),
                'best_score': float(best_score),
                'improvement': float(improvement),
                'best_strategy': best_strategy,
                'num_solutions': len(solutions),
                'time': elapsed
            }
            
            if improvement > 1e-8:
                print(f"  ✅ IMPROVED!")
            else:
                print(f"  ❌ No improvement (generated solutions are worse)")
        else:
            print(f"  ❌ No valid solutions generated")
            results[n] = {
                'baseline_score': float(baseline_score),
                'best_score': None,
                'improvement': 0,
                'num_solutions': 0,
                'time': elapsed
            }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_improvement = sum(r['improvement'] for r in results.values() if r['improvement'])
    improved_count = sum(1 for r in results.values() if r.get('improvement', 0) > 1e-8)
    
    print(f"Total improvement: {total_improvement:.8f}")
    print(f"N values improved: {improved_count}/{len(test_ns)}")
    
    return results

if __name__ == "__main__":
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_solutions = load_baseline(baseline_path)
    
    # Warm up Numba
    print("\nWarming up Numba JIT...")
    x, y, a = baseline_solutions[5]
    _ = compute_bbox(x, y, a, 5)
    _ = has_any_overlap(x, y, a, 5)
    print("JIT compilation complete.")
    
    # Test diverse generation
    test_results = test_diverse_generation(baseline_solutions, test_ns=[10, 20, 30], num_solutions=20)
    
    with open('/home/code/experiments/015_diverse_generation/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nResults saved to test_results.json")
