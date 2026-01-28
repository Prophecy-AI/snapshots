"""
Fresh Start Optimization v2 - Use penalty-based SA that maintains no-overlap constraint.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import time
import json
from shapely.geometry import Polygon
from shapely import affinity

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
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

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def check_overlaps_shapely(xs, ys, angles):
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def compute_overlap_penalty(xs, ys, angles):
    """Compute overlap penalty using Shapely."""
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                # Add penalty proportional to intersection area
                try:
                    inter = polys[i].intersection(polys[j])
                    penalty += inter.area * 100  # Large penalty
                except:
                    penalty += 1.0
    return penalty

def generate_valid_grid_config(n, spacing=1.0):
    """Generate a grid configuration that's guaranteed to have no overlaps."""
    side = int(np.ceil(np.sqrt(n)))
    xs = []
    ys = []
    angles = []
    for i in range(n):
        row = i // side
        col = i % side
        xs.append(col * spacing)
        ys.append(row * spacing)
        angles.append(45.0)
    return np.array(xs), np.array(ys), np.array(angles)

def sa_optimize_with_penalty(xs, ys, angles, iterations=10000, initial_temp=0.5):
    """SA optimization with overlap penalty."""
    n = len(xs)
    xs = xs.copy()
    ys = ys.copy()
    angles = angles.copy()
    
    current_score = compute_bbox_score(xs, ys, angles, TX, TY)
    current_penalty = compute_overlap_penalty(xs, ys, angles)
    current_total = current_score + current_penalty
    
    best_score = current_score if current_penalty == 0 else float('inf')
    best_xs = xs.copy() if current_penalty == 0 else None
    best_ys = ys.copy() if current_penalty == 0 else None
    best_angles = angles.copy() if current_penalty == 0 else None
    
    temperature = initial_temp
    step_size = 0.05
    
    for it in range(iterations):
        # Pick random tree
        i = np.random.randint(n)
        
        # Save old values
        old_x = xs[i]
        old_y = ys[i]
        old_angle = angles[i]
        
        # Random move
        move_type = np.random.randint(3)
        if move_type == 0:
            xs[i] += np.random.uniform(-step_size, step_size)
            ys[i] += np.random.uniform(-step_size, step_size)
        elif move_type == 1:
            angles[i] += np.random.uniform(-20, 20)
            angles[i] = angles[i] % 360
        else:
            xs[i] += np.random.uniform(-step_size/2, step_size/2)
            ys[i] += np.random.uniform(-step_size/2, step_size/2)
            angles[i] += np.random.uniform(-10, 10)
            angles[i] = angles[i] % 360
        
        # Compute new score
        new_score = compute_bbox_score(xs, ys, angles, TX, TY)
        new_penalty = compute_overlap_penalty(xs, ys, angles)
        new_total = new_score + new_penalty
        
        # Accept or reject
        delta = new_total - current_total
        if delta < 0 or np.random.random() < math.exp(-delta / temperature):
            current_score = new_score
            current_penalty = new_penalty
            current_total = new_total
            
            if new_penalty == 0 and new_score < best_score:
                best_score = new_score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_angles = angles.copy()
        else:
            xs[i] = old_x
            ys[i] = old_y
            angles[i] = old_angle
        
        temperature *= 0.9998
        if it % 2000 == 0:
            step_size *= 0.9
    
    return best_xs, best_ys, best_angles, best_score

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Fresh Start Optimization v2 - Penalty-based SA")
    print("="*70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    baseline_scores = {}
    baseline_configs = {}
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs, ys, angles = df_to_arrays(g)
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
        baseline_configs[n] = (xs.copy(), ys.copy(), angles.copy())
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # Test on a few N values first
    test_ns = [15, 20, 25, 30, 35, 40]
    improvements = []
    start_time = time.time()
    
    for n in test_ns:
        print(f"\nOptimizing N={n}...")
        
        best_score = baseline_scores[n]
        best_config = None
        
        # Try multiple restarts
        for restart in range(10):
            # Start from grid
            xs, ys, angles = generate_valid_grid_config(n, spacing=1.0)
            
            # Optimize
            opt_xs, opt_ys, opt_angles, score = sa_optimize_with_penalty(
                xs, ys, angles, iterations=20000
            )
            
            if opt_xs is not None and score < best_score:
                # Verify no overlaps
                if not check_overlaps_shapely(opt_xs, opt_ys, opt_angles):
                    best_score = score
                    best_config = (opt_xs.copy(), opt_ys.copy(), opt_angles.copy())
                    print(f"  Restart {restart}: Found valid config with score {score:.6f}")
        
        improvement = baseline_scores[n] - best_score
        if improvement > 0.0001:
            improvements.append((n, improvement, best_score, baseline_scores[n], best_config))
            print(f"  ✅ IMPROVED: {baseline_scores[n]:.6f} -> {best_score:.6f} (+{improvement:.6f})")
        else:
            print(f"  No improvement: baseline={baseline_scores[n]:.6f}, best={best_score:.6f}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Fresh Start Optimization v2 Complete")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.6f}")
    else:
        print("  No improvements found")
    
    print("="*70)
    
    return improvements

if __name__ == "__main__":
    improvements = main()
