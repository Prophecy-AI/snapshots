"""
Fresh Start Optimization - Generate random configurations and optimize with SA.
Focus on N=11-50 where efficiency is worst (1.5x theoretical minimum).
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
    """Compute bounding box score S²/N."""
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

@njit
def get_tree_vertices(x, y, angle, tx, ty):
    """Get rotated and translated tree vertices."""
    r = angle * math.pi / 180.0
    c = math.cos(r)
    s = math.sin(r)
    vx = np.empty(len(tx))
    vy = np.empty(len(ty))
    for j in range(len(tx)):
        vx[j] = c * tx[j] - s * ty[j] + x
        vy[j] = s * tx[j] + c * ty[j] + y
    return vx, vy

def get_tree_polygon(x, y, angle):
    """Create Shapely polygon for overlap checking."""
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def check_overlaps_shapely(xs, ys, angles):
    """Check for overlaps using Shapely (accurate but slow)."""
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

@njit
def check_overlaps_fast(xs, ys, angles, tx, ty, min_dist=0.6):
    """Fast overlap check using center distance (conservative)."""
    n = len(xs)
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                return True
    return False

def generate_grid_config(n, spacing=1.0):
    """Generate a grid configuration."""
    side = int(np.ceil(np.sqrt(n)))
    xs = []
    ys = []
    angles = []
    for i in range(n):
        row = i // side
        col = i % side
        xs.append(col * spacing)
        ys.append(row * spacing)
        angles.append(45.0)  # Default angle
    return np.array(xs), np.array(ys), np.array(angles)

def generate_random_config(n, spread=1.0):
    """Generate a random configuration with spacing."""
    side = int(np.ceil(np.sqrt(n)))
    xs = []
    ys = []
    angles = []
    for i in range(n):
        row = i // side
        col = i % side
        x = col * spread + np.random.uniform(-0.2, 0.2)
        y = row * spread + np.random.uniform(-0.2, 0.2)
        angle = np.random.uniform(0, 360)
        xs.append(x)
        ys.append(y)
        angles.append(angle)
    return np.array(xs), np.array(ys), np.array(angles)

@njit
def sa_step(xs, ys, angles, tx, ty, temperature, step_size):
    """Single SA step - try a random move."""
    n = len(xs)
    i = np.random.randint(n)
    
    # Save old values
    old_x = xs[i]
    old_y = ys[i]
    old_angle = angles[i]
    
    # Random move type
    move_type = np.random.randint(3)
    if move_type == 0:  # Position move
        xs[i] += np.random.uniform(-step_size, step_size)
        ys[i] += np.random.uniform(-step_size, step_size)
    elif move_type == 1:  # Angle move
        angles[i] += np.random.uniform(-30, 30)
        if angles[i] < 0:
            angles[i] += 360
        if angles[i] >= 360:
            angles[i] -= 360
    else:  # Both
        xs[i] += np.random.uniform(-step_size/2, step_size/2)
        ys[i] += np.random.uniform(-step_size/2, step_size/2)
        angles[i] += np.random.uniform(-15, 15)
        if angles[i] < 0:
            angles[i] += 360
        if angles[i] >= 360:
            angles[i] -= 360
    
    return old_x, old_y, old_angle, i

@njit
def sa_optimize(xs, ys, angles, tx, ty, iterations=5000, initial_temp=1.0, cooling=0.9995):
    """Simulated annealing optimization (fast, no overlap check)."""
    n = len(xs)
    current_score = compute_bbox_score(xs, ys, angles, tx, ty)
    best_score = current_score
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_angles = angles.copy()
    
    temperature = initial_temp
    step_size = 0.1
    
    for it in range(iterations):
        # Try a move
        old_x, old_y, old_angle, i = sa_step(xs, ys, angles, tx, ty, temperature, step_size)
        
        new_score = compute_bbox_score(xs, ys, angles, tx, ty)
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or np.random.random() < math.exp(-delta / temperature):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_angles = angles.copy()
        else:
            # Reject - restore
            xs[i] = old_x
            ys[i] = old_y
            angles[i] = old_angle
        
        temperature *= cooling
        if it % 1000 == 0:
            step_size *= 0.95
    
    return best_xs, best_ys, best_angles, best_score

def optimize_n_fresh_start(n, num_restarts=20, sa_iterations=10000):
    """Optimize N trees from fresh random starts."""
    best_score = float('inf')
    best_config = None
    
    for restart in range(num_restarts):
        # Generate random config
        xs, ys, angles = generate_random_config(n, spread=0.9)
        
        # Run SA
        opt_xs, opt_ys, opt_angles, score = sa_optimize(
            xs, ys, angles, TX, TY, 
            iterations=sa_iterations,
            initial_temp=0.5,
            cooling=0.9998
        )
        
        # Check for overlaps (accurate)
        if check_overlaps_shapely(opt_xs, opt_ys, opt_angles):
            continue
        
        if score < best_score:
            best_score = score
            best_config = (opt_xs.copy(), opt_ys.copy(), opt_angles.copy())
    
    return best_score, best_config

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Fresh Start Optimization - N=11-50")
    print("="*70)
    
    # Load baseline
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    baseline_scores = {}
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs, ys, angles = df_to_arrays(g)
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # Focus on N=11-50 (worst efficiency range)
    improvements = []
    start_time = time.time()
    
    for n in range(11, 51):
        print(f"\nOptimizing N={n}...")
        
        # Run fresh start optimization
        best_score, best_config = optimize_n_fresh_start(
            n, 
            num_restarts=30,  # 30 random restarts
            sa_iterations=15000  # 15k SA iterations each
        )
        
        if best_config is None:
            print(f"  No valid config found (all had overlaps)")
            continue
        
        improvement = baseline_scores[n] - best_score
        
        if improvement > 0.0001:
            improvements.append((n, improvement, best_score, baseline_scores[n], best_config))
            print(f"  ✅ IMPROVED: {baseline_scores[n]:.6f} -> {best_score:.6f} (+{improvement:.6f})")
        else:
            print(f"  No improvement: {baseline_scores[n]:.6f} vs {best_score:.6f}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Fresh Start Optimization Complete")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.6f}")
        print("\nImproved N values:")
        for n, imp, new_score, old_score, _ in sorted(improvements, key=lambda x: -x[1]):
            print(f"  N={n}: {old_score:.6f} -> {new_score:.6f} (+{imp:.6f})")
    else:
        print("  No improvements found")
    
    print("="*70)
    
    # Save results
    results = {
        'improvements': [(n, imp, new_s, old_s) for n, imp, new_s, old_s, _ in improvements],
        'total_improvement': sum(imp for _, imp, _, _, _ in improvements) if improvements else 0,
        'elapsed_time': elapsed,
        'n_range': [11, 50],
        'num_restarts': 30,
        'sa_iterations': 15000
    }
    
    with open('fresh_start_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return improvements

if __name__ == "__main__":
    improvements = main()
