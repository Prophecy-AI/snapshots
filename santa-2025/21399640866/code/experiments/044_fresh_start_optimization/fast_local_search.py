"""
Fast Local Search - Use numba-based optimization for speed.
Focus on small perturbations to baseline configurations.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
import time
import json

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

@njit
def local_search_step(xs, ys, angles, tx, ty, step_size=0.01):
    """Try small perturbations to improve score."""
    n = len(xs)
    current_score = compute_bbox_score(xs, ys, angles, tx, ty)
    improved = False
    
    for i in range(n):
        # Try small position changes
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue
                
                old_x = xs[i]
                old_y = ys[i]
                xs[i] += dx
                ys[i] += dy
                
                new_score = compute_bbox_score(xs, ys, angles, tx, ty)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                else:
                    xs[i] = old_x
                    ys[i] = old_y
        
        # Try small angle changes
        for da in [-1.0, 1.0]:
            old_angle = angles[i]
            angles[i] += da
            if angles[i] < 0:
                angles[i] += 360
            if angles[i] >= 360:
                angles[i] -= 360
            
            new_score = compute_bbox_score(xs, ys, angles, tx, ty)
            if new_score < current_score - 1e-10:
                current_score = new_score
                improved = True
            else:
                angles[i] = old_angle
    
    return current_score, improved

@njit
def local_search(xs, ys, angles, tx, ty, max_iterations=1000):
    """Run local search until no improvement."""
    xs = xs.copy()
    ys = ys.copy()
    angles = angles.copy()
    
    best_score = compute_bbox_score(xs, ys, angles, tx, ty)
    
    for it in range(max_iterations):
        step_size = 0.01 * (1.0 - it / max_iterations)
        score, improved = local_search_step(xs, ys, angles, tx, ty, step_size)
        
        if not improved:
            break
        
        best_score = score
    
    return xs, ys, angles, best_score

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Fast Local Search on Baseline Configurations")
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
    
    # Run local search on all N values
    improvements = []
    new_configs = {}
    start_time = time.time()
    
    for n in range(1, 201):
        xs, ys, angles = baseline_configs[n]
        
        # Run local search
        opt_xs, opt_ys, opt_angles, score = local_search(xs, ys, angles, TX, TY, max_iterations=500)
        
        improvement = baseline_scores[n] - score
        new_configs[n] = (opt_xs, opt_ys, opt_angles)
        
        if improvement > 1e-8:
            improvements.append((n, improvement, score, baseline_scores[n]))
            print(f"N={n}: {baseline_scores[n]:.8f} -> {score:.8f} (+{improvement:.8f})")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Fast Local Search Complete")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.8f}")
        
        # Calculate new total score
        new_total = sum(compute_bbox_score(new_configs[n][0], new_configs[n][1], new_configs[n][2], TX, TY) 
                       for n in range(1, 201))
        print(f"  New total score: {new_total:.6f}")
    else:
        print("  No improvements found - baseline is at local optimum")
    
    print("="*70)
    
    # Save results
    results = {
        'improvements': [(n, imp, new_s, old_s) for n, imp, new_s, old_s in improvements],
        'total_improvement': sum(imp for _, imp, _, _ in improvements) if improvements else 0,
        'elapsed_time': elapsed
    }
    
    with open('fast_local_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return improvements, new_configs

if __name__ == "__main__":
    improvements, new_configs = main()
