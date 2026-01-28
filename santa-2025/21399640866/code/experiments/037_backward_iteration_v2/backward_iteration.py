"""
Backward Iteration v2 - Using current best solution (70.309159)
This is different from exp_002 which used the old baseline (70.615).

The approach:
1. Start from N=200 (best configuration)
2. For N=199, try removing each tree and keeping the best result
3. Continue backward to N=2
4. Compare with current best for each N
"""
import pandas as pd
import numpy as np
from numba import njit
import math
from shapely import Polygon
from shapely.affinity import rotate, translate
import json
import time

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlaps(xs, ys, angles):
    """Check if any trees overlap."""
    n = len(xs)
    if n <= 1:
        return False
    polygons = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    area = polygons[i].intersection(polygons[j]).area
                    if area > 1e-12:
                        return True
    return False

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    """Compute bounding box score."""
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

def strip(v):
    return float(str(v).replace("s", ""))

def main():
    print("=" * 70)
    print("Backward Iteration v2 - Using Current Best (70.309159)")
    print("=" * 70)
    
    # Load current best submission
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    # Load all configurations
    configs = {}
    baseline_scores = {}
    
    for n in range(1, 201):
        g = baseline_df[baseline_df['N'] == n]
        xs = np.array([strip(v) for v in g['x']])
        ys = np.array([strip(v) for v in g['y']])
        angles = np.array([strip(v) for v in g['deg']])
        configs[n] = (xs, ys, angles)
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
    
    print(f"Baseline total score: {sum(baseline_scores.values()):.6f}")
    
    # Backward iteration from N=200 to N=2
    improvements = []
    total_improvement = 0
    
    start_time = time.time()
    
    for n in range(199, 1, -1):
        # Get configuration from N+1
        xs_parent, ys_parent, angles_parent = configs[n+1]
        
        # Current best score for N
        current_best_score = baseline_scores[n]
        current_best_config = configs[n]
        
        # Try removing each tree from N+1 configuration
        best_removal_score = float('inf')
        best_removal_config = None
        
        for i in range(n+1):
            # Remove tree i
            xs_new = np.delete(xs_parent, i)
            ys_new = np.delete(ys_parent, i)
            angles_new = np.delete(angles_parent, i)
            
            # Check for overlaps (should be none since we're removing, not adding)
            if check_overlaps(list(xs_new), list(ys_new), list(angles_new)):
                continue
            
            # Compute score
            score = compute_bbox_score(xs_new, ys_new, angles_new, TX, TY)
            
            if score < best_removal_score:
                best_removal_score = score
                best_removal_config = (xs_new.copy(), ys_new.copy(), angles_new.copy())
        
        # Compare with current best for N
        if best_removal_config is not None and best_removal_score < current_best_score - 0.0001:
            improvement = current_best_score - best_removal_score
            improvements.append((n, improvement))
            total_improvement += improvement
            configs[n] = best_removal_config
            baseline_scores[n] = best_removal_score
            print(f"N={n:3d}: {current_best_score:.6f} -> {best_removal_score:.6f} (+{improvement:.6f}) ✓")
        
        # Progress update every 50 N values
        if n % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: N={n}, elapsed={elapsed:.1f}s, improvements={len(improvements)}")
    
    elapsed = time.time() - start_time
    new_total = sum(baseline_scores.values())
    
    print(f"\n{'='*70}")
    print(f"Backward Iteration Complete")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    print(f"  Total improvement: {total_improvement:.6f}")
    print(f"  New total score: {new_total:.6f}")
    print(f"  Original score: 70.309159")
    print(f"  Net improvement: {70.309159 - new_total:.6f}")
    print(f"{'='*70}")
    
    # Save results
    results = {
        'improvements': improvements,
        'total_improvement': total_improvement,
        'new_total_score': new_total,
        'elapsed_time': elapsed
    }
    
    with open('backward_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return improvements, configs, baseline_scores

if __name__ == "__main__":
    improvements, configs, scores = main()
