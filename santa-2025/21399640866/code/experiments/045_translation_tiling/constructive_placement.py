"""
Constructive Tree-by-Tree Placement.
Build configurations by placing trees one at a time, always choosing
the position that minimizes the bounding box while avoiding overlaps.
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

def has_collision_with_new(existing_polys, new_poly):
    """Check if new polygon collides with any existing polygons."""
    for poly in existing_polys:
        if poly.intersects(new_poly) and not poly.touches(new_poly):
            return True
    return False

def construct_config(n, angle_steps=12, r_steps=20, theta_steps=24):
    """Build configuration tree-by-tree using greedy placement."""
    xs = [0.0]
    ys = [0.0]
    angles = [45.0]  # Start with optimal single-tree angle
    polys = [get_tree_polygon(0.0, 0.0, 45.0)]
    
    for i in range(1, n):
        best_score = float('inf')
        best_pos = None
        best_poly = None
        
        # Generate candidate positions around existing trees
        for existing_idx in range(len(xs)):
            ex, ey = xs[existing_idx], ys[existing_idx]
            
            for angle in np.linspace(0, 360, angle_steps, endpoint=False):
                for r in np.linspace(0.5, 2.5, r_steps):
                    for theta in np.linspace(0, 360, theta_steps, endpoint=False):
                        x = ex + r * np.cos(np.radians(theta))
                        y = ey + r * np.sin(np.radians(theta))
                        
                        new_poly = get_tree_polygon(x, y, angle)
                        
                        # Check for collision
                        if has_collision_with_new(polys, new_poly):
                            continue
                        
                        # Compute score with this tree added
                        test_xs = np.array(xs + [x])
                        test_ys = np.array(ys + [y])
                        test_angles = np.array(angles + [angle])
                        score = compute_bbox_score(test_xs, test_ys, test_angles, TX, TY)
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, angle)
                            best_poly = new_poly
        
        if best_pos is None:
            print(f"  Warning: Could not place tree {i+1}")
            return None
        
        xs.append(best_pos[0])
        ys.append(best_pos[1])
        angles.append(best_pos[2])
        polys.append(best_poly)
    
    return np.array(xs), np.array(ys), np.array(angles)

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def main():
    print("="*70)
    print("Constructive Tree-by-Tree Placement")
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
    
    # Test on small N values first (faster)
    test_ns = [5, 10, 15, 20]
    improvements = []
    start_time = time.time()
    
    for n in test_ns:
        print(f"\nConstructing N={n}...")
        t0 = time.time()
        
        result = construct_config(n, angle_steps=12, r_steps=15, theta_steps=18)
        
        if result is None:
            print(f"  Failed to construct configuration")
            continue
        
        xs, ys, angles = result
        score = compute_bbox_score(xs, ys, angles, TX, TY)
        baseline = baseline_scores[n]
        improvement = baseline - score
        
        elapsed = time.time() - t0
        print(f"  Constructed in {elapsed:.1f}s")
        print(f"  Constructed score: {score:.6f}, Baseline: {baseline:.6f}")
        
        if improvement > 0.0001:
            improvements.append((n, improvement, score, baseline))
            print(f"  ✅ IMPROVED by {improvement:.6f}")
        else:
            print(f"  No improvement (diff: {improvement:.6f})")
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Constructive Placement Complete")
    print(f"  Total elapsed time: {total_elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _ in improvements)
        print(f"  Total improvement: {total_improvement:.6f}")
    else:
        print("  No improvements found")
    
    print("="*70)
    
    return improvements

if __name__ == "__main__":
    improvements = main()
