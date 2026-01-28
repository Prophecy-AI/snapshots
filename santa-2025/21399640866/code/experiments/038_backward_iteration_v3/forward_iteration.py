"""
Forward Iteration - Try adding trees from N-1 configuration to create N.
This is the opposite of backward iteration.
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

def try_add_tree(xs_parent, ys_parent, angles_parent, n_target):
    """Try adding a tree to N-1 configuration to create N configuration."""
    # Get bounding box of parent configuration
    all_x = []
    all_y = []
    for i in range(len(xs_parent)):
        poly = get_tree_polygon(xs_parent[i], ys_parent[i], angles_parent[i])
        coords = list(poly.exterior.coords)
        all_x.extend([c[0] for c in coords])
        all_y.extend([c[1] for c in coords])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Try adding a tree at various positions around the configuration
    best_score = float('inf')
    best_config = None
    
    # Grid of positions to try
    positions = []
    for dx in np.linspace(min_x - 1, max_x + 1, 20):
        for dy in np.linspace(min_y - 1, max_y + 1, 20):
            positions.append((dx, dy))
    
    # Also try positions near existing trees
    for i in range(len(xs_parent)):
        for offset_x in [-0.5, 0, 0.5]:
            for offset_y in [-0.5, 0, 0.5]:
                positions.append((xs_parent[i] + offset_x, ys_parent[i] + offset_y))
    
    # Try different angles
    angles_to_try = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for new_x, new_y in positions:
        for new_angle in angles_to_try:
            # Create new configuration
            xs_new = np.append(xs_parent, new_x)
            ys_new = np.append(ys_parent, new_y)
            angles_new = np.append(angles_parent, new_angle)
            
            # Check for overlaps
            if check_overlaps(list(xs_new), list(ys_new), list(angles_new)):
                continue
            
            # Compute score
            score = compute_bbox_score(xs_new, ys_new, angles_new, TX, TY)
            
            if score < best_score:
                best_score = score
                best_config = (xs_new.copy(), ys_new.copy(), angles_new.copy())
    
    return best_score, best_config

def main():
    print("=" * 70)
    print("Forward Iteration - Adding trees from N-1 to create N")
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
    
    initial_total = sum(baseline_scores.values())
    print(f"Initial total score: {initial_total:.6f}")
    
    # Forward iteration from N=2 to N=50 (focus on small N first)
    improvements = []
    
    start_time = time.time()
    
    for n in range(2, 51):
        # Get configuration from N-1
        xs_parent, ys_parent, angles_parent = configs[n-1]
        
        # Current best score for N
        current_best_score = baseline_scores[n]
        
        # Try adding a tree
        new_score, new_config = try_add_tree(xs_parent, ys_parent, angles_parent, n)
        
        if new_config is not None and new_score < current_best_score - 0.0001:
            improvement = current_best_score - new_score
            improvements.append((n, improvement))
            configs[n] = new_config
            baseline_scores[n] = new_score
            print(f"N={n:3d}: {current_best_score:.6f} -> {new_score:.6f} (+{improvement:.6f}) ✓")
        
        # Progress update
        if n % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: N={n}, elapsed={elapsed:.1f}s, improvements={len(improvements)}")
    
    elapsed = time.time() - start_time
    final_total = sum(baseline_scores.values())
    total_improvement = initial_total - final_total
    
    print(f"\n{'='*70}")
    print(f"Forward Iteration Complete")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Improvements found: {len(improvements)}")
    print(f"  Total improvement: {total_improvement:.6f}")
    print(f"  Initial score: {initial_total:.6f}")
    print(f"  Final score: {final_total:.6f}")
    print(f"{'='*70}")
    
    return improvements

if __name__ == "__main__":
    improvements = main()
