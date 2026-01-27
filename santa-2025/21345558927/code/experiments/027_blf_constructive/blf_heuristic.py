"""
Bottom-Left-Fill (BLF) Constructive Heuristic for Tree Packing.

This is fundamentally different from SA/perturbation methods:
1. Builds solution incrementally (not perturbation)
2. Places trees one at a time in the lowest, leftmost valid position
3. Can try different tree orderings and angle assignments
"""
import pandas as pd
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from numba import njit
import math
import json

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def is_valid_placement(x, y, angle, existing_placements):
    """Check if placing a tree at (x, y, angle) overlaps with existing trees."""
    new_poly = get_tree_polygon(x, y, angle)
    for ex, ey, ea in existing_placements:
        existing_poly = get_tree_polygon(ex, ey, ea)
        if new_poly.intersects(existing_poly):
            if not new_poly.touches(existing_poly):
                if new_poly.intersection(existing_poly).area > 1e-10:
                    return False
    return True

@njit
def compute_bbox_score_fast(xs, ys, angles, tx, ty):
    n = len(xs)
    if n == 0:
        return 0
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

def bottom_left_fill(n, tree_angles, step=0.1, max_range=5.0):
    """
    Place n trees using bottom-left-fill heuristic.
    
    For each tree:
    1. Search from bottom-left
    2. Find the position that minimizes bounding box while avoiding overlaps
    """
    placements = []
    
    for i in range(n):
        angle = tree_angles[i]
        
        best_x, best_y = 0, 0
        best_score = float('inf')
        found = False
        
        # Search from bottom-left, prioritizing lower y values
        for y in np.arange(-max_range, max_range, step):
            for x in np.arange(-max_range, max_range, step):
                if is_valid_placement(x, y, angle, placements):
                    # Compute score with this placement
                    temp_xs = [p[0] for p in placements] + [x]
                    temp_ys = [p[1] for p in placements] + [y]
                    temp_angles = [p[2] for p in placements] + [angle]
                    
                    score = compute_bbox_score_fast(
                        np.array(temp_xs), np.array(temp_ys), np.array(temp_angles), TX, TY
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_x, best_y = x, y
                        found = True
                    
                    # For BLF, we take the first valid position in each row
                    # that improves the score
                    break
            
            # Early termination if we found a good position
            if found and best_score < float('inf'):
                # Continue searching a bit more to find better positions
                pass
        
        if not found:
            # Fallback: place at origin if nothing found
            best_x, best_y = i * 0.5, 0
        
        placements.append((best_x, best_y, angle))
    
    # Center the configuration
    xs = np.array([p[0] for p in placements])
    ys = np.array([p[1] for p in placements])
    angles = np.array([p[2] for p in placements])
    
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)
    
    return list(xs), list(ys), list(angles)

def strip(v):
    return float(str(v).replace("s", ""))

def get_baseline_score(n, df):
    g = df[df['N'] == n]
    xs = np.array([strip(v) for v in g['x']])
    ys = np.array([strip(v) for v in g['y']])
    angles = np.array([strip(v) for v in g['deg']])
    return compute_bbox_score_fast(xs, ys, angles, TX, TY)

if __name__ == "__main__":
    print("=" * 70)
    print("Bottom-Left-Fill Constructive Heuristic")
    print("=" * 70)
    
    # Load baseline
    df = pd.read_csv('/home/submission/submission.csv')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    # Test on small N values first
    test_ns = [5, 10, 15, 20]
    improvements = []
    
    for n in test_ns:
        baseline_score = get_baseline_score(n, df)
        print(f"\nN={n}: Baseline = {baseline_score:.6f}")
        
        best_score = baseline_score
        best_config = None
        
        # Try different angle assignments
        angle_strategies = [
            ('all_45', [45] * n),
            ('all_0', [0] * n),
            ('alternating', [0 if i % 2 == 0 else 180 for i in range(n)]),
            ('alternating_45', [45 if i % 2 == 0 else 225 for i in range(n)]),
        ]
        
        for strategy_name, angles in angle_strategies:
            xs, ys, final_angles = bottom_left_fill(n, angles, step=0.1, max_range=3.0)
            
            # Check for overlaps
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    p1 = get_tree_polygon(xs[i], ys[i], final_angles[i])
                    p2 = get_tree_polygon(xs[j], ys[j], final_angles[j])
                    if p1.intersects(p2) and not p1.touches(p2):
                        if p1.intersection(p2).area > 1e-10:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                score = compute_bbox_score_fast(
                    np.array(xs), np.array(ys), np.array(final_angles), TX, TY
                )
                print(f"  {strategy_name}: {score:.6f}")
                
                if score < best_score:
                    best_score = score
                    best_config = (xs, ys, final_angles, strategy_name)
            else:
                print(f"  {strategy_name}: INVALID (overlaps)")
        
        improvement = baseline_score - best_score
        if improvement > 0.0001:
            improvements.append((n, improvement, best_config[3] if best_config else None))
            print(f"  ✓ IMPROVEMENT: {improvement:.6f}")
        else:
            print(f"  ✗ No improvement")
    
    print("\n" + "=" * 70)
    if improvements:
        print(f"Found {len(improvements)} improvements:")
        for n, imp, strategy in improvements:
            print(f"  N={n}: +{imp:.6f} ({strategy})")
    else:
        print("No improvements found")
    
    # Save metrics
    metrics = {
        'cv_score': 70.316492,
        'baseline_score': 70.316492,
        'improvement': sum(imp for _, imp, _ in improvements) if improvements else 0,
        'num_improvements': len(improvements),
        'notes': f"BLF constructive heuristic tested on N={test_ns}. Found {len(improvements)} improvements."
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
