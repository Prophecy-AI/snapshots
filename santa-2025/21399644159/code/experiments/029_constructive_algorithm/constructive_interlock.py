"""
Constructive Algorithm with Interlock Patterns.

Build solutions from scratch using 2-angle tessellation (45° and 225°).
This is fundamentally different from optimizing existing solutions.
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

def check_overlap(x1, y1, a1, x2, y2, a2):
    """Check if two trees overlap."""
    p1 = get_tree_polygon(x1, y1, a1)
    p2 = get_tree_polygon(x2, y2, a2)
    if p1.intersects(p2):
        if not p1.touches(p2):
            return p1.intersection(p2).area > 1e-10
    return False

def check_any_overlap(xs, ys, angles):
    """Check if any trees overlap."""
    n = len(xs)
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap(xs[i], ys[i], angles[i], xs[j], ys[j], angles[j]):
                return True
    return False

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    n = len(xs)
    if n == 0:
        return float('inf')
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

def find_optimal_interlock_offset(angle1, angle2, step=0.01, max_range=1.0):
    """
    Find the optimal (dx, dy) offset for two trees to interlock tightly.
    Tree 1 at (0, 0) with angle1, Tree 2 at (dx, dy) with angle2.
    """
    best_offset = (0.5, 0)
    best_score = float('inf')
    
    for dx in np.arange(-max_range, max_range, step):
        for dy in np.arange(-max_range, max_range, step):
            if not check_overlap(0, 0, angle1, dx, dy, angle2):
                # Compute bounding box for this pair
                score = compute_bbox_score(
                    np.array([0.0, dx]), np.array([0.0, dy]), 
                    np.array([angle1, angle2]), TX, TY
                )
                if score < best_score:
                    best_score = score
                    best_offset = (dx, dy)
    
    return best_offset, best_score

def build_interlock_grid(n, angle1, angle2, dx_offset, dy_offset, cell_spacing_x, cell_spacing_y):
    """
    Build N trees using interlock unit cells in a grid pattern.
    
    Unit cell: Tree1 at (0,0) with angle1, Tree2 at (dx_offset, dy_offset) with angle2
    Cells are placed in a grid with spacing (cell_spacing_x, cell_spacing_y)
    """
    xs = []
    ys = []
    angles = []
    
    # Calculate grid dimensions
    cells_needed = (n + 1) // 2
    cols = int(np.ceil(np.sqrt(cells_needed)))
    rows = int(np.ceil(cells_needed / cols))
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            
            # Base position for this cell
            base_x = col * cell_spacing_x
            base_y = row * cell_spacing_y
            
            # Tree 1 (angle1)
            xs.append(base_x)
            ys.append(base_y)
            angles.append(angle1)
            count += 1
            
            if count >= n:
                break
            
            # Tree 2 (angle2)
            xs.append(base_x + dx_offset)
            ys.append(base_y + dy_offset)
            angles.append(angle2)
            count += 1
        
        if count >= n:
            break
    
    # Center the configuration
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)
    
    return list(xs), list(ys), angles

def strip(v):
    return float(str(v).replace("s", ""))

def get_baseline_score(n, df):
    g = df[df['N'] == n]
    xs = np.array([strip(v) for v in g['x']])
    ys = np.array([strip(v) for v in g['y']])
    angles = np.array([strip(v) for v in g['deg']])
    return compute_bbox_score(xs, ys, angles, TX, TY)

if __name__ == "__main__":
    print("=" * 70)
    print("Constructive Algorithm with Interlock Patterns")
    print("=" * 70)
    
    # Load baseline
    df = pd.read_csv('/home/submission/submission.csv')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    # Step 1: Find optimal interlock offset for 45° and 225° trees
    print("\nStep 1: Finding optimal interlock offset...")
    angle1, angle2 = 45, 225
    offset, pair_score = find_optimal_interlock_offset(angle1, angle2, step=0.02, max_range=0.8)
    print(f"  Optimal offset for {angle1}°/{angle2}° pair: dx={offset[0]:.3f}, dy={offset[1]:.3f}")
    print(f"  Pair score: {pair_score:.6f}")
    
    # Step 2: Test on small N values
    print("\nStep 2: Testing on small N values...")
    test_ns = [10, 20, 30, 50, 100]
    improvements = []
    
    for n in test_ns:
        baseline_score = get_baseline_score(n, df)
        print(f"\nN={n}: Baseline = {baseline_score:.6f}")
        
        best_score = baseline_score
        best_config = None
        
        # Try different cell spacings
        for cell_spacing_x in np.arange(0.4, 1.0, 0.1):
            for cell_spacing_y in np.arange(0.4, 1.0, 0.1):
                xs, ys, angles = build_interlock_grid(
                    n, angle1, angle2, offset[0], offset[1], 
                    cell_spacing_x, cell_spacing_y
                )
                
                # Check for overlaps
                if not check_any_overlap(xs, ys, angles):
                    score = compute_bbox_score(
                        np.array(xs), np.array(ys), np.array(angles), TX, TY
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_config = (cell_spacing_x, cell_spacing_y, xs, ys, angles)
                        print(f"  New best: {score:.6f} (spacing: {cell_spacing_x:.1f}, {cell_spacing_y:.1f})")
        
        improvement = baseline_score - best_score
        if improvement > 0.0001:
            improvements.append((n, improvement, best_config[:2] if best_config else None))
            print(f"  ✓ IMPROVEMENT: {improvement:.6f}")
        else:
            print(f"  ✗ No improvement (best: {best_score:.6f})")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if improvements:
        print(f"Found {len(improvements)} improvements:")
        total_improvement = 0
        for n, imp, spacing in improvements:
            print(f"  N={n}: +{imp:.6f} (spacing: {spacing})")
            total_improvement += imp
        print(f"Total improvement: {total_improvement:.6f}")
    else:
        print("No improvements found")
    
    # Save metrics
    metrics = {
        'cv_score': 70.316492,
        'baseline_score': 70.316492,
        'improvement': sum(imp for _, imp, _ in improvements) if improvements else 0,
        'num_improvements': len(improvements),
        'notes': f"Constructive interlock algorithm tested on N={test_ns}. Found {len(improvements)} improvements."
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
