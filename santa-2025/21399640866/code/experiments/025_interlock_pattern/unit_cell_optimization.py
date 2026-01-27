"""
2-Tree Unit Cell Optimization.

Instead of optimizing N independent trees, optimize a 2-tree unit cell
(one "up" tree and one "down" tree) and tile it to create configurations.

Parameters to optimize:
- angle1: angle of first tree (up)
- angle2: angle of second tree (down) 
- dx: x offset between trees in unit cell
- dy: y offset between trees in unit cell
- cell_dx: x spacing between unit cells
- cell_dy: y spacing between unit cells
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

def check_overlaps(xs, ys, angles):
    n = len(xs)
    if n <= 1:
        return False
    polygons = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    area = polygons[i].intersection(polygons[j]).area
                    if area > 1e-10:
                        return True
    return False

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

def create_unit_cell_config(n, angle1, angle2, dx, dy, cell_dx, cell_dy):
    """
    Create N trees by tiling a 2-tree unit cell.
    
    Unit cell: Tree1 at (0,0) with angle1, Tree2 at (dx, dy) with angle2
    Cells are placed in a grid with spacing (cell_dx, cell_dy)
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
            base_x = col * cell_dx
            base_y = row * cell_dy
            
            # Tree 1 (up)
            xs.append(base_x)
            ys.append(base_y)
            angles.append(angle1)
            count += 1
            
            if count >= n:
                break
            
            # Tree 2 (down)
            xs.append(base_x + dx)
            ys.append(base_y + dy)
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
    print("2-Tree Unit Cell Optimization")
    print("=" * 70)
    
    # Load baseline
    df = pd.read_csv('/home/submission/submission.csv')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    # Test on large N values where interlock pattern is most prominent
    test_ns = [50, 100, 150, 200]
    
    improvements = []
    
    for n in test_ns:
        baseline_score = get_baseline_score(n, df)
        print(f"\nN={n}: Baseline score = {baseline_score:.6f}")
        
        best_score = baseline_score
        best_config = None
        valid_count = 0
        
        # Search unit cell parameters
        # Based on analysis: angle difference ~180, small dx/dy offsets
        for angle1 in range(0, 180, 15):
            angle2 = (angle1 + 180) % 360  # Opposite orientation
            
            for dx in np.arange(-0.3, 0.3, 0.05):
                for dy in np.arange(-0.3, 0.3, 0.05):
                    for cell_dx in np.arange(0.3, 0.8, 0.05):
                        for cell_dy in np.arange(0.3, 0.8, 0.05):
                            xs, ys, angles = create_unit_cell_config(
                                n, angle1, angle2, dx, dy, cell_dx, cell_dy
                            )
                            
                            if not check_overlaps(xs, ys, angles):
                                valid_count += 1
                                score = compute_bbox_score(
                                    np.array(xs), np.array(ys), np.array(angles), TX, TY
                                )
                                
                                if score < best_score:
                                    best_score = score
                                    best_config = (angle1, angle2, dx, dy, cell_dx, cell_dy)
                                    print(f"  New best: {score:.6f} (a1={angle1}, a2={angle2}, dx={dx:.2f}, dy={dy:.2f}, cdx={cell_dx:.2f}, cdy={cell_dy:.2f})")
        
        print(f"  Valid configs tested: {valid_count}")
        
        improvement = baseline_score - best_score
        if improvement > 0:
            improvements.append((n, improvement, best_config))
            print(f"  ✓ IMPROVEMENT: {improvement:.6f}")
        else:
            print(f"  ✗ No improvement found")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if improvements:
        print(f"Found improvements for {len(improvements)} N values:")
        total_improvement = 0
        for n, imp, config in improvements:
            print(f"  N={n}: +{imp:.6f}")
            total_improvement += imp
        print(f"Total improvement: {total_improvement:.6f}")
    else:
        print("No improvements found for any tested N value.")
    
    # Save metrics
    metrics = {
        'cv_score': 70.316492,
        'baseline_score': 70.316492,
        'improvement': sum(imp for _, imp, _ in improvements) if improvements else 0,
        'num_improvements': len(improvements),
        'notes': f"2-tree unit cell optimization tested on N={test_ns}. Found {len(improvements)} improvements."
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
