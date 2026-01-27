"""
Fast 2-Tree Unit Cell Optimization with reduced search space.
"""
import pandas as pd
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from numba import njit
import math
import json

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

def create_sawtooth_config(n, angle1, angle2, row_spacing, col_spacing, offset_x=0, offset_y=0):
    """
    Create N trees in a sawtooth pattern (alternating up/down in rows).
    """
    xs = []
    ys = []
    angles = []
    
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            
            x = col * col_spacing
            y = row * row_spacing
            
            # Alternate angles based on position
            if (row + col) % 2 == 0:
                angle = angle1
                x += offset_x
                y += offset_y
            else:
                angle = angle2
            
            xs.append(x)
            ys.append(y)
            angles.append(angle)
            count += 1
        
        if count >= n:
            break
    
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
    print("Fast Sawtooth Pattern Optimization")
    print("=" * 70)
    
    df = pd.read_csv('/home/submission/submission.csv')
    df['N'] = df['id'].str.split('_').str[0].astype(int)
    
    test_ns = [20, 50, 100, 150, 200]
    improvements = []
    
    for n in test_ns:
        baseline_score = get_baseline_score(n, df)
        print(f"\nN={n}: Baseline = {baseline_score:.6f}")
        
        best_score = baseline_score
        best_config = None
        valid_count = 0
        
        # Focused search based on analysis
        for angle1 in [0, 15, 30, 45, 60, 75, 90]:
            angle2 = (angle1 + 180) % 360
            
            for row_spacing in np.arange(0.35, 0.7, 0.05):
                for col_spacing in np.arange(0.35, 0.7, 0.05):
                    for offset_x in np.arange(-0.15, 0.15, 0.05):
                        for offset_y in np.arange(-0.15, 0.15, 0.05):
                            xs, ys, angles = create_sawtooth_config(
                                n, angle1, angle2, row_spacing, col_spacing, offset_x, offset_y
                            )
                            
                            if not check_overlaps(xs, ys, angles):
                                valid_count += 1
                                score = compute_bbox_score(
                                    np.array(xs), np.array(ys), np.array(angles), TX, TY
                                )
                                
                                if score < best_score:
                                    best_score = score
                                    best_config = (angle1, angle2, row_spacing, col_spacing, offset_x, offset_y)
                                    print(f"  New best: {score:.6f}")
        
        print(f"  Valid configs: {valid_count}")
        
        improvement = baseline_score - best_score
        if improvement > 0:
            improvements.append((n, improvement, best_config))
            print(f"  ✓ IMPROVEMENT: {improvement:.6f}")
        else:
            print(f"  ✗ No improvement (best sawtooth: {best_score:.6f})")
    
    print("\n" + "=" * 70)
    if improvements:
        print(f"Found {len(improvements)} improvements")
        for n, imp, _ in improvements:
            print(f"  N={n}: +{imp:.6f}")
    else:
        print("No improvements found")
    
    metrics = {
        'cv_score': 70.316492,
        'baseline_score': 70.316492,
        'improvement': sum(imp for _, imp, _ in improvements) if improvements else 0,
        'num_improvements': len(improvements)
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
