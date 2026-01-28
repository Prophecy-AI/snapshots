"""
Final ensemble attempt - check if latest saspav has any per-N improvements.
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
                    if area > 1e-12:
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

def strip(v):
    return float(str(v).replace("s", ""))

if __name__ == "__main__":
    print("=" * 70)
    print("Final Ensemble - Check Latest saspav for Per-N Improvements")
    print("=" * 70)
    
    # Load current best
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    # Load latest saspav
    saspav_df = pd.read_csv('/home/code/data/external/saspav_latest_jan27/santa-2025.csv')
    saspav_df['N'] = saspav_df['id'].str.split('_').str[0].astype(int)
    
    # Calculate per-N scores and find improvements
    baseline_scores = {}
    saspav_scores = {}
    improvements = []
    
    for n in range(1, 201):
        # Baseline
        g = baseline_df[baseline_df['N'] == n]
        xs = np.array([strip(v) for v in g['x']])
        ys = np.array([strip(v) for v in g['y']])
        angles = np.array([strip(v) for v in g['deg']])
        baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)
        
        # Saspav
        g2 = saspav_df[saspav_df['N'] == n]
        if len(g2) == n:
            xs2 = np.array([strip(v) for v in g2['x']])
            ys2 = np.array([strip(v) for v in g2['y']])
            angles2 = np.array([strip(v) for v in g2['deg']])
            saspav_scores[n] = compute_bbox_score(xs2, ys2, angles2, TX, TY)
            
            improvement = baseline_scores[n] - saspav_scores[n]
            if improvement > 0.0001:
                # Check for overlaps
                if not check_overlaps(list(xs2), list(ys2), list(angles2)):
                    improvements.append((n, improvement))
                    print(f"N={n}: {baseline_scores[n]:.6f} -> {saspav_scores[n]:.6f} (+{improvement:.6f})")
    
    print(f"\nTotal improvements found: {len(improvements)}")
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # Save metrics
    metrics = {
        'cv_score': baseline_total,
        'baseline_score': baseline_total,
        'improvement': sum(imp for _, imp in improvements) if improvements else 0,
        'num_improvements': len(improvements),
        'notes': f"Checked latest saspav (Jan 27). Found {len(improvements)} per-N improvements."
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
