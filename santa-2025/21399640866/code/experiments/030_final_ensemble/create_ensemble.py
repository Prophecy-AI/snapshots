"""
Create ensemble submission with per-N improvements from latest saspav.
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
    print("Creating Ensemble Submission")
    print("=" * 70)
    
    # Load current best
    baseline_df = pd.read_csv('/home/submission/submission.csv')
    baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)
    
    # Load latest saspav
    saspav_df = pd.read_csv('/home/code/data/external/saspav_latest_jan27/santa-2025.csv')
    saspav_df['N'] = saspav_df['id'].str.split('_').str[0].astype(int)
    
    # Build ensemble
    rows = []
    total_score = 0
    improvements_applied = 0
    total_improvement = 0
    
    for n in range(1, 201):
        # Baseline
        g = baseline_df[baseline_df['N'] == n].copy()
        xs = np.array([strip(v) for v in g['x']])
        ys = np.array([strip(v) for v in g['y']])
        angles = np.array([strip(v) for v in g['deg']])
        baseline_score = compute_bbox_score(xs, ys, angles, TX, TY)
        
        # Saspav
        g2 = saspav_df[saspav_df['N'] == n].copy()
        use_saspav = False
        
        if len(g2) == n:
            xs2 = np.array([strip(v) for v in g2['x']])
            ys2 = np.array([strip(v) for v in g2['y']])
            angles2 = np.array([strip(v) for v in g2['deg']])
            saspav_score = compute_bbox_score(xs2, ys2, angles2, TX, TY)
            
            if saspav_score < baseline_score - 0.0001:
                # Check for overlaps
                if not check_overlaps(list(xs2), list(ys2), list(angles2)):
                    use_saspav = True
                    improvement = baseline_score - saspav_score
                    improvements_applied += 1
                    total_improvement += improvement
                    print(f"N={n}: Using saspav ({baseline_score:.6f} -> {saspav_score:.6f}, +{improvement:.6f})")
        
        if use_saspav:
            for i in range(n):
                rows.append({
                    'id': f'{n}_{i}',
                    'x': f's{xs2[i]}',
                    'y': f's{ys2[i]}',
                    'deg': f's{angles2[i]}'
                })
            total_score += saspav_score
        else:
            for i in range(n):
                rows.append({
                    'id': f'{n}_{i}',
                    'x': f's{xs[i]}',
                    'y': f's{ys[i]}',
                    'deg': f's{angles[i]}'
                })
            total_score += baseline_score
    
    # Save submission
    result_df = pd.DataFrame(rows)
    result_df.to_csv('submission.csv', index=False)
    
    # Copy to submission folder
    result_df.to_csv('/home/submission/submission.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"Ensemble Results:")
    print(f"  Improvements applied: {improvements_applied}")
    print(f"  Total improvement: {total_improvement:.6f}")
    print(f"  New total score: {total_score:.6f}")
    print(f"  Previous best: 70.315653")
    print(f"  Improvement: {70.315653 - total_score:.6f}")
    print(f"{'='*70}")
    
    # Save metrics
    metrics = {
        'cv_score': total_score,
        'baseline_score': 70.315653,
        'improvement': total_improvement,
        'num_improvements': improvements_applied,
        'notes': f"Ensemble with latest saspav (Jan 27). Applied {improvements_applied} per-N improvements."
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
