"""
Fast search for small N improvements using known good angle pairs.
The dense block approach uses pairs at angles θ and θ-180 which interlock well.
"""
import numpy as np
import pandas as pd
from numba import njit
import math
from shapely.geometry import Polygon
from shapely import affinity
from scipy.optimize import minimize
import time

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

def check_overlap_shapely(xs, ys, angles):
    n = len(xs)
    polys = [get_tree_polygon(xs[i], ys[i], angles[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

# Load baseline
baseline_df = pd.read_csv('/home/submission/submission.csv')
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)

baseline_scores = {}
for n in range(1, 201):
    g = baseline_df[baseline_df['N'] == n]
    xs, ys, angles = df_to_arrays(g)
    baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)

print("Baseline scores for N=2-5:")
for n in range(2, 6):
    print(f"  N={n}: {baseline_scores[n]:.6f}")

# For N=2, try the interlocking pair approach
print("\n" + "="*60)
print("Optimizing N=2 with interlocking pairs...")

def optimize_n2_pair(base_angle):
    """Optimize a pair of trees at angles base_angle and base_angle-180."""
    angle1 = base_angle
    angle2 = base_angle - 180
    
    def objective(params):
        dx, dy = params
        xs = np.array([0.0, dx])
        ys = np.array([0.0, dy])
        angles = np.array([angle1, angle2])
        
        if check_overlap_shapely(xs, ys, angles):
            return 1000.0
        
        return compute_bbox_score(xs, ys, angles, TX, TY)
    
    # Try multiple starting points
    best_score = float('inf')
    best_params = None
    
    for dx0 in np.linspace(-0.3, 0.3, 7):
        for dy0 in np.linspace(-0.8, 0.8, 17):
            try:
                result = minimize(objective, [dx0, dy0], method='Powell', 
                                options={'maxiter': 100})
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except:
                continue
    
    return best_score, best_params, angle1, angle2

# Try different base angles
best_n2_score = baseline_scores[2]
best_n2_config = None

for base_angle in range(0, 360, 10):
    score, params, a1, a2 = optimize_n2_pair(base_angle)
    if score < best_n2_score - 0.0001:
        best_n2_score = score
        best_n2_config = (params, a1, a2)
        print(f"  Found better: angle={base_angle}, score={score:.6f}")

print(f"\nBest N=2 score: {best_n2_score:.6f}")
print(f"Baseline N=2: {baseline_scores[2]:.6f}")
print(f"Improvement: {baseline_scores[2] - best_n2_score:.6f}")

# For N=3, try triangle arrangements
print("\n" + "="*60)
print("Optimizing N=3...")

def optimize_n3():
    """Optimize 3 trees."""
    best_score = baseline_scores[3]
    best_config = None
    
    # Try different angle combinations
    for a1 in range(0, 360, 30):
        for a2 in range(0, 360, 30):
            for a3 in range(0, 360, 30):
                def objective(params):
                    x1, y1, x2, y2, x3, y3 = params
                    xs = np.array([x1, x2, x3])
                    ys = np.array([y1, y2, y3])
                    angles = np.array([float(a1), float(a2), float(a3)])
                    
                    if check_overlap_shapely(xs, ys, angles):
                        return 1000.0
                    
                    return compute_bbox_score(xs, ys, angles, TX, TY)
                
                # Try a few starting points
                for _ in range(3):
                    x0 = np.random.uniform(-0.5, 0.5, 6)
                    try:
                        result = minimize(objective, x0, method='Powell',
                                        options={'maxiter': 50})
                        if result.fun < best_score - 0.0001:
                            best_score = result.fun
                            best_config = (result.x, a1, a2, a3)
                            print(f"  Found better: angles=({a1},{a2},{a3}), score={result.fun:.6f}")
                    except:
                        continue
    
    return best_score, best_config

best_n3_score, best_n3_config = optimize_n3()
print(f"\nBest N=3 score: {best_n3_score:.6f}")
print(f"Baseline N=3: {baseline_scores[3]:.6f}")
print(f"Improvement: {baseline_scores[3] - best_n3_score:.6f}")
