import numpy as np
import pandas as pd
from numba import njit
import math
import os
import subprocess
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon

getcontext().prec = 25

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

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

def check_overlaps(df, n):
    """Check for overlaps using Shapely with high precision."""
    g = df[df['N'] == n]
    
    scale_factor = Decimal('1e15')
    trunk_w = Decimal('0.15')
    trunk_h = Decimal('0.2')
    base_w = Decimal('0.7')
    mid_w = Decimal('0.4')
    top_w = Decimal('0.25')
    tip_y = Decimal('0.8')
    tier_1_y = Decimal('0.5')
    tier_2_y = Decimal('0.25')
    base_y = Decimal('0.0')
    trunk_bottom_y = -trunk_h
    
    polygons = []
    for _, row in g.iterrows():
        cx = Decimal(str(strip(row['x'])))
        cy = Decimal(str(strip(row['y'])))
        angle = Decimal(str(strip(row['deg'])))
        
        initial_polygon = Polygon([
            (float(Decimal('0.0') * scale_factor), float(tip_y * scale_factor)),
            (float(top_w / Decimal('2') * scale_factor), float(tier_1_y * scale_factor)),
            (float(top_w / Decimal('4') * scale_factor), float(tier_1_y * scale_factor)),
            (float(mid_w / Decimal('2') * scale_factor), float(tier_2_y * scale_factor)),
            (float(mid_w / Decimal('4') * scale_factor), float(tier_2_y * scale_factor)),
            (float(base_w / Decimal('2') * scale_factor), float(base_y * scale_factor)),
            (float(trunk_w / Decimal('2') * scale_factor), float(base_y * scale_factor)),
            (float(trunk_w / Decimal('2') * scale_factor), float(trunk_bottom_y * scale_factor)),
            (float(-(trunk_w / Decimal('2')) * scale_factor), float(trunk_bottom_y * scale_factor)),
            (float(-(trunk_w / Decimal('2')) * scale_factor), float(base_y * scale_factor)),
            (float(-(base_w / Decimal('2')) * scale_factor), float(base_y * scale_factor)),
            (float(-(mid_w / Decimal('4')) * scale_factor), float(tier_2_y * scale_factor)),
            (float(-(mid_w / Decimal('2')) * scale_factor), float(tier_2_y * scale_factor)),
            (float(-(top_w / Decimal('4')) * scale_factor), float(tier_1_y * scale_factor)),
            (float(-(top_w / Decimal('2')) * scale_factor), float(tier_1_y * scale_factor)),
        ])
        rotated = affinity.rotate(initial_polygon, float(angle), origin=(0, 0))
        translated = affinity.translate(rotated,
                                        xoff=float(cx * scale_factor),
                                        yoff=float(cy * scale_factor))
        polygons.append(translated)
    
    # Check for overlaps
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-10:  # Non-trivial overlap
                    return True, intersection.area
    return False, 0

# Load current best
baseline_df = pd.read_csv('/home/submission/submission.csv')
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)

# Calculate baseline per-N scores
baseline_scores = {}
for n in range(1, 201):
    g = baseline_df[baseline_df['N'] == n]
    xs, ys, angles = df_to_arrays(g)
    baseline_scores[n] = compute_bbox_score(xs, ys, angles, TX, TY)

baseline_total = sum(baseline_scores.values())
print(f"Baseline total: {baseline_total:.6f}")

# Find ALL CSV files recursively
result = subprocess.run(['find', '/home/nonroot/snapshots', '-name', '*.csv'], 
                       capture_output=True, text=True)
all_csvs = [f.strip() for f in result.stdout.split('\n') if f.strip()]
print(f"Found {len(all_csvs)} CSV files")

# Track best per-N across all sources WITH overlap checking
best_per_n = {n: (baseline_scores[n], 'baseline', None) for n in range(1, 201)}
valid_improvements = []

# Test a few specific N values that showed big improvements
test_ns = [24, 41, 33, 44, 49]

for csv_path in all_csvs[:100]:  # Test first 100 files
    try:
        df = pd.read_csv(csv_path)
        if 'id' not in df.columns:
            continue
        df['N'] = df['id'].str.split('_').str[0].astype(int)
        
        for n in test_ns:
            g = df[df['N'] == n]
            if len(g) != n:
                continue
            
            xs, ys, angles = df_to_arrays(g)
            score = compute_bbox_score(xs, ys, angles, TX, TY)
            
            if score < baseline_scores[n] - 0.01:  # Significant improvement
                # Check for overlaps
                has_overlap, area = check_overlaps(df, n)
                if has_overlap:
                    print(f"  N={n}: score {score:.6f} (improvement {baseline_scores[n]-score:.6f}) - OVERLAPS (area={area:.2e})")
                else:
                    print(f"  N={n}: score {score:.6f} (improvement {baseline_scores[n]-score:.6f}) - VALID!")
                    valid_improvements.append((n, baseline_scores[n] - score, csv_path))
    except Exception as e:
        continue

print(f"\nValid improvements found: {len(valid_improvements)}")
