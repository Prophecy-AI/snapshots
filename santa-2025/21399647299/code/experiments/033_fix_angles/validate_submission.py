"""
Validate submission for overlaps using strict integer arithmetic.
"""

import pandas as pd
import numpy as np
import math
from shapely import Polygon
import json

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

SCALE = 10**12

def parse_s(x):
    if pd.isna(x):
        return float('nan')
    s = str(x).strip()
    if s.startswith('s'):
        s = s[1:]
    return float(s)

def get_tree_vertices(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        vertices.append((rx + x, ry + y))
    
    return vertices

def get_tree_polygon_int(x, y, angle_deg):
    vertices = get_tree_vertices(x, y, angle_deg)
    int_vertices = [(int(round(vx * SCALE)), int(round(vy * SCALE))) for vx, vy in vertices]
    return Polygon(int_vertices)

def validate_no_overlap(trees):
    """Validate no overlaps using integer arithmetic."""
    n = len(trees)
    polygons = [get_tree_polygon_int(x, y, deg) for x, y, deg in trees]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    intersection = polygons[i].intersection(polygons[j])
                    if intersection.area > 0:
                        return False, f"Trees {i} and {j} overlap"
    
    return True, "OK"

def compute_bbox(xs, ys, angles):
    all_x = []
    all_y = []
    
    for x, y, angle in zip(xs, ys, angles):
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score(xs, ys, angles, n):
    bbox = compute_bbox(xs, ys, angles)
    return bbox ** 2 / n

def main():
    print("=" * 60)
    print("VALIDATING SUBMISSION")
    print("=" * 60)
    
    # Load submission
    df = pd.read_csv('/home/submission/submission.csv')
    print(f"Loaded {len(df)} rows")
    
    # Parse values
    df['x_val'] = df['x'].apply(parse_s)
    df['y_val'] = df['y'].apply(parse_s)
    df['deg_val'] = df['deg'].apply(parse_s)
    
    # Group by N
    df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    
    # Validate each N
    all_valid = True
    failed_n = []
    total_score = 0
    
    for n in range(1, 201):
        group = df[df['n'] == n]
        if len(group) != n:
            print(f"N={n}: Wrong number of trees ({len(group)} vs {n})")
            all_valid = False
            continue
        
        trees = list(zip(group['x_val'], group['y_val'], group['deg_val']))
        
        # Validate
        ok, msg = validate_no_overlap(trees)
        if not ok:
            print(f"N={n}: OVERLAP - {msg}")
            all_valid = False
            failed_n.append(n)
        
        # Compute score
        xs = [t[0] for t in trees]
        ys = [t[1] for t in trees]
        angles = [t[2] for t in trees]
        score = compute_score(xs, ys, angles, n)
        total_score += score
    
    print(f"\nTotal score: {total_score:.6f}")
    print(f"All valid: {all_valid}")
    print(f"Failed N values: {failed_n}")
    
    # Save metrics
    metrics = {
        'cv_score': total_score,
        'all_valid': all_valid,
        'failed_n': failed_n,
        'num_failed': len(failed_n)
    }
    
    with open('/home/code/experiments/033_fix_angles/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return all_valid, total_score

if __name__ == "__main__":
    all_valid, total_score = main()
