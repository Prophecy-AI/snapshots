"""
Strict overlap detection for Santa 2025 submissions.
Uses high precision (scale_factor = 1e15) like the official Kaggle checker.
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import sys

# Set high precision
getcontext().prec = 25
scale_factor = Decimal('1e15')

# Tree geometry
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

class ChristmasTreeStrict:
    """Christmas tree with high precision polygon"""
    def __init__(self, center_x, center_y, angle):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))
        
        # Build polygon with high precision
        vertices = []
        angle_rad = float(self.angle) * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        for i in range(15):
            # Rotate and translate
            x = TX[i] * cos_a - TY[i] * sin_a + float(self.center_x)
            y = TX[i] * sin_a + TY[i] * cos_a + float(self.center_y)
            # Scale up for precision
            vertices.append((x * float(scale_factor), y * float(scale_factor)))
        
        self.polygon = Polygon(vertices)

def parse_value(val):
    """Remove 's' prefix from value"""
    if isinstance(val, str) and val.startswith('s'):
        return val[1:]
    return str(val)

def load_trees_for_n(df, n):
    """Load all trees for configuration N"""
    prefix = f"{n:03d}_"
    rows = df[df['id'].str.startswith(prefix)]
    trees = []
    for _, row in rows.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        trees.append(ChristmasTreeStrict(x, y, deg))
    return trees

def has_overlap_strict(trees):
    """
    Strict overlap detection using high precision.
    Returns (has_overlap, overlapping_pairs)
    """
    if len(trees) <= 1:
        return False, []
    
    polygons = [t.polygon for t in trees]
    overlapping_pairs = []
    
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            poly_i = polygons[i]
            poly_j = polygons[j]
            
            # Check if bounding boxes overlap first (fast check)
            if not poly_i.bounds[2] < poly_j.bounds[0] and \
               not poly_j.bounds[2] < poly_i.bounds[0] and \
               not poly_i.bounds[3] < poly_j.bounds[1] and \
               not poly_j.bounds[3] < poly_i.bounds[1]:
                
                # Check actual intersection
                if poly_i.intersects(poly_j):
                    intersection = poly_i.intersection(poly_j)
                    # Use strict check: area > 0 (not > 1e-10)
                    if intersection.area > 0:
                        overlapping_pairs.append((i, j, intersection.area))
    
    return len(overlapping_pairs) > 0, overlapping_pairs

def validate_submission(csv_path, verbose=True):
    """
    Validate a submission file for overlaps.
    Returns dict with validation results.
    """
    df = pd.read_csv(csv_path)
    
    results = {
        'valid': True,
        'overlapping_groups': [],
        'total_groups': 200,
        'checked_groups': 0
    }
    
    for n in range(1, 201):
        trees = load_trees_for_n(df, n)
        if not trees:
            if verbose:
                print(f"Warning: No trees found for N={n}")
            continue
            
        results['checked_groups'] += 1
        has_overlap, pairs = has_overlap_strict(trees)
        
        if has_overlap:
            results['valid'] = False
            results['overlapping_groups'].append({
                'n': n,
                'pairs': pairs
            })
            if verbose:
                print(f"❌ N={n:03d}: OVERLAP DETECTED! {len(pairs)} overlapping pairs")
                for i, j, area in pairs[:3]:  # Show first 3
                    print(f"    Trees {i} and {j}: intersection area = {area:.2e}")
        elif verbose and n % 50 == 0:
            print(f"✓ Checked N=1 to {n}...")
    
    if verbose:
        print("\n" + "="*60)
        if results['valid']:
            print("✓ VALIDATION PASSED: No overlaps detected!")
        else:
            print(f"❌ VALIDATION FAILED: {len(results['overlapping_groups'])} groups have overlaps")
            print(f"   Overlapping groups: {[g['n'] for g in results['overlapping_groups']]}")
        print("="*60)
    
    return results

def get_bounding_box_side(trees):
    """Get bounding box side length"""
    if not trees:
        return 0
    all_points = []
    for tree in trees:
        coords = np.asarray(tree.polygon.exterior.xy).T / float(scale_factor)
        all_points.append(coords)
    all_points = np.concatenate(all_points)
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    return max(max_coords - min_coords)

def calculate_score(csv_path):
    """Calculate total score"""
    df = pd.read_csv(csv_path)
    total = 0
    for n in range(1, 201):
        trees = load_trees_for_n(df, n)
        side = get_bounding_box_side(trees)
        total += (side ** 2) / n
    return total

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python strict_overlap_check.py <submission.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    print(f"Validating: {csv_path}")
    print("Using strict overlap detection (scale_factor = 1e15)")
    print()
    
    results = validate_submission(csv_path)
    
    if results['valid']:
        score = calculate_score(csv_path)
        print(f"\nScore: {score:.6f}")
