"""
SAT-Based Validation to Match Kaggle's Behavior

The improved ensemble (CV 70.306052) passed local validation but failed Kaggle.
This experiment implements Separating Axis Theorem (SAT) validation to understand
why Kaggle rejects solutions that pass local Shapely validation.
"""

import numpy as np
import pandas as pd
from shapely import Polygon
import math
import json
from decimal import Decimal, getcontext

# Set high precision for decimal arithmetic
getcontext().prec = 50

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(x, y, angle_deg):
    """Get the vertices of a tree at position (x, y) with rotation angle_deg."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        vertices.append((rx + x, ry + y))
    
    return vertices

def get_tree_polygon(x, y, angle_deg):
    """Get Shapely polygon for a tree."""
    vertices = get_tree_vertices(x, y, angle_deg)
    return Polygon(vertices)

def sat_overlap_check(poly1_vertices, poly2_vertices):
    """
    Separating Axis Theorem (SAT) for convex polygon overlap detection.
    Returns True if polygons overlap, False if they don't.
    
    Note: The tree polygon is NOT convex, so SAT may not be exact.
    However, Kaggle likely uses a similar approach.
    """
    def get_axes(polygon):
        """Get all edge normals as potential separating axes."""
        axes = []
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # Normal to edge (perpendicular)
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-15:
                axes.append((normal[0]/length, normal[1]/length))
        return axes
    
    def project(polygon, axis):
        """Project polygon onto axis and return min/max."""
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in polygon]
        return min(dots), max(dots)
    
    def overlap_on_axis(poly1, poly2, axis):
        """Check if projections overlap on this axis."""
        min1, max1 = project(poly1, axis)
        min2, max2 = project(poly2, axis)
        # No overlap if one projection is entirely before the other
        return not (max1 < min2 or max2 < min1)
    
    # Check all axes from both polygons
    all_axes = get_axes(poly1_vertices) + get_axes(poly2_vertices)
    
    for axis in all_axes:
        if not overlap_on_axis(poly1_vertices, poly2_vertices, axis):
            return False  # Found separating axis - no overlap
    
    return True  # No separating axis found - overlap exists

def shapely_overlap_check(poly1_vertices, poly2_vertices, tolerance=1e-9):
    """
    Shapely-based overlap detection (our current local validation).
    """
    poly1 = Polygon(poly1_vertices)
    poly2 = Polygon(poly2_vertices)
    
    if poly1.intersects(poly2) and not poly1.touches(poly2):
        intersection = poly1.intersection(poly2)
        if intersection.area > tolerance:
            return True, intersection.area
    
    return False, 0.0

def validate_n_sat(trees):
    """Validate N trees using SAT."""
    n = len(trees)
    overlaps = []
    
    for i in range(n):
        for j in range(i + 1, n):
            verts_i = get_tree_vertices(trees[i][0], trees[i][1], trees[i][2])
            verts_j = get_tree_vertices(trees[j][0], trees[j][1], trees[j][2])
            
            if sat_overlap_check(verts_i, verts_j):
                overlaps.append((i, j))
    
    return overlaps

def validate_n_shapely(trees, tolerance=1e-9):
    """Validate N trees using Shapely."""
    n = len(trees)
    overlaps = []
    
    for i in range(n):
        for j in range(i + 1, n):
            verts_i = get_tree_vertices(trees[i][0], trees[i][1], trees[i][2])
            verts_j = get_tree_vertices(trees[j][0], trees[j][1], trees[j][2])
            
            has_overlap, area = shapely_overlap_check(verts_i, verts_j, tolerance)
            if has_overlap:
                overlaps.append((i, j, area))
    
    return overlaps

def load_solution(csv_path):
    """Load solution from CSV."""
    df = pd.read_csv(csv_path)
    
    solutions = {}
    for _, row in df.iterrows():
        id_parts = str(row['id']).split('_')
        n = int(id_parts[0])
        i = int(id_parts[1])
        
        x = float(str(row['x']).replace('s', ''))
        y = float(str(row['y']).replace('s', ''))
        deg = float(str(row['deg']).replace('s', ''))
        
        if n not in solutions:
            solutions[n] = []
        solutions[n].append((x, y, deg))
    
    return solutions

def compute_score(trees, n):
    """Compute score for N trees."""
    all_x = []
    all_y = []
    
    for x, y, angle in trees:
        vertices = get_tree_vertices(x, y, angle)
        for vx, vy in vertices:
            all_x.append(vx)
            all_y.append(vy)
    
    bbox = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    return bbox ** 2 / n

def main():
    print("=" * 60)
    print("SAT-BASED VALIDATION")
    print("=" * 60)
    
    # Load exp_022 (verified working on Kaggle)
    exp022_path = "/home/code/experiments/022_extended_cpp_optimization/optimized.csv"
    print(f"\nLoading exp_022 from {exp022_path}...")
    exp022 = load_solution(exp022_path)
    
    # Validate exp_022 with both methods
    print("\n" + "=" * 60)
    print("VALIDATING exp_022 (should pass - it passed Kaggle)")
    print("=" * 60)
    
    sat_failures_022 = []
    shapely_failures_022 = []
    
    for n in range(1, 201):
        if n not in exp022:
            continue
        
        trees = exp022[n]
        
        # SAT validation
        sat_overlaps = validate_n_sat(trees)
        if sat_overlaps:
            sat_failures_022.append((n, sat_overlaps))
        
        # Shapely validation (strict)
        shapely_overlaps = validate_n_shapely(trees, tolerance=1e-15)
        if shapely_overlaps:
            shapely_failures_022.append((n, shapely_overlaps))
    
    print(f"\nexp_022 SAT failures: {len(sat_failures_022)}")
    print(f"exp_022 Shapely failures (tol=1e-15): {len(shapely_failures_022)}")
    
    if sat_failures_022:
        print("\nSAT failures in exp_022:")
        for n, overlaps in sat_failures_022[:5]:
            print(f"  N={n}: {overlaps}")
    
    if shapely_failures_022:
        print("\nShapely failures in exp_022:")
        for n, overlaps in shapely_failures_022[:5]:
            print(f"  N={n}: {overlaps}")
    
    # Load current submission
    current_path = "/home/submission/submission.csv"
    print(f"\nLoading current submission from {current_path}...")
    current = load_solution(current_path)
    
    # Validate current submission
    print("\n" + "=" * 60)
    print("VALIDATING CURRENT SUBMISSION")
    print("=" * 60)
    
    sat_failures_current = []
    shapely_failures_current = []
    
    for n in range(1, 201):
        if n not in current:
            continue
        
        trees = current[n]
        
        # SAT validation
        sat_overlaps = validate_n_sat(trees)
        if sat_overlaps:
            sat_failures_current.append((n, sat_overlaps))
        
        # Shapely validation (strict)
        shapely_overlaps = validate_n_shapely(trees, tolerance=1e-15)
        if shapely_overlaps:
            shapely_failures_current.append((n, shapely_overlaps))
    
    print(f"\nCurrent submission SAT failures: {len(sat_failures_current)}")
    print(f"Current submission Shapely failures (tol=1e-15): {len(shapely_failures_current)}")
    
    if sat_failures_current:
        print("\nSAT failures in current submission:")
        for n, overlaps in sat_failures_current[:10]:
            print(f"  N={n}: {overlaps}")
    
    # Compute total score
    total_score = 0
    for n in range(1, 201):
        if n in current:
            score = compute_score(current[n], n)
            total_score += score
    
    print(f"\nTotal score: {total_score:.6f}")
    
    # Save results
    results = {
        'exp022_sat_failures': len(sat_failures_022),
        'exp022_shapely_failures': len(shapely_failures_022),
        'current_sat_failures': len(sat_failures_current),
        'current_shapely_failures': len(shapely_failures_current),
        'total_score': total_score
    }
    
    with open('/home/code/experiments/039_sat_validation/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved")
    
    return results

if __name__ == "__main__":
    results = main()
