"""
Rotation tightening (fix_direction) - rotates entire solution to minimize bounding box.
Based on the kernel yongsukprasertsuk/santa-2025-best-keeping-bbox3-runner
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar
import sys

getcontext().prec = 30

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_coord(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return Decimal(val[1:])
        return Decimal(val)
    return Decimal(str(val))

def get_tree_polygon(x, y, angle_deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, float(angle_deg), origin=(0, 0))
    poly = affinity.translate(poly, float(x), float(y))
    return poly

def calculate_bbox_side_at_angle(angle_deg, points):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated_points = points.dot(rot_matrix_T)
    min_xy = np.min(rotated_points, axis=0)
    max_xy = np.max(rotated_points, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

def optimize_rotation(trees_data, angle_max=89.999, epsilon=1e-7):
    """Find optimal rotation angle for a group of trees."""
    all_points = []
    for x, y, angle in trees_data:
        poly = get_tree_polygon(x, y, angle)
        all_points.extend(list(poly.exterior.coords))
    points_np = np.array(all_points)
    
    hull_points = points_np[ConvexHull(points_np).vertices]
    initial_side = calculate_bbox_side_at_angle(0, hull_points)
    
    res = minimize_scalar(
        lambda a: calculate_bbox_side_at_angle(a, hull_points),
        bounds=(0.001, float(angle_max)),
        method="bounded",
    )
    
    found_angle_deg = float(res.x)
    found_side = float(res.fun)
    
    improvement = initial_side - found_side
    if improvement > float(epsilon):
        return found_angle_deg, found_side
    else:
        return 0.0, initial_side

def apply_rotation(trees_data, angle_deg):
    """Apply rotation to all trees in a group."""
    if abs(angle_deg) < 1e-12:
        return trees_data
    
    # Get bounding box center
    all_points = []
    for x, y, angle in trees_data:
        poly = get_tree_polygon(x, y, angle)
        all_points.extend(list(poly.exterior.coords))
    points_np = np.array(all_points)
    
    min_x, min_y = np.min(points_np, axis=0)
    max_x, max_y = np.max(points_np, axis=0)
    rotation_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    rotated_trees = []
    for x, y, angle in trees_data:
        point = np.array([float(x), float(y)])
        shifted = point - rotation_center
        rotated = shifted.dot(rot_matrix.T) + rotation_center
        new_angle = Decimal(str(angle)) + Decimal(str(angle_deg))
        rotated_trees.append((Decimal(str(rotated[0])), Decimal(str(rotated[1])), new_angle))
    
    return rotated_trees

def compute_score(trees_data, n):
    """Compute score for a group of trees."""
    all_points = []
    for x, y, angle in trees_data:
        poly = get_tree_polygon(x, y, angle)
        all_points.extend(list(poly.exterior.coords))
    points_np = np.array(all_points)
    
    min_x, min_y = np.min(points_np, axis=0)
    max_x, max_y = np.max(points_np, axis=0)
    side = max(max_x - min_x, max_y - min_y)
    return (side ** 2) / n

def fix_direction(in_csv, out_csv, passes=1):
    """Apply rotation tightening to all groups."""
    df = pd.read_csv(in_csv)
    df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    df['x'] = df['x'].apply(parse_coord)
    df['y'] = df['y'].apply(parse_coord)
    df['deg'] = df['deg'].apply(parse_coord)
    
    total_improvement = Decimal('0')
    improved_groups = 0
    
    for pass_num in range(passes):
        for n in range(1, 201):
            n_df = df[df['n'] == n].copy()
            if len(n_df) != n:
                continue
            
            trees_data = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
            original_score = compute_score(trees_data, n)
            
            # Find optimal rotation
            best_angle, best_side = optimize_rotation(trees_data)
            
            if abs(best_angle) > 1e-6:
                rotated_trees = apply_rotation(trees_data, best_angle)
                new_score = compute_score(rotated_trees, n)
                
                if new_score < original_score - 1e-10:
                    improvement = original_score - new_score
                    total_improvement += Decimal(str(improvement))
                    improved_groups += 1
                    
                    # Update dataframe
                    for i, (idx, row) in enumerate(n_df.iterrows()):
                        df.loc[idx, 'x'] = rotated_trees[i][0]
                        df.loc[idx, 'y'] = rotated_trees[i][1]
                        df.loc[idx, 'deg'] = rotated_trees[i][2]
    
    # Write output
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'id': row['id'],
            'x': f"s{row['x']}",
            'y': f"s{row['y']}",
            'deg': f"s{row['deg']}"
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    
    # Compute final score
    final_score = Decimal('0')
    for n in range(1, 201):
        n_df = df[df['n'] == n]
        if len(n_df) == n:
            trees_data = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
            final_score += Decimal(str(compute_score(trees_data, n)))
    
    print(f"fix_direction: {improved_groups} groups improved, total improvement: {total_improvement:.10f}")
    print(f"Final score: {final_score:.10f}")
    return float(final_score)

if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "submission_fixed.csv"
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    fix_direction(in_csv, out_csv, passes)
