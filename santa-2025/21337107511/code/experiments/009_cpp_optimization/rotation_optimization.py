"""
Rotation Optimization (fix_direction)

For each N, find the optimal rotation angle that minimizes the bounding box.
This rotates ALL trees in the group by the same angle.

From the bbox3 runner kernel.
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar
import time
import json

getcontext().prec = 30
scale_factor = Decimal('1e18')

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        
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

        initial_polygon = Polygon([
            (Decimal('0.0') * scale_factor, tip_y * scale_factor),
            (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
        ])
        
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )
    
    def clone(self):
        return ChristmasTree(
            center_x=str(self.center_x),
            center_y=str(self.center_y),
            angle=str(self.angle)
        )

def get_tree_list_side_length(tree_list):
    all_polygons = [t.polygon for t in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor

def parse_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['x'] = df['x'].str.strip('s')
    df['y'] = df['y'].str.strip('s')
    df['deg'] = df['deg'].str.strip('s')
    df[['group_id', 'item_id']] = df['id'].str.split('_', n=2, expand=True)
    
    dict_of_tree_list = {}
    dict_of_side_length = {}
    
    for group_id, group_data in df.groupby('group_id'):
        tree_list = [
            ChristmasTree(center_x=row['x'], center_y=row['y'], angle=row['deg'])
            for _, row in group_data.iterrows()
        ]
        dict_of_tree_list[group_id] = tree_list
        dict_of_side_length[group_id] = get_tree_list_side_length(tree_list)
    
    return dict_of_tree_list, dict_of_side_length

def get_total_score(dict_of_side_length):
    score = Decimal('0')
    for k, v in dict_of_side_length.items():
        score += v ** 2 / Decimal(k)
    return score

def calculate_bbox_side_at_angle(angle_deg, points):
    """Calculate bounding box side after rotating all points by angle_deg."""
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated_points = points.dot(rot_matrix_T)
    min_xy = np.min(rotated_points, axis=0)
    max_xy = np.max(rotated_points, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

def optimize_rotation(trees, angle_max=89.999, epsilon=1e-7):
    """Find optimal rotation angle for the group."""
    # Get all polygon vertices
    all_points = []
    for tree in trees:
        all_points.extend(list(tree.polygon.exterior.coords))
    points_np = np.array(all_points)
    
    # Use convex hull for efficiency
    hull_points = points_np[ConvexHull(points_np).vertices]
    initial_side = calculate_bbox_side_at_angle(0, hull_points)
    
    # Find optimal angle
    res = minimize_scalar(
        lambda a: calculate_bbox_side_at_angle(a, hull_points),
        bounds=(0.001, float(angle_max)),
        method='bounded'
    )
    
    found_angle_deg = float(res.x)
    found_side = float(res.fun)
    
    improvement = initial_side - found_side
    if improvement > float(epsilon):
        return found_side / float(scale_factor), found_angle_deg, improvement / float(scale_factor)
    else:
        return initial_side / float(scale_factor), 0.0, 0.0

def apply_rotation(trees, angle_deg):
    """Apply rotation to all trees in the group."""
    if not trees or abs(angle_deg) < 1e-12:
        return [t.clone() for t in trees]
    
    # Find rotation center (center of bounding box)
    bounds = [t.polygon.bounds for t in trees]
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    rotation_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    # Get tree centers
    points = np.array([[float(t.center_x) * float(scale_factor), 
                        float(t.center_y) * float(scale_factor)] for t in trees])
    
    # Rotate around center
    shifted = points - rotation_center
    rotated = shifted.dot(rot_matrix.T) + rotation_center
    
    # Create new trees
    rotated_trees = []
    for i in range(len(trees)):
        new_x = Decimal(rotated[i, 0]) / scale_factor
        new_y = Decimal(rotated[i, 1]) / scale_factor
        new_angle = trees[i].angle + Decimal(str(angle_deg))
        rotated_trees.append(ChristmasTree(str(new_x), str(new_y), str(new_angle)))
    
    return rotated_trees

def rotation_optimization(dict_of_tree_list, dict_of_side_length):
    """Optimize rotation for all groups."""
    improvements = []
    
    for n in range(1, 201):
        group_id = f'{n:03d}'
        if group_id not in dict_of_tree_list:
            continue
        
        trees = dict_of_tree_list[group_id]
        current_side = dict_of_side_length[group_id]
        
        # Find optimal rotation
        new_side, angle, improvement = optimize_rotation(trees)
        
        if improvement > 1e-10:
            # Apply rotation
            rotated_trees = apply_rotation(trees, angle)
            actual_new_side = get_tree_list_side_length(rotated_trees)
            
            if actual_new_side < current_side:
                old_score = float(current_side ** 2 / Decimal(n))
                new_score = float(actual_new_side ** 2 / Decimal(n))
                score_improvement = old_score - new_score
                
                improvements.append({
                    'n': n,
                    'old_side': float(current_side),
                    'new_side': float(actual_new_side),
                    'angle': angle,
                    'score_improvement': score_improvement
                })
                
                print(f"N={n}: IMPROVED! {float(current_side):.8f} -> {float(actual_new_side):.8f} (angle={angle:.4f}°)")
                
                dict_of_tree_list[group_id] = rotated_trees
                dict_of_side_length[group_id] = actual_new_side
    
    return improvements

def save_submission(dict_of_tree_list, output_path):
    rows = []
    for group_name in sorted(dict_of_tree_list.keys(), key=lambda x: int(x)):
        tree_list = dict_of_tree_list[group_name]
        for item_id, tree in enumerate(tree_list):
            rows.append({
                'id': f'{group_name}_{item_id}',
                'x': f's{tree.center_x}',
                'y': f's{tree.center_y}',
                'deg': f's{tree.angle}'
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    
    dict_of_tree_list, dict_of_side_length = parse_csv(baseline_path)
    
    initial_score = get_total_score(dict_of_side_length)
    print(f"Initial score: {float(initial_score):.6f}")
    
    print("\n" + "=" * 60)
    print("RUNNING ROTATION OPTIMIZATION")
    print("=" * 60)
    
    start_time = time.time()
    improvements = rotation_optimization(dict_of_tree_list, dict_of_side_length)
    elapsed = time.time() - start_time
    
    final_score = get_total_score(dict_of_side_length)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial score: {float(initial_score):.6f}")
    print(f"Final score: {float(final_score):.6f}")
    print(f"Total improvement: {float(initial_score - final_score):.8f}")
    print(f"Number of improvements: {len(improvements)}")
    print(f"Time: {elapsed:.2f}s")
    
    if improvements:
        total_score_improvement = sum(imp['score_improvement'] for imp in improvements)
        print(f"Sum of score improvements: {total_score_improvement:.8f}")
        
        save_submission(dict_of_tree_list, "/home/code/experiments/009_cpp_optimization/rotation_optimized.csv")
        
        with open('/home/code/experiments/009_cpp_optimization/rotation_improvements.json', 'w') as f:
            json.dump(improvements, f, indent=2)
    else:
        print("No improvements found.")
