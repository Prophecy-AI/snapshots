"""
Tree Deletion Optimization

Technique from chistyakov kernel:
For each N, try removing trees that touch the bounding box.
If removing a tree from N gives a better solution for N-1, use it.

This is a NOVEL approach that doesn't rely on random perturbations.
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
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
        
        # Create polygon with integer scaling for precision
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
    """Calculate bounding box side length."""
    all_polygons = [t.polygon for t in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor

def get_bbox_touching_tree_indices(tree_list):
    """Get indices of trees that touch the bounding box boundary."""
    if not tree_list:
        return []
    
    polys = [t.polygon for t in tree_list]
    
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    
    bbox = box(minx, miny, maxx, maxy)
    
    touching_indices = [
        i for i, poly in enumerate(polys)
        if poly.boundary.intersects(bbox.boundary)
    ]
    
    return touching_indices

def parse_csv(csv_path):
    """Parse CSV and return tree lists and side lengths."""
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
    """Calculate total score."""
    score = Decimal('0')
    for k, v in dict_of_side_length.items():
        score += v ** 2 / Decimal(k)
    return score

def tree_deletion_optimization(dict_of_tree_list, dict_of_side_length, max_depth=5):
    """
    For each N from 200 down to 3:
    - Take the N trees
    - Try removing each tree that touches the bbox
    - If the resulting (N-1) trees have a smaller bbox than the current best for N-1, update it
    """
    improvements = []
    
    for n_main in range(200, 2, -1):
        group_id_main = f'{n_main:03d}'
        
        if group_id_main not in dict_of_tree_list:
            continue
        
        candidate_tree_list = [tree.clone() for tree in dict_of_tree_list[group_id_main]]
        
        depth = 0
        while len(candidate_tree_list) > 1 and depth < max_depth:
            n_prev = len(candidate_tree_list) - 1
            group_id_prev = f'{n_prev:03d}'
            
            if group_id_prev not in dict_of_side_length:
                break
            
            best_side_length = dict_of_side_length[group_id_prev]
            best_tree_idx_to_delete = None
            best_new_side = None
            
            # Get trees touching the bbox
            touching_indices = get_bbox_touching_tree_indices(candidate_tree_list)
            
            for tree_idx in touching_indices:
                # Try deleting this tree
                candidate_short = [t.clone() for i, t in enumerate(candidate_tree_list) if i != tree_idx]
                candidate_side = get_tree_list_side_length(candidate_short)
                
                if candidate_side < best_side_length:
                    if best_new_side is None or candidate_side < best_new_side:
                        best_new_side = candidate_side
                        best_tree_idx_to_delete = tree_idx
            
            if best_tree_idx_to_delete is not None:
                # Found an improvement!
                improvement = float(best_side_length - best_new_side)
                old_score = float(best_side_length ** 2 / Decimal(n_prev))
                new_score = float(best_new_side ** 2 / Decimal(n_prev))
                score_improvement = old_score - new_score
                
                improvements.append({
                    'source_n': n_main,
                    'target_n': n_prev,
                    'old_side': float(best_side_length),
                    'new_side': float(best_new_side),
                    'side_improvement': improvement,
                    'score_improvement': score_improvement
                })
                
                print(f"N={n_prev}: IMPROVED! {float(best_side_length):.8f} -> {float(best_new_side):.8f} (from N={n_main})")
                
                # Update the solution
                del candidate_tree_list[best_tree_idx_to_delete]
                dict_of_tree_list[group_id_prev] = [t.clone() for t in candidate_tree_list]
                dict_of_side_length[group_id_prev] = best_new_side
                
                depth += 1
            else:
                break
            
            # Don't go too deep from one source
            if n_main - n_prev > max_depth:
                break
    
    return improvements

def save_submission(dict_of_tree_list, output_path):
    """Save solution to CSV."""
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
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    
    dict_of_tree_list, dict_of_side_length = parse_csv(baseline_path)
    
    initial_score = get_total_score(dict_of_side_length)
    print(f"Initial score: {float(initial_score):.6f}")
    
    # Run tree deletion optimization
    print("\n" + "=" * 60)
    print("RUNNING TREE DELETION OPTIMIZATION")
    print("=" * 60)
    
    start_time = time.time()
    improvements = tree_deletion_optimization(dict_of_tree_list, dict_of_side_length, max_depth=10)
    elapsed = time.time() - start_time
    
    # Calculate final score
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
        
        # Save improved solution
        save_submission(dict_of_tree_list, "/home/code/experiments/009_cpp_optimization/tree_deletion_optimized.csv")
        
        # Save improvements log
        with open('/home/code/experiments/009_cpp_optimization/tree_deletion_improvements.json', 'w') as f:
            json.dump(improvements, f, indent=2)
    else:
        print("No improvements found.")
