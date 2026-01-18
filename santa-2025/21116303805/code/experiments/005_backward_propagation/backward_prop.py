"""
Backward Propagation Optimizer
Start from N=200, work down to N=2:
- For each N, try removing each tree
- If resulting (N-1) config is better than stored, save it
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
import time
import copy

getcontext().prec = 30

class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self._update_polygon()
    
    def _update_polygon(self):
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
            (float(0), float(tip_y)),
            (float(top_w / 2), float(tier_1_y)),
            (float(top_w / 4), float(tier_1_y)),
            (float(mid_w / 2), float(tier_2_y)),
            (float(mid_w / 4), float(tier_2_y)),
            (float(base_w / 2), float(base_y)),
            (float(trunk_w / 2), float(base_y)),
            (float(trunk_w / 2), float(trunk_bottom_y)),
            (float(-trunk_w / 2), float(trunk_bottom_y)),
            (float(-trunk_w / 2), float(base_y)),
            (float(-base_w / 2), float(base_y)),
            (float(-mid_w / 4), float(tier_2_y)),
            (float(-mid_w / 2), float(tier_2_y)),
            (float(-top_w / 4), float(tier_1_y)),
            (float(-top_w / 2), float(tier_1_y)),
        ])

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated, xoff=float(self.center_x), yoff=float(self.center_y))
    
    def copy(self):
        return ChristmasTree(str(self.center_x), str(self.center_y), str(self.angle))

def load_trees_for_n(df, n):
    prefix = f"{n:03d}_"
    subset = df[df['id'].str.startswith(prefix)]
    trees = []
    for _, row in subset.iterrows():
        x = str(row['x']).lstrip('s')
        y = str(row['y']).lstrip('s')
        deg = str(row['deg']).lstrip('s')
        trees.append(ChristmasTree(x, y, deg))
    return trees

def has_overlap(trees):
    if len(trees) <= 1:
        return False
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].polygon.intersects(trees[j].polygon) and not trees[i].polygon.touches(trees[j].polygon):
                intersection = trees[i].polygon.intersection(trees[j].polygon)
                if intersection.area > 1e-12:
                    return True
    return False

def get_bounding_box_side(trees):
    if not trees:
        return 0
    all_coords = []
    for tree in trees:
        coords = np.array(tree.polygon.exterior.coords)
        all_coords.append(coords)
    all_coords = np.vstack(all_coords)
    x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
    y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
    return max(x_range, y_range)

def get_score_contribution(trees, n):
    """Get score contribution for n trees: side^2 / n"""
    side = get_bounding_box_side(trees)
    return side ** 2 / n

def backward_propagation(input_file, output_file):
    """Apply backward propagation to improve smaller N configurations."""
    df = pd.read_csv(input_file)
    
    # Load all configurations
    configs = {}
    sides = {}
    for n in range(1, 201):
        trees = load_trees_for_n(df, n)
        if len(trees) == n:
            configs[n] = trees
            sides[n] = get_bounding_box_side(trees)
    
    print(f"Loaded {len(configs)} configurations")
    
    # Calculate initial score
    initial_score = sum(sides[n] ** 2 / n for n in range(1, 201))
    print(f"Initial score: {initial_score:.6f}")
    
    improvements = 0
    total_improvement = 0
    
    # Backward propagation: from N=200 down to N=2
    for n in range(200, 1, -1):
        if n not in configs or (n-1) not in configs:
            continue
        
        current_n_minus_1_side = sides[n-1]
        current_n_minus_1_score = current_n_minus_1_side ** 2 / (n - 1)
        
        # Try removing each tree from config[n]
        best_candidate = None
        best_candidate_side = current_n_minus_1_side
        
        for tree_to_remove in range(n):
            # Create candidate by removing one tree
            candidate = [t.copy() for i, t in enumerate(configs[n]) if i != tree_to_remove]
            
            # Check for overlaps (should be none since we're removing a tree)
            if has_overlap(candidate):
                continue
            
            candidate_side = get_bounding_box_side(candidate)
            
            if candidate_side < best_candidate_side - 1e-12:
                best_candidate = candidate
                best_candidate_side = candidate_side
        
        # If we found a better configuration, update
        if best_candidate is not None:
            old_score = current_n_minus_1_score
            new_score = best_candidate_side ** 2 / (n - 1)
            improvement = old_score - new_score
            
            configs[n-1] = best_candidate
            sides[n-1] = best_candidate_side
            
            improvements += 1
            total_improvement += improvement
            print(f"n={n-1}: side {current_n_minus_1_side:.6f} -> {best_candidate_side:.6f}, improvement: {improvement:.6f}")
    
    # Calculate final score
    final_score = sum(sides[n] ** 2 / n for n in range(1, 201))
    print(f"\nFinal score: {final_score:.6f}")
    print(f"Total improvements: {improvements}")
    print(f"Total score improvement: {total_improvement:.6f}")
    
    # Save output
    new_rows = []
    for n in range(1, 201):
        for i, tree in enumerate(configs[n]):
            new_rows.append({
                'id': f"{n:03d}_{i}",
                'x': f"s{tree.center_x}",
                'y': f"s{tree.center_y}",
                'deg': f"s{tree.angle}"
            })
    
    df_out = pd.DataFrame(new_rows)
    df_out.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    return final_score, improvements

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "/home/code/preoptimized/santa-2025.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "/home/code/experiments/005_backward_propagation/optimized.csv"
    
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    start = time.time()
    backward_propagation(input_file, output_file)
    print(f"\nTime: {time.time() - start:.1f}s")
