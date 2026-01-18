"""
Fractional Translation Optimizer
Implements micro-adjustments at 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001 step sizes
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import time

getcontext().prec = 30
scale_factor = Decimal("1e15")

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

def has_overlap_for_tree(trees, idx):
    """Check if tree at idx overlaps with any other tree."""
    if len(trees) <= 1:
        return False
    poly = trees[idx].polygon
    for j, other in enumerate(trees):
        if j != idx:
            if poly.intersects(other.polygon) and not poly.touches(other.polygon):
                intersection = poly.intersection(other.polygon)
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

def fractional_translation(trees, max_iter=100):
    """Micro-adjust tree positions to minimize bounding box."""
    frac_steps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    best_side = get_bounding_box_side(trees)
    improvements = 0
    
    for iteration in range(max_iter):
        improved = False
        for i, tree in enumerate(trees):
            for step in frac_steps:
                for dx, dy in directions:
                    # Save original position
                    ox, oy = tree.center_x, tree.center_y
                    
                    # Try micro-adjustment
                    tree.center_x = ox + Decimal(str(dx * step))
                    tree.center_y = oy + Decimal(str(dy * step))
                    tree._update_polygon()
                    
                    # Check if valid (no overlap) and better
                    if not has_overlap_for_tree(trees, i):
                        new_side = get_bounding_box_side(trees)
                        if new_side < best_side - 1e-12:
                            best_side = new_side
                            improved = True
                            improvements += 1
                            continue  # Keep the improvement
                    
                    # Revert
                    tree.center_x, tree.center_y = ox, oy
                    tree._update_polygon()
        
        if not improved:
            break
    
    return trees, best_side, improvements

def optimize_submission(input_file, output_file, max_iter=50):
    """Apply fractional translation to all N configurations."""
    df = pd.read_csv(input_file)
    
    total_improvements = 0
    total_score_improvement = 0
    
    new_rows = []
    
    for n in range(1, 201):
        trees = load_trees_for_n(df, n)
        if len(trees) != n:
            print(f"Warning: n={n} has {len(trees)} trees")
            continue
        
        original_side = get_bounding_box_side(trees)
        
        # Apply fractional translation
        trees, new_side, improvements = fractional_translation(trees, max_iter=max_iter)
        
        if improvements > 0:
            total_improvements += improvements
            score_diff = (original_side ** 2 - new_side ** 2) / n
            total_score_improvement += score_diff
            print(f"n={n}: side {original_side:.6f} -> {new_side:.6f}, {improvements} improvements")
        
        # Save the trees
        for i, tree in enumerate(trees):
            new_rows.append({
                'id': f"{n:03d}_{i}",
                'x': f"s{tree.center_x}",
                'y': f"s{tree.center_y}",
                'deg': f"s{tree.angle}"
            })
        
        if n % 20 == 0:
            print(f"Processed n={n}")
    
    # Save output
    df_out = pd.DataFrame(new_rows)
    df_out.to_csv(output_file, index=False)
    
    print(f"\nTotal improvements: {total_improvements}")
    print(f"Total score improvement: {total_score_improvement:.6f}")
    
    return total_improvements, total_score_improvement

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "/home/code/preoptimized/santa-2025.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "/home/code/experiments/004_sa_v1_parallel/optimized.csv"
    
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    start = time.time()
    optimize_submission(input_file, output_file, max_iter=30)
    print(f"\nTime: {time.time() - start:.1f}s")
