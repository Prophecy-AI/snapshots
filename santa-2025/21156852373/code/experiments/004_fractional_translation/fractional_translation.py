"""
Fractional Translation Post-Processing for Santa 2025
Based on jonathanchan's kernel: santa25-ensemble-sa-fractional-translation

This technique applies tiny movements (0.001 to 0.00001) in 8 directions
to squeeze out final improvements from an already-optimized solution.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import time
import sys

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

class Tree:
    def __init__(self, x, y, angle):
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)
        self.update_polygon()
    
    def update_polygon(self):
        angle_rad = self.angle * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        vertices = []
        for i in range(15):
            px = TX[i] * cos_a - TY[i] * sin_a + self.x
            py = TX[i] * sin_a + TY[i] * cos_a + self.y
            vertices.append((px, py))
        
        self.polygon = Polygon(vertices)
        self.bounds = self.polygon.bounds  # (minx, miny, maxx, maxy)

def parse_value(val):
    if isinstance(val, str) and val.startswith('s'):
        return val[1:]
    return str(val)

def load_trees_for_n(df, n):
    prefix = f"{n:03d}_"
    rows = df[df['id'].str.startswith(prefix)]
    trees = []
    for _, row in rows.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        trees.append(Tree(x, y, deg))
    return trees

def get_bounding_box_side(trees):
    """Get bounding box side length"""
    if not trees:
        return float('inf')
    
    minx = min(t.bounds[0] for t in trees)
    miny = min(t.bounds[1] for t in trees)
    maxx = max(t.bounds[2] for t in trees)
    maxy = max(t.bounds[3] for t in trees)
    
    return max(maxx - minx, maxy - miny)

def has_overlap_with_tree(trees, idx):
    """Check if tree at idx overlaps with any other tree"""
    tree = trees[idx]
    for i, other in enumerate(trees):
        if i == idx:
            continue
        # Quick bounding box check
        if (tree.bounds[2] < other.bounds[0] or other.bounds[2] < tree.bounds[0] or
            tree.bounds[3] < other.bounds[1] or other.bounds[3] < tree.bounds[1]):
            continue
        # Detailed intersection check
        if tree.polygon.intersects(other.polygon):
            intersection = tree.polygon.intersection(other.polygon)
            if intersection.area > 1e-12:
                return True
    return False

def fractional_translation(trees, n, max_iter=200, verbose=False):
    """
    Apply fractional translation to squeeze out improvements.
    
    For each tree, try tiny movements in 8 directions.
    Keep the movement if it reduces the bounding box without creating overlaps.
    """
    # Fractional steps from smallest to largest (try smallest first for fine-tuning)
    frac_steps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    
    # 8 directions: up, down, right, left, and 4 diagonals
    dx = [0, 0, 1, -1, 1, 1, -1, -1]
    dy = [1, -1, 0, 0, 1, -1, 1, -1]
    
    best_side = get_bounding_box_side(trees)
    total_improvement = 0
    
    for iteration in range(max_iter):
        improved = False
        
        for i in range(n):
            for step in frac_steps:
                for d in range(8):
                    # Save original position
                    old_x, old_y = trees[i].x, trees[i].y
                    
                    # Try movement
                    trees[i].x += dx[d] * step
                    trees[i].y += dy[d] * step
                    trees[i].update_polygon()
                    
                    # Check if valid (no overlap)
                    if not has_overlap_with_tree(trees, i):
                        new_side = get_bounding_box_side(trees)
                        if new_side < best_side - 1e-12:
                            # Keep the improvement
                            improvement = best_side - new_side
                            total_improvement += improvement
                            best_side = new_side
                            improved = True
                            if verbose:
                                print(f"  Iter {iteration}, tree {i}: step={step}, dir={d}, improvement={improvement:.9f}")
                        else:
                            # Revert
                            trees[i].x, trees[i].y = old_x, old_y
                            trees[i].update_polygon()
                    else:
                        # Revert
                        trees[i].x, trees[i].y = old_x, old_y
                        trees[i].update_polygon()
        
        if not improved:
            break
    
    return trees, best_side, total_improvement

def apply_fractional_translation_to_submission(input_path, output_path, max_iter=200, verbose=True):
    """Apply fractional translation to all N configurations in a submission"""
    df = pd.read_csv(input_path)
    
    # Calculate initial score
    initial_score = 0
    for n in range(1, 201):
        trees = load_trees_for_n(df, n)
        side = get_bounding_box_side(trees)
        initial_score += (side ** 2) / n
    
    print(f"Initial score: {initial_score:.6f}")
    print("="*60)
    
    # Apply fractional translation to each N
    improved_rows = []
    total_improvement = 0
    improved_count = 0
    
    start_time = time.time()
    
    for n in range(1, 201):
        trees = load_trees_for_n(df, n)
        original_side = get_bounding_box_side(trees)
        original_score = (original_side ** 2) / n
        
        # Apply fractional translation
        trees, new_side, improvement = fractional_translation(trees, n, max_iter=max_iter, verbose=False)
        new_score = (new_side ** 2) / n
        
        score_improvement = original_score - new_score
        if score_improvement > 1e-12:
            total_improvement += score_improvement
            improved_count += 1
            if verbose:
                print(f"N={n:3d}: {original_score:.9f} -> {new_score:.9f} (improvement: {score_improvement:.9f})")
        
        # Save the trees
        for i, tree in enumerate(trees):
            improved_rows.append({
                'id': f"{n:03d}_{i}",
                'x': f"s{tree.x}",
                'y': f"s{tree.y}",
                'deg': f"s{tree.angle}"
            })
        
        if n % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed N=1 to {n} in {elapsed:.1f}s...")
    
    # Create output dataframe
    improved_df = pd.DataFrame(improved_rows)
    improved_df.to_csv(output_path, index=False)
    
    # Calculate final score
    final_score = 0
    for n in range(1, 201):
        trees = load_trees_for_n(improved_df, n)
        side = get_bounding_box_side(trees)
        final_score += (side ** 2) / n
    
    elapsed = time.time() - start_time
    
    print("="*60)
    print(f"Final score: {final_score:.6f}")
    print(f"Total improvement: {initial_score - final_score:.6f}")
    print(f"Improved N values: {improved_count}/200")
    print(f"Time: {elapsed:.1f}s")
    
    return final_score, initial_score - final_score

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else '/home/code/submission_candidates/candidate_000.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/home/code/experiments/004_fractional_translation/improved.csv'
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    final_score, improvement = apply_fractional_translation_to_submission(
        input_path, output_path, max_iter=200, verbose=True
    )
