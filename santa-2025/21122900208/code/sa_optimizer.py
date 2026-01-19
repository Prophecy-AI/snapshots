import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from decimal import Decimal, getcontext
import math
import random
import time
import sys

# Set precision
getcontext().prec = 30
scale_factor = Decimal("1")

class ChristmasTree:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))
        self.polygon = self._create_polygon()

    def _create_polygon(self):
        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w  = Decimal("0.4")
        top_w  = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal("0.0") * scale_factor, tip_y * scale_factor),
                (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
                (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
                (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
                (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
            ]
        )

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        return affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )

    def update(self, x, y, angle):
        self.center_x = Decimal(str(x))
        self.center_y = Decimal(str(y))
        self.angle = Decimal(str(angle))
        self.polygon = self._create_polygon()

    def clone(self):
        return ChristmasTree(self.center_x, self.center_y, self.angle)

def get_bbox_side(trees):
    if not trees: return 0.0
    # Use float for speed in SA
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for t in trees:
        bounds = t.polygon.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
        
    return max(max_x - min_x, max_y - min_y)

def check_overlap_all(trees):
    polys = [t.polygon for t in trees]
    tree_index = STRtree(polys)
    
    for i, poly in enumerate(polys):
        # query returns indices of geometries that *might* intersect
        candidates = tree_index.query(poly)
        for idx in candidates:
            if idx == i: continue
            if poly.intersects(polys[idx]) and not poly.touches(polys[idx]):
                return True
    return False

def check_overlap_one(idx, trees, tree_index=None):
    # If tree_index (STRtree) is provided, use it. 
    # Note: STRtree is immutable in shapely 2.0+, so we'd need to rebuild it or use it for static checks.
    # For SA where we move one tree, we check that one tree against all others.
    
    target = trees[idx].polygon
    # Simple check against all others (O(N)) - faster than rebuilding STRtree every step for small N
    # For large N, we might want a persistent index, but rebuilding is expensive.
    # Let's try simple loop first.
    
    for i, t in enumerate(trees):
        if i == idx: continue
        if target.intersects(t.polygon) and not target.touches(t.polygon):
            return True
    return False

def simulated_annealing(n, trees, max_steps=10000, time_limit=60, initial_temp=1.0):
    current_trees = [t.clone() for t in trees]
    current_score = get_bbox_side(current_trees)
    
    best_trees = [t.clone() for t in current_trees]
    best_score = current_score
    
    start_time = time.time()
    temp = initial_temp
    cooling_rate = 0.999
    
    print(f"N={n}: Start Score={current_score:.6f}")
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Pick random tree
        idx = random.randint(0, n-1)
        original_tree = current_trees[idx].clone()
        
        # Perturb
        # Adaptive perturbation based on temp?
        move_scale = temp * 0.5
        rot_scale = temp * 10.0
        
        new_x = float(original_tree.center_x) + random.gauss(0, move_scale)
        new_y = float(original_tree.center_y) + random.gauss(0, move_scale)
        new_angle = float(original_tree.angle) + random.gauss(0, rot_scale)
        
        current_trees[idx].update(new_x, new_y, new_angle)
        
        # Check constraints
        # Optimization: Check overlap only for the moved tree
        if check_overlap_one(idx, current_trees):
            # Invalid, revert
            current_trees[idx] = original_tree
            continue
            
        new_score = get_bbox_side(current_trees)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_trees = [t.clone() for t in current_trees]
                # print(f"  Step {step}: New Best={best_score:.6f}")
        else:
            # Revert
            current_trees[idx] = original_tree
            
        temp *= cooling_rate
        
    print(f"N={n}: End Score={best_score:.6f} (Time: {time.time()-start_time:.2f}s)")
    return best_trees, best_score

def load_submission(filepath):
    df = pd.read_csv(filepath)
    df['x'] = df['x'].astype(str).str.replace('s', '')
    df['y'] = df['y'].astype(str).str.replace('s', '')
    df['deg'] = df['deg'].astype(str).str.replace('s', '')
    return df

def save_submission(df, filepath):
    # Format back to 's' prefix
    out_df = df.copy()
    out_df['x'] = 's' + out_df['x'].astype(str)
    out_df['y'] = 's' + out_df['y'].astype(str)
    out_df['deg'] = 's' + out_df['deg'].astype(str)
    out_df.to_csv(filepath, index=False)

def main():
    input_csv = "submission.csv"
    output_csv = "submission_optimized.csv"
    
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    
    print(f"Loading {input_csv}...")
    df = load_submission(input_csv)
    
    # Identify groups
    df['group_id'] = df['id'].apply(lambda x: x.split('_')[0])
    
    # Select N to optimize
    # For testing, let's pick a few small Ns and maybe one larger one
    # Or iterate all. For this run, let's do N=1..10 to verify.
    target_groups = [f"{i:03d}" for i in range(1, 11)] 
    
    total_improvement = 0
    
    for group_id in target_groups:
        group_data = df[df['group_id'] == group_id]
        if group_data.empty: continue
        
        n = len(group_data)
        trees = []
        for _, row in group_data.iterrows():
            trees.append(ChristmasTree(row['x'], row['y'], row['deg']))
            
        initial_score = get_bbox_side(trees)**2 / n
        
        # Run SA
        optimized_trees, optimized_side = simulated_annealing(n, trees, max_steps=5000, time_limit=10)
        
        final_score = optimized_side**2 / n
        
        if final_score < initial_score:
            print(f"  Improved N={n}: {initial_score:.6f} -> {final_score:.6f}")
            total_improvement += (initial_score - final_score)
            
            # Update DataFrame
            for i, tree in enumerate(optimized_trees):
                idx = group_data.index[i]
                df.at[idx, 'x'] = tree.center_x
                df.at[idx, 'y'] = tree.center_y
                df.at[idx, 'deg'] = tree.angle
        else:
            print(f"  No improvement for N={n}")

    print(f"Total Score Improvement: {total_improvement:.6f}")
    save_submission(df.drop(columns=['group_id']), output_csv)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    main()
