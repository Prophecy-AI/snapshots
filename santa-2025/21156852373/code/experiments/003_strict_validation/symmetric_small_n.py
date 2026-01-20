"""
Symmetric packing optimization for small N values.
Small N values (1-20) contribute disproportionately to the score.
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import itertools

getcontext().prec = 25

# Tree geometry
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

class ChristmasTree:
    def __init__(self, x, y, angle):
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)
        
        # Build polygon
        angle_rad = self.angle * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        vertices = []
        for i in range(15):
            px = TX[i] * cos_a - TY[i] * sin_a + self.x
            py = TX[i] * sin_a + TY[i] * cos_a + self.y
            vertices.append((px, py))
        
        self.polygon = Polygon(vertices)

def has_overlap(trees):
    """Check if any trees overlap"""
    if len(trees) <= 1:
        return False
    
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].polygon.intersects(trees[j].polygon):
                intersection = trees[i].polygon.intersection(trees[j].polygon)
                if intersection.area > 1e-12:
                    return True
    return False

def get_bounding_box_side(trees):
    """Get bounding box side length"""
    if not trees:
        return float('inf')
    
    all_points = []
    for tree in trees:
        coords = np.asarray(tree.polygon.exterior.xy).T
        all_points.append(coords)
    all_points = np.concatenate(all_points)
    
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    return max(max_coords - min_coords)

def get_score(trees, n):
    """Get score contribution for N trees"""
    side = get_bounding_box_side(trees)
    return (side ** 2) / n

# ============================================================
# N=1: Single tree optimization
# ============================================================
def optimize_n1():
    """Find optimal angle for single tree"""
    best_angle = 45.0
    best_side = float('inf')
    
    # Try angles from 0 to 90 (symmetry means we only need this range)
    for angle in np.linspace(0, 90, 901):
        tree = ChristmasTree(0, 0, angle)
        side = get_bounding_box_side([tree])
        if side < best_side:
            best_side = side
            best_angle = angle
    
    return best_angle, best_side

# ============================================================
# N=2: Two trees - try symmetric placements
# ============================================================
def optimize_n2():
    """Find optimal placement for 2 trees using symmetry"""
    best_config = None
    best_side = float('inf')
    
    # Try different symmetric configurations
    # Trees mirrored across y-axis
    for angle1 in [0, 45, 90, 135, 180, 225, 270, 315]:
        angle2 = 360 - angle1  # Mirror angle
        
        for dx in np.linspace(0.3, 1.5, 25):
            tree1 = ChristmasTree(-dx/2, 0, angle1)
            tree2 = ChristmasTree(dx/2, 0, angle2)
            
            if not has_overlap([tree1, tree2]):
                side = get_bounding_box_side([tree1, tree2])
                if side < best_side:
                    best_side = side
                    best_config = [(tree1.x, tree1.y, tree1.angle), 
                                   (tree2.x, tree2.y, tree2.angle)]
    
    # Trees mirrored across x-axis
    for angle1 in [0, 45, 90, 135, 180, 225, 270, 315]:
        angle2 = 180 - angle1  # Mirror angle
        
        for dy in np.linspace(0.3, 1.5, 25):
            tree1 = ChristmasTree(0, -dy/2, angle1)
            tree2 = ChristmasTree(0, dy/2, angle2)
            
            if not has_overlap([tree1, tree2]):
                side = get_bounding_box_side([tree1, tree2])
                if side < best_side:
                    best_side = side
                    best_config = [(tree1.x, tree1.y, tree1.angle), 
                                   (tree2.x, tree2.y, tree2.angle)]
    
    return best_config, best_side

# ============================================================
# N=4: Four trees - try grid/symmetric placements
# ============================================================
def optimize_n4():
    """Find optimal placement for 4 trees using symmetry"""
    best_config = None
    best_side = float('inf')
    
    # Try 2x2 grid with various angles
    for base_angle in [0, 45, 90, 135]:
        for spacing in np.linspace(0.5, 1.5, 21):
            # 4-fold rotational symmetry
            trees = [
                ChristmasTree(-spacing/2, -spacing/2, base_angle),
                ChristmasTree(spacing/2, -spacing/2, base_angle + 90),
                ChristmasTree(spacing/2, spacing/2, base_angle + 180),
                ChristmasTree(-spacing/2, spacing/2, base_angle + 270),
            ]
            
            if not has_overlap(trees):
                side = get_bounding_box_side(trees)
                if side < best_side:
                    best_side = side
                    best_config = [(t.x, t.y, t.angle) for t in trees]
    
    # Try mirror symmetry (2 pairs)
    for angle1 in [0, 45, 90, 135]:
        angle2 = 180 - angle1
        for dx in np.linspace(0.4, 1.2, 17):
            for dy in np.linspace(0.4, 1.2, 17):
                trees = [
                    ChristmasTree(-dx/2, -dy/2, angle1),
                    ChristmasTree(dx/2, -dy/2, angle2),
                    ChristmasTree(-dx/2, dy/2, 360-angle1),
                    ChristmasTree(dx/2, dy/2, 360-angle2),
                ]
                
                if not has_overlap(trees):
                    side = get_bounding_box_side(trees)
                    if side < best_side:
                        best_side = side
                        best_config = [(t.x, t.y, t.angle) for t in trees]
    
    return best_config, best_side

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
        trees.append(ChristmasTree(x, y, deg))
    return trees

def main():
    # Load baseline
    baseline_path = '/home/code/submission_candidates/candidate_000.csv'
    df = pd.read_csv(baseline_path)
    
    print("Optimizing small N values with symmetric configurations...")
    print("="*60)
    
    improvements = {}
    
    # N=1
    print("\nN=1:")
    baseline_trees = load_trees_for_n(df, 1)
    baseline_side = get_bounding_box_side(baseline_trees)
    baseline_score = get_score(baseline_trees, 1)
    print(f"  Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
    
    best_angle, best_side = optimize_n1()
    best_score = (best_side ** 2) / 1
    print(f"  Optimized: angle={best_angle:.1f}°, side={best_side:.6f}, score={best_score:.6f}")
    print(f"  Improvement: {baseline_score - best_score:.6f}")
    
    if best_score < baseline_score:
        improvements[1] = [(0, 0, best_angle)]
    
    # N=2
    print("\nN=2:")
    baseline_trees = load_trees_for_n(df, 2)
    baseline_side = get_bounding_box_side(baseline_trees)
    baseline_score = get_score(baseline_trees, 2)
    print(f"  Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
    
    best_config, best_side = optimize_n2()
    if best_config:
        best_score = (best_side ** 2) / 2
        print(f"  Optimized: side={best_side:.6f}, score={best_score:.6f}")
        print(f"  Improvement: {baseline_score - best_score:.6f}")
        
        if best_score < baseline_score:
            improvements[2] = best_config
    
    # N=4
    print("\nN=4:")
    baseline_trees = load_trees_for_n(df, 4)
    baseline_side = get_bounding_box_side(baseline_trees)
    baseline_score = get_score(baseline_trees, 4)
    print(f"  Baseline: side={baseline_side:.6f}, score={baseline_score:.6f}")
    
    best_config, best_side = optimize_n4()
    if best_config:
        best_score = (best_side ** 2) / 4
        print(f"  Optimized: side={best_side:.6f}, score={best_score:.6f}")
        print(f"  Improvement: {baseline_score - best_score:.6f}")
        
        if best_score < baseline_score:
            improvements[4] = best_config
    
    print("\n" + "="*60)
    print(f"Total improvements found: {len(improvements)} N values")
    
    return improvements

if __name__ == "__main__":
    improvements = main()
