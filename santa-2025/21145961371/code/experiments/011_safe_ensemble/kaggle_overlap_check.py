"""
Kaggle-exact overlap detection using scale_factor = 1e15
"""

import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# Match Kaggle's exact settings
getcontext().prec = 25
scale_factor = Decimal('1e15')

class ChristmasTreeKaggle:
    """Kaggle-exact Christmas tree implementation with 1e15 scaling."""
    
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.raw_x = center_x
        self.raw_y = center_y
        self.raw_deg = angle

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

        # Use scale_factor like Kaggle does
        initial_polygon = Polygon([
            (float(Decimal('0.0') * scale_factor), float(tip_y * scale_factor)),
            (float(top_w / Decimal('2') * scale_factor), float(tier_1_y * scale_factor)),
            (float(top_w / Decimal('4') * scale_factor), float(tier_1_y * scale_factor)),
            (float(mid_w / Decimal('2') * scale_factor), float(tier_2_y * scale_factor)),
            (float(mid_w / Decimal('4') * scale_factor), float(tier_2_y * scale_factor)),
            (float(base_w / Decimal('2') * scale_factor), float(base_y * scale_factor)),
            (float(trunk_w / Decimal('2') * scale_factor), float(base_y * scale_factor)),
            (float(trunk_w / Decimal('2') * scale_factor), float(trunk_bottom_y * scale_factor)),
            (float(-(trunk_w / Decimal('2')) * scale_factor), float(trunk_bottom_y * scale_factor)),
            (float(-(trunk_w / Decimal('2')) * scale_factor), float(base_y * scale_factor)),
            (float(-(base_w / Decimal('2')) * scale_factor), float(base_y * scale_factor)),
            (float(-(mid_w / Decimal('4')) * scale_factor), float(tier_2_y * scale_factor)),
            (float(-(mid_w / Decimal('2')) * scale_factor), float(tier_2_y * scale_factor)),
            (float(-(top_w / Decimal('4')) * scale_factor), float(tier_1_y * scale_factor)),
            (float(-(top_w / Decimal('2')) * scale_factor), float(tier_1_y * scale_factor)),
        ])

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))

def load_trees_for_n_kaggle(df, n):
    """Load trees using Kaggle-exact implementation."""
    prefix = f'{n:03d}_'
    subset = df[df['id'].str.startswith(prefix)]
    trees = []
    for _, row in subset.iterrows():
        try:
            x = str(row['x']).lstrip('s')
            y = str(row['y']).lstrip('s')
            deg = str(row['deg']).lstrip('s')
            float(x); float(y); float(deg)
            trees.append(ChristmasTreeKaggle(x, y, deg))
        except:
            return []
    return trees

def has_overlap_kaggle(trees):
    """Kaggle-exact overlap detection."""
    if len(trees) <= 1:
        return False
    try:
        polygons = [t.polygon for t in trees]
        tree_index = STRtree(polygons)
        
        for i, poly in enumerate(polygons):
            indices = tree_index.query(poly)
            for idx in indices:
                if idx != i:
                    # Kaggle's exact check
                    if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                        return True
        return False
    except Exception as e:
        print(f"Error in overlap check: {e}")
        return True

def get_bounding_box_side_kaggle(trees):
    """Calculate bounding box side length with Kaggle scaling."""
    if not trees:
        return float('inf')
    try:
        all_coords = []
        for tree in trees:
            coords = np.array(tree.polygon.exterior.coords)
            all_coords.append(coords)
        all_coords = np.vstack(all_coords)
        x_range = (all_coords[:, 0].max() - all_coords[:, 0].min()) / float(scale_factor)
        y_range = (all_coords[:, 1].max() - all_coords[:, 1].min()) / float(scale_factor)
        return max(x_range, y_range)
    except:
        return float('inf')

def get_raw_config(trees):
    return [(t.raw_x, t.raw_y, t.raw_deg) for t in trees]

if __name__ == '__main__':
    # Check the 009 submission for overlaps using Kaggle-exact detection
    df_009 = pd.read_csv('/home/code/experiments/009_full_ensemble_v2/submission.csv', dtype=str)
    print(f"009 submission has {len(df_009)} rows")
    
    overlaps = []
    for n in range(1, 201):
        trees = load_trees_for_n_kaggle(df_009, n)
        if len(trees) != n:
            print(f"N={n}: Wrong number of trees ({len(trees)})")
            continue
        if has_overlap_kaggle(trees):
            overlaps.append(n)
            print(f"N={n}: HAS OVERLAP (Kaggle detection)")
    
    print(f"\nTotal overlapping N values: {len(overlaps)}")
    print(f"Overlapping N values: {overlaps}")
    
    # Calculate score
    total_score = 0
    for n in range(1, 201):
        trees = load_trees_for_n_kaggle(df_009, n)
        if len(trees) == n:
            side = get_bounding_box_side_kaggle(trees)
            total_score += (side ** 2) / n
    
    print(f"\n009 total score: {total_score:.6f}")
    
    # Also check baseline
    print("\n--- Checking baseline ---")
    df_baseline = pd.read_csv('/home/code/experiments/001_baseline/santa-2025.csv', dtype=str)
    
    baseline_overlaps = []
    for n in range(1, 201):
        trees = load_trees_for_n_kaggle(df_baseline, n)
        if len(trees) != n:
            print(f"N={n}: Wrong number of trees ({len(trees)})")
            continue
        if has_overlap_kaggle(trees):
            baseline_overlaps.append(n)
            print(f"N={n}: HAS OVERLAP (Kaggle detection)")
    
    print(f"\nBaseline overlapping N values: {len(baseline_overlaps)}")
    
    baseline_score = 0
    for n in range(1, 201):
        trees = load_trees_for_n_kaggle(df_baseline, n)
        if len(trees) == n:
            side = get_bounding_box_side_kaggle(trees)
            baseline_score += (side ** 2) / n
    
    print(f"Baseline total score: {baseline_score:.6f}")
