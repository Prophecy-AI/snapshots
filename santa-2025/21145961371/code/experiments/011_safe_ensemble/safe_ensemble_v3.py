"""
Safe Ensemble v3: Use baseline for any N that differs from baseline.
Only accept improvements that are IDENTICAL to baseline except for the improved N.
"""

import pandas as pd
import numpy as np
import os
import glob
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

getcontext().prec = 25
scale_factor = Decimal('1e15')

class ChristmasTreeKaggle:
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

def load_trees_for_n(df, n):
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
    if len(trees) <= 1:
        return False
    try:
        polygons = [t.polygon for t in trees]
        tree_index = STRtree(polygons)
        
        for i, poly in enumerate(polygons):
            indices = tree_index.query(poly)
            for idx in indices:
                if idx != i:
                    if poly.intersects(polygons[idx]):
                        intersection = poly.intersection(polygons[idx])
                        if intersection.area > 0:
                            return True
        return False
    except:
        return True

def get_bounding_box_side(trees):
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

def main():
    # Load baseline - this is the ONLY source we trust
    baseline_path = '/home/code/experiments/001_baseline/santa-2025.csv'
    df_baseline = pd.read_csv(baseline_path, dtype=str)
    
    print("Loading baseline configurations...")
    baseline_configs = {}
    baseline_scores = {}
    for n in range(1, 201):
        trees = load_trees_for_n(df_baseline, n)
        baseline_configs[n] = get_raw_config(trees)
        baseline_scores[n] = (get_bounding_box_side(trees) ** 2) / n
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total score: {baseline_total:.6f}")
    
    # Just use baseline as-is for now
    # The baseline is the safest option since we don't know why 009 failed
    
    # Create submission from baseline
    submission_rows = []
    for n in range(1, 201):
        config = baseline_configs[n]
        for i, (x, y, deg) in enumerate(config):
            row_id = f'{n:03d}_{i}'
            submission_rows.append({'id': row_id, 'x': f's{x}', 'y': f's{y}', 'deg': f's{deg}'})
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv('/home/submission/submission.csv', index=False)
    submission_df.to_csv('/home/code/experiments/011_safe_ensemble/submission.csv', index=False)
    print(f"\nSaved baseline submission with {len(submission_df)} rows")
    
    import json
    with open('/home/code/experiments/011_safe_ensemble/metrics.json', 'w') as f:
        json.dump({
            'cv_score': baseline_total,
            'baseline_score': baseline_total,
            'improvement': 0.0,
            'num_improvements': 0,
            'note': 'Using baseline as-is since 009 failed on Kaggle for unknown overlap reasons'
        }, f, indent=2)
    
    print(f"\n=== FINAL SCORE: {baseline_total:.6f} ===")
    return baseline_total

if __name__ == '__main__':
    main()
