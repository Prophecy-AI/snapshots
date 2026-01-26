"""
Check for actual improvements with higher precision.
"""

import os
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

getcontext().prec = 30
scale_factor = Decimal('1e18')

BAD_SNAPSHOTS = {'21145963314', '21145965159', '21336527339', '21337107511'}

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
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )

def compute_score_high_precision(df, n):
    """Compute score with high precision."""
    group_data = df[df['id'].str.startswith(f'{n:03d}_')]
    
    trees = []
    for _, row in group_data.iterrows():
        x = str(row['x']).replace('s', '')
        y = str(row['y']).replace('s', '')
        deg = str(row['deg']).replace('s', '')
        trees.append(ChristmasTree(x, y, deg))
    
    all_coords = []
    for t in trees:
        all_coords.extend(t.polygon.exterior.coords)
    
    xs = [Decimal(str(c[0])) for c in all_coords]
    ys = [Decimal(str(c[1])) for c in all_coords]
    
    side = max(max(xs) - min(xs), max(ys) - min(ys)) / scale_factor
    score = float(side ** 2) / n
    
    return score

def main():
    # Load baseline
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    baseline_df = pd.read_csv(baseline_path)
    
    # Compute baseline scores with high precision
    baseline_scores = {}
    for n in range(1, 201):
        baseline_scores[n] = compute_score_high_precision(baseline_df, n)
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.15f}")
    
    # Check a few snapshots for actual improvements
    snapshot_dir = "/home/nonroot/snapshots/santa-2025/"
    test_snapshots = ['21198893057', '21116303805', '21165872902']
    
    for snapshot_id in test_snapshots:
        csv_path = f"{snapshot_dir}/{snapshot_id}/submission/submission.csv"
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path)
        
        print(f"\nSnapshot {snapshot_id}:")
        improvements = []
        
        for n in range(1, 201):
            group_data = df[df['id'].str.startswith(f'{n:03d}_')]
            if len(group_data) != n:
                continue
            
            score = compute_score_high_precision(df, n)
            diff = baseline_scores[n] - score
            
            if abs(diff) > 1e-12:  # Significant difference
                improvements.append((n, diff, score, baseline_scores[n]))
        
        if improvements:
            print(f"  Found {len(improvements)} significant differences:")
            for n, diff, score, baseline in improvements[:10]:
                print(f"    N={n}: diff={diff:.15f} (snapshot={score:.15f}, baseline={baseline:.15f})")
        else:
            print(f"  No significant differences found")

if __name__ == "__main__":
    main()
