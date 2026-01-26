"""
Find REAL improvements (positive diff = snapshot is better than baseline).
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

def kaggle_has_overlap(trees):
    if len(trees) <= 1:
        return False
    
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    
    for i, poly in enumerate(polygons):
        indices = tree_index.query(poly)
        for idx in indices:
            if idx == i:
                continue
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False

def validate_n_kaggle(df, n):
    group_data = df[df['id'].str.startswith(f'{n:03d}_')]
    if len(group_data) != n:
        return False
    
    trees = []
    for _, row in group_data.iterrows():
        x = str(row['x']).replace('s', '')
        y = str(row['y']).replace('s', '')
        deg = str(row['deg']).replace('s', '')
        trees.append(ChristmasTree(x, y, deg))
    
    return not kaggle_has_overlap(trees)

def compute_score_high_precision(df, n):
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
    
    # Compute baseline scores
    baseline_scores = {}
    for n in range(1, 201):
        baseline_scores[n] = compute_score_high_precision(baseline_df, n)
    
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.15f}")
    
    # Check ALL snapshots for REAL improvements (positive diff)
    snapshot_dir = "/home/nonroot/snapshots/santa-2025/"
    snapshot_ids = [d for d in os.listdir(snapshot_dir) if d not in BAD_SNAPSHOTS and d != '21337353543']
    
    all_improvements = []  # (n, improvement, snapshot_id, valid)
    
    for snapshot_id in snapshot_ids:
        csv_path = f"{snapshot_dir}/{snapshot_id}/submission/submission.csv"
        if not os.path.exists(csv_path):
            continue
        
        try:
            df = pd.read_csv(csv_path)
        except:
            continue
        
        if 'id' not in df.columns or 'x' not in df.columns:
            continue
        
        sample_x = str(df['x'].iloc[0])
        if not sample_x.startswith('s'):
            continue
        
        for n in range(1, 201):
            group_data = df[df['id'].str.startswith(f'{n:03d}_')]
            if len(group_data) != n:
                continue
            
            score = compute_score_high_precision(df, n)
            diff = baseline_scores[n] - score
            
            # Only keep IMPROVEMENTS (positive diff)
            if diff > 1e-12:
                valid = validate_n_kaggle(df, n)
                all_improvements.append((n, diff, snapshot_id, valid, score, baseline_scores[n]))
    
    # Sort by improvement size
    all_improvements.sort(key=lambda x: -x[1])
    
    print(f"\nFound {len(all_improvements)} potential improvements")
    print(f"\nTop 20 improvements:")
    for n, diff, snapshot_id, valid, score, baseline in all_improvements[:20]:
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"  N={n}: improvement={diff:.12f} from {snapshot_id} {status}")
        print(f"         snapshot={score:.15f}, baseline={baseline:.15f}")
    
    # Count valid improvements
    valid_improvements = [x for x in all_improvements if x[3]]
    print(f"\nValid improvements: {len(valid_improvements)}")
    
    if valid_improvements:
        total_valid_improvement = sum(x[1] for x in valid_improvements)
        print(f"Total valid improvement: {total_valid_improvement:.12f}")
        
        print(f"\nAll valid improvements:")
        for n, diff, snapshot_id, valid, score, baseline in valid_improvements:
            print(f"  N={n}: improvement={diff:.12f} from {snapshot_id}")

if __name__ == "__main__":
    main()
