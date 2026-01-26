"""
Kaggle-Validated Ensemble

The ensemble approach (exp_002, exp_004, exp_006) found CV=70.522682 - a 0.09 improvement!
But ALL 3 attempts FAILED Kaggle validation with "Overlapping trees" errors.

THE PROBLEM: Our local Shapely validation passes, but Kaggle's validation fails.
THE SOLUTION: Use Kaggle's EXACT validation method from the chistyakov kernel.

Key differences:
1. Use Decimal with 25 digits of precision
2. Scale coordinates by 1e18 to integers
3. Check for polygon overlaps using integer arithmetic
"""

import os
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import json
import time

# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e18')  # CRITICAL: Use 1e18

# Bad snapshots to exclude (known format issues)
BAD_SNAPSHOTS = {'21145963314', '21145965159', '21336527339', '21337107511'}

class ChristmasTree:
    """Kaggle's exact tree implementation with integer-scaled coordinates."""
    
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        
        # Tree dimensions
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
        
        # Create polygon with integer-scaled coordinates
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
    """Check for overlaps using Kaggle's exact method."""
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

def validate_n_kaggle(n, df):
    """Validate a single N value using Kaggle's exact method."""
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

def compute_score_from_df(df, n):
    """Compute score for N trees from dataframe."""
    group_data = df[df['id'].str.startswith(f'{n:03d}_')]
    
    trees = []
    for _, row in group_data.iterrows():
        x = float(str(row['x']).replace('s', ''))
        y = float(str(row['y']).replace('s', ''))
        deg = float(str(row['deg']).replace('s', ''))
        trees.append(ChristmasTree(str(x), str(y), str(deg)))
    
    # Get bounding box from scaled polygons
    all_coords = []
    for t in trees:
        all_coords.extend(t.polygon.exterior.coords)
    
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    # Unscale to get actual side length
    side = max(max(xs) - min(xs), max(ys) - min(ys)) / float(scale_factor)
    score = (side ** 2) / n
    
    return score

def build_validated_ensemble():
    """Build ensemble, validating EACH N value before including."""
    
    # Load baseline (known to pass Kaggle validation)
    baseline_path = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
    print(f"Loading baseline from {baseline_path}")
    baseline_df = pd.read_csv(baseline_path)
    
    # Verify baseline format
    print(f"Baseline shape: {baseline_df.shape}")
    print(f"Baseline columns: {baseline_df.columns.tolist()}")
    print(f"Sample row: {baseline_df.iloc[0].to_dict()}")
    
    # First, validate the baseline itself
    print("\nValidating baseline...")
    baseline_valid_count = 0
    baseline_invalid = []
    for n in range(1, 201):
        if validate_n_kaggle(n, baseline_df):
            baseline_valid_count += 1
        else:
            baseline_invalid.append(n)
    
    print(f"Baseline validation: {baseline_valid_count}/200 N values pass")
    if baseline_invalid:
        print(f"Invalid N values in baseline: {baseline_invalid[:10]}...")
    
    # Initialize best_per_n with baseline
    best_per_n = {}
    for n in range(1, 201):
        group_data = baseline_df[baseline_df['id'].str.startswith(f'{n:03d}_')]
        score = compute_score_from_df(baseline_df, n)
        best_per_n[n] = {
            'score': score,
            'rows': group_data.copy(),
            'source': 'baseline',
            'valid': n not in baseline_invalid
        }
    
    baseline_total = sum(best_per_n[n]['score'] for n in range(1, 201))
    print(f"\nBaseline total score: {baseline_total:.6f}")
    
    # Load all snapshots
    snapshot_dir = "/home/nonroot/snapshots/santa-2025/"
    snapshot_ids = [d for d in os.listdir(snapshot_dir) if d not in BAD_SNAPSHOTS and d != '21337353543']
    
    print(f"\nChecking {len(snapshot_ids)} snapshots for improvements...")
    
    improvements_found = 0
    improvements_validated = 0
    
    for i, snapshot_id in enumerate(snapshot_ids):
        csv_path = f"{snapshot_dir}/{snapshot_id}/submission/submission.csv"
        if not os.path.exists(csv_path):
            continue
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            continue
        
        # Check format
        if 'id' not in df.columns or 'x' not in df.columns:
            continue
        
        # Check for 's' prefix
        sample_x = str(df['x'].iloc[0])
        if not sample_x.startswith('s'):
            continue
        
        for n in range(1, 201):
            group_data = df[df['id'].str.startswith(f'{n:03d}_')]
            if len(group_data) != n:
                continue
            
            score = compute_score_from_df(df, n)
            
            # Only consider if better than current best
            if score >= best_per_n[n]['score'] - 1e-10:
                continue
            
            improvements_found += 1
            
            # CRITICAL: Validate using Kaggle's exact method
            if not validate_n_kaggle(n, df):
                # Skip - fails Kaggle validation
                continue
            
            # Improvement found and validated!
            improvement = best_per_n[n]['score'] - score
            improvements_validated += 1
            print(f"  N={n}: IMPROVED by {improvement:.8f} from {snapshot_id}")
            
            best_per_n[n] = {
                'score': score,
                'rows': group_data.copy(),
                'source': snapshot_id,
                'valid': True
            }
        
        if (i + 1) % 20 == 0:
            print(f"  Checked {i + 1}/{len(snapshot_ids)} snapshots, {improvements_validated} validated improvements")
    
    print(f"\nTotal improvements found: {improvements_found}")
    print(f"Validated improvements: {improvements_validated}")
    
    # Build final submission
    rows = []
    for n in range(1, 201):
        rows.append(best_per_n[n]['rows'])
    
    final_df = pd.concat(rows, ignore_index=True)
    
    # Final validation
    print("\nFinal validation of ensemble...")
    final_valid_count = 0
    final_invalid = []
    for n in range(1, 201):
        if validate_n_kaggle(n, final_df):
            final_valid_count += 1
        else:
            final_invalid.append(n)
    
    print(f"Final validation: {final_valid_count}/200 N values pass")
    if final_invalid:
        print(f"Invalid N values: {final_invalid}")
    
    # Calculate final score
    final_total = sum(best_per_n[n]['score'] for n in range(1, 201))
    improvement = baseline_total - final_total
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline score: {baseline_total:.6f}")
    print(f"Ensemble score: {final_total:.6f}")
    print(f"Improvement: {improvement:.6f}")
    print(f"Validated improvements: {improvements_validated}")
    
    # List sources used
    sources = {}
    for n in range(1, 201):
        src = best_per_n[n]['source']
        sources[src] = sources.get(src, 0) + 1
    
    print(f"\nSources used:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count} N values")
    
    return final_df, final_total, improvements_validated

if __name__ == "__main__":
    start_time = time.time()
    
    final_df, final_score, num_improvements = build_validated_ensemble()
    
    elapsed = time.time() - start_time
    print(f"\nTime elapsed: {elapsed:.2f}s")
    
    # Save submission
    output_path = "/home/submission/submission.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved submission to {output_path}")
    
    # Save metrics
    metrics = {
        'cv_score': final_score,
        'baseline_score': 70.615107,
        'improvement': 70.615107 - final_score,
        'validated_improvements': num_improvements
    }
    
    with open('/home/code/experiments/011_kaggle_validated_ensemble/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFinal CV score: {final_score:.6f}")
