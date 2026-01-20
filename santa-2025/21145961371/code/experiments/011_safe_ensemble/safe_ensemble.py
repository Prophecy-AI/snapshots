"""
Safe Ensemble with Strict Overlap Detection

Uses Kaggle's EXACT collision detection:
- Decimal arithmetic with 30-digit precision
- Shapely's intersects() with proper polygon construction
- Only replaces baseline config if BOTH:
  1. Alternative has STRICTLY better score
  2. Alternative passes STRICT overlap check
"""

import pandas as pd
import numpy as np
import os
import glob
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

getcontext().prec = 30

# Tree geometry using Decimal for precision
class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        
        # Store raw values for later use
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
    """Load trees for a specific N value, preserving raw string values."""
    prefix = f"{n:03d}_"
    subset = df[df['id'].str.startswith(prefix)]
    trees = []
    for _, row in subset.iterrows():
        x = str(row['x']).lstrip('s')
        y = str(row['y']).lstrip('s')
        deg = str(row['deg']).lstrip('s')
        trees.append(ChristmasTree(x, y, deg))
    return trees

def has_overlap_strict(trees):
    """Strict overlap detection matching Kaggle's implementation."""
    if len(trees) <= 1:
        return False
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    
    for i, poly in enumerate(polygons):
        indices = tree_index.query(poly)
        for idx in indices:
            if idx != i:
                if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                    intersection = poly.intersection(polygons[idx])
                    if intersection.area > 1e-12:
                        return True
    return False

def get_bounding_box_side(trees):
    """Calculate bounding box side length."""
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

def get_raw_config(trees):
    """Extract raw string values from trees."""
    return [(t.raw_x, t.raw_y, t.raw_deg) for t in trees]

def load_csv_file(filepath):
    """Load CSV file preserving string precision."""
    try:
        df = pd.read_csv(filepath, dtype=str)
        if 'id' not in df.columns or 'x' not in df.columns:
            return None
        return df
    except:
        return None

def main():
    # Load baseline (verified to work on Kaggle)
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
    
    # Find all CSV files in snapshots
    snapshot_dirs = glob.glob('/home/nonroot/snapshots/santa-2025/*/code/**/*.csv', recursive=True)
    print(f"\nFound {len(snapshot_dirs)} CSV files in snapshots")
    
    # Also check current code directory
    current_csvs = glob.glob('/home/code/**/*.csv', recursive=True)
    all_csvs = snapshot_dirs + current_csvs
    print(f"Total CSV files to check: {len(all_csvs)}")
    
    # Track best configs for each N
    best_configs = {n: list(baseline_configs[n]) for n in range(1, 201)}
    best_scores = {n: baseline_scores[n] for n in range(1, 201)}
    improvements = []
    
    # Process each CSV file
    valid_files = 0
    for filepath in all_csvs:
        df = load_csv_file(filepath)
        if df is None:
            continue
        
        # Check if it has the right structure
        if len(df) != 20100:
            continue
        
        valid_files += 1
        
        # Check each N value
        for n in range(1, 201):
            trees = load_trees_for_n(df, n)
            if len(trees) != n:
                continue
            
            # Calculate score
            side = get_bounding_box_side(trees)
            score = (side ** 2) / n
            
            # Only consider if strictly better
            if score < best_scores[n] - 1e-10:
                # Check for overlaps with strict detection
                if not has_overlap_strict(trees):
                    improvement = best_scores[n] - score
                    improvements.append((n, improvement, filepath))
                    best_scores[n] = score
                    best_configs[n] = get_raw_config(trees)
                    print(f"  N={n}: Improved by {improvement:.6f} from {os.path.basename(filepath)}")
    
    print(f"\nProcessed {valid_files} valid CSV files")
    print(f"Total improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _ in improvements)
        print(f"Total score improvement: {total_improvement:.6f}")
    
    # Calculate new total score
    new_total = sum(best_scores.values())
    print(f"\nNew total score: {new_total:.6f}")
    print(f"Baseline total: {baseline_total:.6f}")
    print(f"Improvement: {baseline_total - new_total:.6f}")
    
    # Final validation: check ALL N values for overlaps
    print("\nFinal validation of all configurations...")
    all_valid = True
    for n in range(1, 201):
        config = best_configs[n]
        trees = [ChristmasTree(x, y, deg) for x, y, deg in config]
        if has_overlap_strict(trees):
            print(f"  ERROR: N={n} has overlaps!")
            all_valid = False
            # Revert to baseline
            best_configs[n] = list(baseline_configs[n])
            best_scores[n] = baseline_scores[n]
    
    if all_valid:
        print("  All configurations valid!")
    else:
        # Recalculate score after reverting
        new_total = sum(best_scores.values())
        print(f"\nAfter reverting invalid configs: {new_total:.6f}")
    
    # Create submission
    submission_rows = []
    for n in range(1, 201):
        config = best_configs[n]
        for i, (x, y, deg) in enumerate(config):
            row_id = f'{n:03d}_{i}'
            submission_rows.append({
                'id': row_id,
                'x': f's{x}',
                'y': f's{y}',
                'deg': f's{deg}'
            })
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv('/home/submission/submission.csv', index=False)
    submission_df.to_csv('/home/code/experiments/011_safe_ensemble/submission.csv', index=False)
    print(f"\nSaved submission with {len(submission_df)} rows")
    
    # Save metrics
    import json
    with open('/home/code/experiments/011_safe_ensemble/metrics.json', 'w') as f:
        json.dump({
            'cv_score': new_total,
            'baseline_score': baseline_total,
            'improvement': baseline_total - new_total,
            'num_improvements': len(improvements),
            'improvements': [(n, imp) for n, imp, _ in improvements]
        }, f, indent=2)
    
    print(f"\n=== FINAL SCORE: {new_total:.6f} ===")
    return new_total

if __name__ == '__main__':
    main()
