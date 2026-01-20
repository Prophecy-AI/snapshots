"""
Safe Ensemble with Kaggle-exact overlap detection.
Only counts as overlap if intersection has non-zero area.
"""

import pandas as pd
import numpy as np
import os
import glob
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# Match Kaggle's exact settings
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
    """
    Kaggle-exact overlap detection.
    Only counts as overlap if intersection has non-zero area.
    Edge-to-edge contact (LineString intersection) is allowed.
    """
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
                        # Only count as overlap if intersection has area > 0
                        # This allows edge-to-edge contact (LineString/Point intersections)
                        if intersection.area > 0:
                            return True
        return False
    except Exception as e:
        print(f"Error in overlap check: {e}")
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
    # Load baseline
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
    
    # Verify baseline has no overlaps
    print("\nVerifying baseline has no overlaps...")
    baseline_overlaps = []
    for n in range(1, 201):
        trees = load_trees_for_n(df_baseline, n)
        if has_overlap_kaggle(trees):
            baseline_overlaps.append(n)
    print(f"Baseline overlaps: {baseline_overlaps}")
    
    # Find all CSV files
    snapshot_dirs = glob.glob('/home/nonroot/snapshots/santa-2025/*/code/**/*.csv', recursive=True)
    current_csvs = glob.glob('/home/code/**/*.csv', recursive=True)
    all_csvs = snapshot_dirs + current_csvs
    print(f"\nTotal CSV files to check: {len(all_csvs)}")
    
    # Track best configs
    best_configs = {n: list(baseline_configs[n]) for n in range(1, 201)}
    best_scores = {n: baseline_scores[n] for n in range(1, 201)}
    improvements = []
    
    # Process each CSV file
    valid_files = 0
    for filepath in all_csvs:
        try:
            df = pd.read_csv(filepath, dtype=str)
            if 'id' not in df.columns or len(df) != 20100:
                continue
        except:
            continue
        
        valid_files += 1
        
        for n in range(1, 201):
            try:
                trees = load_trees_for_n(df, n)
                if len(trees) != n:
                    continue
                
                side = get_bounding_box_side(trees)
                if side == float('inf'):
                    continue
                    
                score = (side ** 2) / n
                
                # Only consider if strictly better
                if score < best_scores[n] - 1e-10:
                    # Check for overlaps with Kaggle-exact detection
                    if not has_overlap_kaggle(trees):
                        improvement = best_scores[n] - score
                        improvements.append((n, improvement, filepath))
                        best_scores[n] = score
                        best_configs[n] = get_raw_config(trees)
                        print(f"  N={n}: Improved by {improvement:.6f} from {os.path.basename(filepath)}")
            except:
                continue
    
    print(f"\nProcessed {valid_files} valid CSV files")
    print(f"Total improvements found: {len(improvements)}")
    
    if improvements:
        total_improvement = sum(imp for _, imp, _ in improvements)
        print(f"Total score improvement: {total_improvement:.6f}")
    
    new_total = sum(best_scores.values())
    print(f"\nNew total score: {new_total:.6f}")
    print(f"Baseline total: {baseline_total:.6f}")
    print(f"Improvement: {baseline_total - new_total:.6f}")
    
    # Final validation
    print("\nFinal validation...")
    all_valid = True
    for n in range(1, 201):
        config = best_configs[n]
        try:
            trees = [ChristmasTreeKaggle(x, y, deg) for x, y, deg in config]
            if has_overlap_kaggle(trees):
                print(f"  N={n} has overlaps - reverting to baseline")
                all_valid = False
                best_configs[n] = list(baseline_configs[n])
                best_scores[n] = baseline_scores[n]
        except:
            print(f"  N={n} invalid - reverting to baseline")
            all_valid = False
            best_configs[n] = list(baseline_configs[n])
            best_scores[n] = baseline_scores[n]
    
    if all_valid:
        print("All configurations valid!")
    else:
        new_total = sum(best_scores.values())
        print(f"After reverting: {new_total:.6f}")
    
    # Create submission
    submission_rows = []
    for n in range(1, 201):
        config = best_configs[n]
        for i, (x, y, deg) in enumerate(config):
            row_id = f'{n:03d}_{i}'
            submission_rows.append({'id': row_id, 'x': f's{x}', 'y': f's{y}', 'deg': f's{deg}'})
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv('/home/submission/submission.csv', index=False)
    submission_df.to_csv('/home/code/experiments/011_safe_ensemble/submission.csv', index=False)
    print(f"\nSaved submission with {len(submission_df)} rows")
    
    import json
    with open('/home/code/experiments/011_safe_ensemble/metrics.json', 'w') as f:
        json.dump({
            'cv_score': new_total,
            'baseline_score': baseline_total,
            'improvement': baseline_total - new_total,
            'num_improvements': len(improvements)
        }, f, indent=2)
    
    print(f"\n=== FINAL SCORE: {new_total:.6f} ===")
    return new_total

if __name__ == '__main__':
    main()
