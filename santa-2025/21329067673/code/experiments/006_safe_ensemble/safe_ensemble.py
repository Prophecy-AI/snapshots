"""
Safe Ensemble - Only picks configurations that are STRICTLY valid
Uses baseline as fallback for any N with potential overlap issues
"""
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
import glob
import os

getcontext().prec = 50  # Very high precision

TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def make_tree_polygon(x, y, deg):
    """Create tree polygon with high precision"""
    x = float(str(x).replace('s', ''))
    y = float(str(y).replace('s', ''))
    deg = float(str(deg).replace('s', ''))
    initial_polygon = Polygon(list(zip(TX, TY)))
    rotated = affinity.rotate(initial_polygon, deg, origin=(0, 0))
    return affinity.translate(rotated, xoff=x, yoff=y)

def has_strict_overlap(trees, threshold=1e-15):
    """Check for overlaps with VERY strict threshold"""
    if len(trees) <= 1:
        return False
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if trees[i].intersects(trees[j]) and not trees[i].touches(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > threshold:
                    return True
    return False

def get_side_length(trees):
    """Calculate bounding box side length"""
    xys = np.concatenate([np.asarray(t.exterior.xy).T for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    return max(max_x - min_x, max_y - min_y)

def load_configs_from_csv(csv_path):
    """Load all N configurations from a CSV"""
    try:
        df = pd.read_csv(csv_path)
        configs = {}
        for _, row in df.iterrows():
            id_parts = row['id'].split('_')
            n = int(id_parts[0])
            x = str(row['x']).replace('s', '')
            y = str(row['y']).replace('s', '')
            deg = str(row['deg']).replace('s', '')
            if n not in configs:
                configs[n] = []
            configs[n].append({'x': x, 'y': y, 'deg': deg})
        return configs
    except Exception as e:
        return {}

def evaluate_config(config_list):
    """Evaluate a configuration: returns (score, has_overlap, trees)"""
    trees = [make_tree_polygon(c['x'], c['y'], c['deg']) for c in config_list]
    has_overlap = has_strict_overlap(trees)
    side = get_side_length(trees)
    n = len(trees)
    score = side**2 / n
    return score, has_overlap, trees

# Load baseline (known to be accepted by Kaggle)
baseline_path = "/home/nonroot/snapshots/santa-2025/21116303805/code/preoptimized/santa-2025.csv"
print(f"Loading baseline from {baseline_path}")
baseline_configs = load_configs_from_csv(baseline_path)

# Collect all CSV files
csv_dirs = [
    "/home/nonroot/snapshots/santa-2025/21116303805/code/preoptimized/",
    "/home/nonroot/snapshots/santa-2025/21116303805/code/experiments/",
    "/home/nonroot/snapshots/santa-2025/21328309254/code/experiments/",
    "/home/code/experiments/",
]

all_csvs = []
for d in csv_dirs:
    all_csvs.extend(glob.glob(d + "**/*.csv", recursive=True))

print(f"Found {len(all_csvs)} CSV files")

# For each N, find the best VALID configuration
best_configs = {}
best_scores = {}

# Start with baseline
for n in range(1, 201):
    if n in baseline_configs:
        score, has_overlap, _ = evaluate_config(baseline_configs[n])
        if not has_overlap:
            best_configs[n] = baseline_configs[n]
            best_scores[n] = score
        else:
            print(f"WARNING: Baseline N={n} has overlap!")

print(f"Baseline total score: {sum(best_scores.values()):.6f}")

# Now scan all CSVs for better configurations
improvements = 0
for csv_path in all_csvs:
    if csv_path == baseline_path:
        continue
    
    configs = load_configs_from_csv(csv_path)
    for n, config_list in configs.items():
        if len(config_list) != n:
            continue  # Invalid config
        
        score, has_overlap, _ = evaluate_config(config_list)
        
        # Only accept if STRICTLY better AND no overlap
        if not has_overlap and n in best_scores and score < best_scores[n]:
            improvement = best_scores[n] - score
            if improvement > 1e-10:  # Meaningful improvement
                best_configs[n] = config_list
                best_scores[n] = score
                improvements += 1

print(f"\nFound {improvements} improvements over baseline")
print(f"New total score: {sum(best_scores.values()):.6f}")

# Final validation pass
print("\nFinal validation...")
final_score = 0
overlap_count = 0
for n in range(1, 201):
    config = best_configs[n]
    score, has_overlap, _ = evaluate_config(config)
    final_score += score
    if has_overlap:
        overlap_count += 1
        print(f"WARNING: N={n} has overlap in final output!")

print(f"\nFinal score: {final_score:.6f}")
print(f"Overlapping N values: {overlap_count}")

# Write output
output_path = "submission_safe_ensemble.csv"
rows = []
for n in range(1, 201):
    for i, c in enumerate(best_configs[n]):
        rows.append({
            'id': f'{n:03d}_{i}',
            'x': f"s{c['x']}",
            'y': f"s{c['y']}",
            'deg': f"s{c['deg']}"
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")

# Save metrics
import json
metrics = {
    'cv_score': final_score,
    'baseline_score': 70.676102,
    'improvement': 70.676102 - final_score,
    'overlap_count': overlap_count
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
