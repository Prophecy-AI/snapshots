import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from shapely import affinity
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_coord(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def get_tree_vertices(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    return rx + x, ry + y

def compute_bbox_size(trees):
    all_x, all_y = [], []
    for x, y, angle in trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    if not all_x:
        return 0
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score_for_n(trees, n):
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def create_tree_polygon(x, y, angle_deg):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

def validate_no_overlap(trees, threshold=1e-20):
    polygons = [create_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > threshold:
                    return False, f"Trees {i} and {j} overlap (area={intersection.area:.2e})"
    return True, "OK"

def load_submission(path):
    df = pd.read_csv(path)
    if 'id' in df.columns:
        df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
        df['i'] = df['id'].apply(lambda x: int(x.split('_')[1]))
    df['x'] = df['x'].apply(parse_coord)
    df['y'] = df['y'].apply(parse_coord)
    if 'deg' in df.columns:
        df['deg'] = df['deg'].apply(parse_coord)
    else:
        df['deg'] = 0
    
    result = {}
    for n in range(1, 201):
        n_df = df[df['n'] == n]
        if len(n_df) == n:
            trees = [(row['x'], row['y'], row['deg']) for _, row in n_df.iterrows()]
            result[n] = trees
    return result

# Load baseline (exp_039)
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
baseline = load_submission(baseline_path)

# Load all new sources
sources = {}
source_paths = [
    ("abhishek_fork", "abhishek_fork/submission.csv"),
    ("hvanphucs_update", "hvanphucs_update/submission.csv"),
    ("hvanphucs_ensemble", "hvanphucs_update/submission_ensemble.csv"),
    ("saspav_dataset", "saspav_dataset/santa-2025.csv"),
]

for name, path in source_paths:
    if Path(path).exists():
        sources[name] = load_submission(path)
        print(f"Loaded {name}: {len(sources[name])} N values")

# Compute baseline scores
baseline_scores = {n: compute_score_for_n(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"\nBaseline total: {total_baseline:.6f}")

# Create ensemble with best per-N
ensemble = {n: baseline[n] for n in range(1, 201)}
improvements_found = []

for n in range(1, 201):
    best_score = baseline_scores[n]
    best_source = "baseline"
    best_trees = baseline[n]
    
    for source_name, source_data in sources.items():
        if n in source_data:
            new_score = compute_score_for_n(source_data[n], n)
            if new_score < best_score - 1e-9:
                # Validate no overlaps
                valid, msg = validate_no_overlap(source_data[n])
                if valid:
                    best_score = new_score
                    best_source = source_name
                    best_trees = source_data[n]
                else:
                    print(f"  N={n}: {source_name} has overlap - skipping")
    
    if best_source != "baseline":
        improvement = baseline_scores[n] - best_score
        improvements_found.append((n, improvement, best_source))
        ensemble[n] = best_trees

# Summary
print(f"\nImprovements found: {len(improvements_found)}")
total_improvement = sum(imp for _, imp, _ in improvements_found)
print(f"Total improvement: {total_improvement:.6f}")

for n, imp, source in sorted(improvements_found, key=lambda x: -x[1]):
    print(f"  N={n}: {imp:.6f} from {source}")

# Calculate new total
new_total = sum(compute_score_for_n(ensemble[n], n) for n in range(1, 201))
print(f"\nNew total score: {new_total:.6f}")
print(f"Improvement from baseline: {total_baseline - new_total:.6f}")

# Final validation of all N values
print("\nValidating all N values...")
all_valid = True
for n in range(1, 201):
    valid, msg = validate_no_overlap(ensemble[n])
    if not valid:
        print(f"  N={n}: OVERLAP - {msg}")
        all_valid = False

if all_valid:
    print("All N values validated - no overlaps!")
else:
    print("WARNING: Some N values have overlaps!")

# Save ensemble
rows = []
for n in range(1, 201):
    for i, (x, y, deg) in enumerate(ensemble[n]):
        rows.append({
            'id': f'{n:03d}_{i}',
            'x': f's{x:.20e}',
            'y': f's{y:.20e}',
            'deg': f's{deg:.20e}'
        })

df_out = pd.DataFrame(rows)
df_out.to_csv('final_ensemble_040.csv', index=False)
print(f"\nSaved ensemble to final_ensemble_040.csv")

# Copy to submission folder
import shutil
shutil.copy('final_ensemble_040.csv', '/home/submission/submission.csv')
print("Copied to /home/submission/submission.csv")
