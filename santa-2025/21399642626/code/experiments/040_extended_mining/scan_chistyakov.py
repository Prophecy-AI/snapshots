import pandas as pd
import numpy as np
from pathlib import Path
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

def load_submission(path):
    try:
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
    except:
        return {}

# Load baseline
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
baseline = load_submission(baseline_path)
baseline_scores = {n: compute_score_for_n(baseline[n], n) for n in range(1, 201)}
total_baseline = sum(baseline_scores.values())
print(f"Baseline total: {total_baseline:.6f}")

# Check chistyakov files
for csv_path in Path("chistyakov_dataset").glob("*.csv"):
    sub = load_submission(csv_path)
    print(f"\n{csv_path.name}: Loaded {len(sub)} N values")
    
    improvements = []
    for n in range(1, 201):
        if n in sub:
            new_score = compute_score_for_n(sub[n], n)
            if new_score < baseline_scores[n] - 1e-9:
                improvement = baseline_scores[n] - new_score
                improvements.append((n, improvement, new_score, baseline_scores[n]))
    
    if improvements:
        total_improvement = sum(imp for _, imp, _, _ in improvements)
        print(f"  Found {len(improvements)} improvements, total {total_improvement:.6f}")
        for n, imp, new_s, base_s in sorted(improvements, key=lambda x: -x[1])[:5]:
            print(f"    N={n}: {imp:.6f}")
    else:
        print("  No improvements found")
