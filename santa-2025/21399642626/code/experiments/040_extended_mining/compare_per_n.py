import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_coord(val):
    """Parse coordinate value (handles 's' prefix)"""
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def get_tree_vertices(x, y, angle_deg):
    """Get tree polygon vertices after rotation and translation"""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Rotate
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    
    # Translate
    return rx + x, ry + y

def compute_bbox_size(trees):
    """Compute bounding box size for a set of trees"""
    all_x, all_y = [], []
    for x, y, angle in trees:
        vx, vy = get_tree_vertices(x, y, angle)
        all_x.extend(vx)
        all_y.extend(vy)
    
    if not all_x:
        return 0
    
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))

def compute_score_for_n(trees, n):
    """Compute score contribution for N trees"""
    size = compute_bbox_size(trees)
    return (size ** 2) / n

def load_submission(path):
    """Load submission and return dict of n -> list of (x, y, angle)"""
    df = pd.read_csv(path)
    
    # Handle different column formats
    if 'id' in df.columns:
        # Format: id like "001_0", "002_1", etc.
        df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
        df['i'] = df['id'].apply(lambda x: int(x.split('_')[1]))
    
    if 'x' in df.columns:
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

def compare_submissions(baseline_path, new_paths):
    """Compare new submissions against baseline per-N"""
    print(f"Loading baseline: {baseline_path}")
    baseline = load_submission(baseline_path)
    print(f"Loaded {len(baseline)} N values from baseline")
    
    # Compute baseline scores per N
    baseline_scores = {}
    for n in range(1, 201):
        if n in baseline:
            baseline_scores[n] = compute_score_for_n(baseline[n], n)
    
    total_baseline = sum(baseline_scores.values())
    print(f"Baseline total score: {total_baseline:.6f}")
    print()
    
    # Compare each new submission
    improvements = {}
    
    for name, path in new_paths.items():
        if not Path(path).exists():
            print(f"  {name}: FILE NOT FOUND")
            continue
            
        try:
            new_sub = load_submission(path)
            if len(new_sub) < 200:
                print(f"  {name}: Only {len(new_sub)} N values (need 200)")
                continue
                
            # Find improvements
            found = []
            for n in range(1, 201):
                if n in new_sub and n in baseline_scores:
                    new_score = compute_score_for_n(new_sub[n], n)
                    if new_score < baseline_scores[n] - 1e-9:
                        improvement = baseline_scores[n] - new_score
                        found.append((n, improvement, new_score, baseline_scores[n]))
            
            if found:
                total_improvement = sum(f[1] for f in found)
                print(f"  {name}: {len(found)} improvements, total {total_improvement:.6f}")
                for n, imp, new_s, base_s in sorted(found, key=lambda x: -x[1])[:5]:
                    print(f"    N={n}: {base_s:.6f} -> {new_s:.6f} (improvement: {imp:.6f})")
                improvements[name] = found
            else:
                print(f"  {name}: No improvements found")
                
        except Exception as e:
            print(f"  {name}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    return improvements, baseline, baseline_scores

# Paths to compare
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
new_paths = {
    "ibrahim_ensemble": "ibrahim_ensemble/best_ensemble_submission.csv",
    "abhishek_fork": "abhishek_fork/submission.csv",
    "hvanphucs_update": "hvanphucs_update/submission.csv",
    "hvanphucs_ensemble": "hvanphucs_update/submission_ensemble.csv",
    "jurgen_prepacker": "jurgen_prepacker/santa_2025_v2_submission.csv",
}

improvements, baseline, baseline_scores = compare_submissions(baseline_path, new_paths)

# Summary
print("\n" + "="*60)
print("SUMMARY OF ALL IMPROVEMENTS FOUND")
print("="*60)

all_improvements = {}
for source, found in improvements.items():
    for n, imp, new_s, base_s in found:
        if n not in all_improvements or imp > all_improvements[n][1]:
            all_improvements[n] = (source, imp, new_s, base_s)

if all_improvements:
    total_potential = sum(v[1] for v in all_improvements.values())
    print(f"\nTotal unique improvements: {len(all_improvements)}")
    print(f"Total potential improvement: {total_potential:.6f}")
    print(f"\nTop 10 improvements:")
    for n, (source, imp, new_s, base_s) in sorted(all_improvements.items(), key=lambda x: -x[1][1])[:10]:
        print(f"  N={n}: {imp:.6f} from {source}")
else:
    print("\nNo improvements found from any source")
