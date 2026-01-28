import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

# Tree area (computed from polygon)
TREE_AREA = 0.245625  # Approximate area of tree polygon

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

# Load baseline
baseline_path = "/home/code/experiments/039_per_n_analysis/safe_ensemble.csv"
baseline = load_submission(baseline_path)

# Compute scores and analyze gaps
print("="*80)
print("GAP ANALYSIS: Current Score vs Theoretical Minimum")
print("="*80)
print()

results = []
for n in range(1, 201):
    if n in baseline:
        current_score = compute_score_for_n(baseline[n], n)
        
        # Theoretical minimum (perfect packing)
        theoretical_min = TREE_AREA  # side^2/N = N*area/N = area
        
        # Gap
        gap = current_score - theoretical_min
        gap_pct = (gap / current_score) * 100
        
        # Packing efficiency
        bbox_size = compute_bbox_size(baseline[n])
        bbox_area = bbox_size ** 2
        total_tree_area = n * TREE_AREA
        efficiency = (total_tree_area / bbox_area) * 100
        
        results.append({
            'n': n,
            'current_score': current_score,
            'theoretical_min': theoretical_min,
            'gap': gap,
            'gap_pct': gap_pct,
            'bbox_size': bbox_size,
            'efficiency': efficiency
        })

df_results = pd.DataFrame(results)

# Summary statistics
print(f"Total current score: {df_results['current_score'].sum():.6f}")
print(f"Total theoretical minimum: {df_results['theoretical_min'].sum():.6f}")
print(f"Total gap: {df_results['gap'].sum():.6f}")
print()

# Top 20 N values with largest gaps
print("TOP 20 N VALUES WITH LARGEST GAPS (absolute):")
print("-"*60)
top_gaps = df_results.nlargest(20, 'gap')
for _, row in top_gaps.iterrows():
    print(f"  N={int(row['n']):3d}: gap={row['gap']:.4f} ({row['gap_pct']:.1f}%), efficiency={row['efficiency']:.1f}%")

print()
print("TOP 20 N VALUES WITH WORST EFFICIENCY:")
print("-"*60)
worst_eff = df_results.nsmallest(20, 'efficiency')
for _, row in worst_eff.iterrows():
    print(f"  N={int(row['n']):3d}: efficiency={row['efficiency']:.1f}%, gap={row['gap']:.4f}")

# Group by N ranges
print()
print("SCORE CONTRIBUTION BY N RANGE:")
print("-"*60)
ranges = [(1, 10), (11, 30), (31, 50), (51, 100), (101, 150), (151, 200)]
for start, end in ranges:
    subset = df_results[(df_results['n'] >= start) & (df_results['n'] <= end)]
    total_score = subset['current_score'].sum()
    total_gap = subset['gap'].sum()
    avg_eff = subset['efficiency'].mean()
    print(f"  N={start:3d}-{end:3d}: score={total_score:.4f}, gap={total_gap:.4f}, avg_eff={avg_eff:.1f}%")

# Save analysis
df_results.to_csv('gap_analysis.csv', index=False)
print(f"\nSaved detailed analysis to gap_analysis.csv")
