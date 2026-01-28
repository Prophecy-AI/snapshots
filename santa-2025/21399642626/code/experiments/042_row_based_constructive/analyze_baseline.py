"""
Analyze the baseline solution to understand its structure.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import pandas as pd
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

# Analyze structure for small N values
print("="*60)
print("BASELINE SOLUTION STRUCTURE ANALYSIS")
print("="*60)

for n in [2, 3, 4, 5, 10, 20]:
    print(f"\nN={n}:")
    trees = baseline[n]
    
    # Analyze angles
    angles = [t[2] for t in trees]
    unique_angles = set(round(a, 1) for a in angles)
    print(f"  Unique angles: {sorted(unique_angles)}")
    
    # Analyze positions
    xs = [t[0] for t in trees]
    ys = [t[1] for t in trees]
    print(f"  X range: [{min(xs):.3f}, {max(xs):.3f}]")
    print(f"  Y range: [{min(ys):.3f}, {max(ys):.3f}]")
    
    # Check for patterns
    if len(trees) > 1:
        # Check if trees are in rows
        y_values = sorted(set(round(y, 2) for _, y, _ in trees))
        print(f"  Y levels: {len(y_values)} ({y_values[:5]}...)" if len(y_values) > 5 else f"  Y levels: {len(y_values)} ({y_values})")
        
        # Check angle distribution
        angle_counts = {}
        for a in angles:
            a_rounded = round(a, 0)
            angle_counts[a_rounded] = angle_counts.get(a_rounded, 0) + 1
        print(f"  Angle distribution: {dict(sorted(angle_counts.items()))}")
