#!/usr/bin/env python3
"""
Analyze the lattice structure of existing solutions to understand
what lattice vectors are being used and if they can be optimized.
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from collections import defaultdict

# Read the baseline
df = pd.read_csv('/home/submission/submission.csv')

def parse_value(s):
    return float(s[1:]) if isinstance(s, str) and s.startswith('s') else float(s)

# Analyze each N value
for N in [100, 150, 200]:
    n_str = f'{N:03d}'
    df_n = df[df['id'].str.startswith(n_str + '_')]
    
    positions = []
    angles = []
    for _, row in df_n.iterrows():
        x = parse_value(row['x'])
        y = parse_value(row['y'])
        deg = parse_value(row['deg'])
        positions.append([x, y])
        angles.append(deg % 360)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    print(f"\n=== N={N} ===")
    print(f"Number of trees: {len(positions)}")
    
    # Analyze angle distribution
    angle_counts = defaultdict(int)
    for a in angles:
        # Round to nearest degree
        rounded = round(a)
        angle_counts[rounded] += 1
    
    print(f"Angle distribution (top 5):")
    sorted_angles = sorted(angle_counts.items(), key=lambda x: -x[1])[:5]
    for angle, count in sorted_angles:
        print(f"  {angle}°: {count} trees")
    
    # Check for 180° pairs
    pairs_180 = 0
    for i, a1 in enumerate(angles):
        for j, a2 in enumerate(angles):
            if i < j:
                diff = abs((a1 - a2) % 360)
                if abs(diff - 180) < 1:
                    pairs_180 += 1
    
    print(f"180° angle pairs: {pairs_180}")
    
    # Analyze nearest neighbor distances
    if len(positions) > 1:
        dist_mat = distance_matrix(positions, positions)
        np.fill_diagonal(dist_mat, np.inf)
        
        nn_distances = dist_mat.min(axis=1)
        print(f"Nearest neighbor distances: min={nn_distances.min():.4f}, max={nn_distances.max():.4f}, mean={nn_distances.mean():.4f}")
        
        # Find common distances (potential lattice vectors)
        all_distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d = dist_mat[i, j]
                if d < 2.0:  # Only consider nearby trees
                    all_distances.append(d)
        
        if all_distances:
            all_distances = np.array(all_distances)
            print(f"Common distances (< 2.0): {len(all_distances)}")
            
            # Histogram of distances
            hist, bins = np.histogram(all_distances, bins=20)
            peak_idx = np.argmax(hist)
            peak_dist = (bins[peak_idx] + bins[peak_idx+1]) / 2
            print(f"Most common distance: {peak_dist:.4f}")
