"""
Analyze interlock patterns in current best solution.
Look for blue/pink (up/down) tree orientations and their offsets.
"""
import pandas as pd
import numpy as np
import math

def strip(v):
    return float(str(v).replace("s", ""))

# Load current best submission
df = pd.read_csv('/home/submission/submission.csv')
df['N'] = df['id'].str.split('_').str[0].astype(int)

print("Analyzing interlock patterns in current best solution...")
print("=" * 70)

# For large N values, analyze the patterns
for n in [50, 100, 150, 200]:
    g = df[df['N'] == n]
    angles = np.array([strip(v) for v in g['deg']])
    xs = np.array([strip(v) for v in g['x']])
    ys = np.array([strip(v) for v in g['y']])
    
    # Normalize angles to 0-360
    angles = angles % 360
    
    # Classify as "up" (pointing up, angle near 0 or 180) or "down" (pointing down)
    # The tree tip is at y=0.8 when angle=0
    # When angle=180, tip points down
    up_mask = (angles < 90) | (angles > 270)
    down_mask = ~up_mask
    
    up_count = np.sum(up_mask)
    down_count = np.sum(down_mask)
    
    print(f"\nN={n}:")
    print(f"  Up trees: {up_count}, Down trees: {down_count}")
    print(f"  Angle stats: mean={np.mean(angles):.1f}, std={np.std(angles):.1f}")
    print(f"  Angle range: [{np.min(angles):.1f}, {np.max(angles):.1f}]")
    
    # Find nearest neighbor for each tree
    nn_dists = []
    nn_angles = []
    for i in range(n):
        min_dist = float('inf')
        min_j = -1
        for j in range(n):
            if i != j:
                dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if dist < min_dist:
                    min_dist = dist
                    min_j = j
        nn_dists.append(min_dist)
        # Check if nearest neighbor has opposite orientation
        if up_mask[i] != up_mask[min_j]:
            nn_angles.append('opposite')
        else:
            nn_angles.append('same')
    
    opposite_nn = sum(1 for a in nn_angles if a == 'opposite')
    print(f"  Nearest neighbor orientation: {opposite_nn}/{n} opposite ({100*opposite_nn/n:.1f}%)")
    print(f"  NN distance: mean={np.mean(nn_dists):.3f}, std={np.std(nn_dists):.3f}")
    
    # Find the most common angle pairs
    angle_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if dist < np.mean(nn_dists) * 1.5:  # Close neighbors
                a1, a2 = angles[i], angles[j]
                angle_diff = abs(a1 - a2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                angle_pairs.append(angle_diff)
    
    if angle_pairs:
        print(f"  Common angle differences: mean={np.mean(angle_pairs):.1f}, mode~{np.median(angle_pairs):.1f}")
    
    # Find dx, dy offsets between up and down trees
    if up_count > 0 and down_count > 0:
        up_xs = xs[up_mask]
        up_ys = ys[up_mask]
        down_xs = xs[down_mask]
        down_ys = ys[down_mask]
        
        # For each up tree, find nearest down tree
        offsets = []
        for i in range(len(up_xs)):
            min_dist = float('inf')
            min_dx, min_dy = 0, 0
            for j in range(len(down_xs)):
                dist = math.sqrt((up_xs[i] - down_xs[j])**2 + (up_ys[i] - down_ys[j])**2)
                if dist < min_dist:
                    min_dist = dist
                    min_dx = down_xs[j] - up_xs[i]
                    min_dy = down_ys[j] - up_ys[i]
            if min_dist < 1.0:  # Only consider close pairs
                offsets.append((min_dx, min_dy))
        
        if offsets:
            dx_mean = np.mean([o[0] for o in offsets])
            dy_mean = np.mean([o[1] for o in offsets])
            print(f"  Up->Down offset: dx={dx_mean:.3f}, dy={dy_mean:.3f}")
