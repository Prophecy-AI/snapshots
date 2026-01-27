"""
Analyze crystallization patterns in current best submission.
"""
import pandas as pd
import numpy as np
from numba import njit
import math

def strip(v):
    return float(str(v).replace("s", ""))

# Load current best submission
df = pd.read_csv('/home/submission/submission.csv')
df['N'] = df['id'].str.split('_').str[0].astype(int)

print("Analyzing tree orientations in current best submission...")
print("=" * 60)

# For a few N values, analyze the patterns
for n in [10, 20, 50, 100]:
    g = df[df['N'] == n]
    angles = [strip(v) for v in g['deg']]
    xs = [strip(v) for v in g['x']]
    ys = [strip(v) for v in g['y']]
    
    # Classify angles as "up" (around 0-90, 270-360) or "down" (around 90-270)
    up_count = sum(1 for a in angles if (a < 90 or a > 270))
    down_count = n - up_count
    
    # Calculate spacing statistics
    if n > 1:
        # Find average nearest neighbor distance
        min_dists = []
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    min_dist = min(min_dist, dist)
            min_dists.append(min_dist)
        avg_spacing = np.mean(min_dists)
    else:
        avg_spacing = 0
    
    print(f"N={n:3d}: Up={up_count:2d}, Down={down_count:2d}, Avg spacing={avg_spacing:.3f}")
    print(f"       Angle range: [{min(angles):.1f}, {max(angles):.1f}]")
    print(f"       X range: [{min(xs):.3f}, {max(xs):.3f}]")
    print(f"       Y range: [{min(ys):.3f}, {max(ys):.3f}]")
    print()
