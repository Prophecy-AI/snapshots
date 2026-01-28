"""
Exhaustive search for N=1 optimal rotation angle.
N=1 has 0.311 room for improvement (actual 0.661 vs theoretical 0.350).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(x, y, angle_deg):
    """Get tree polygon vertices after rotation and translation"""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = TX * cos_a - TY * sin_a
    ry = TX * sin_a + TY * cos_a
    return rx + x, ry + y

def compute_bbox_size_for_angle(angle_deg):
    """Compute bounding box size for a single tree at origin with given angle"""
    vx, vy = get_tree_vertices(0, 0, angle_deg)
    width = max(vx) - min(vx)
    height = max(vy) - min(vy)
    return max(width, height)

def compute_score_n1(angle_deg):
    """Compute score for N=1 with given angle"""
    size = compute_bbox_size_for_angle(angle_deg)
    return size ** 2  # N=1, so score = size^2 / 1 = size^2

# Exhaustive search over all angles with 0.001 degree resolution
print("="*60)
print("EXHAUSTIVE SEARCH FOR N=1 OPTIMAL ROTATION")
print("="*60)

# First, coarse search with 0.1 degree resolution
print("\nPhase 1: Coarse search (0.1° resolution)...")
best_angle_coarse = 0
best_score_coarse = float('inf')

for angle_int in range(0, 3600):  # 0 to 360 degrees, 0.1 degree steps
    angle = angle_int / 10.0
    score = compute_score_n1(angle)
    if score < best_score_coarse:
        best_score_coarse = score
        best_angle_coarse = angle

print(f"  Best coarse angle: {best_angle_coarse:.1f}°")
print(f"  Best coarse score: {best_score_coarse:.6f}")

# Fine search around the best coarse angle with 0.001 degree resolution
print("\nPhase 2: Fine search (0.001° resolution)...")
best_angle_fine = best_angle_coarse
best_score_fine = best_score_coarse

# Search ±1 degree around best coarse angle
for angle_int in range(int((best_angle_coarse - 1) * 1000), int((best_angle_coarse + 1) * 1000) + 1):
    angle = angle_int / 1000.0
    score = compute_score_n1(angle)
    if score < best_score_fine:
        best_score_fine = score
        best_angle_fine = angle

print(f"  Best fine angle: {best_angle_fine:.3f}°")
print(f"  Best fine score: {best_score_fine:.9f}")

# Ultra-fine search with 0.0001 degree resolution
print("\nPhase 3: Ultra-fine search (0.0001° resolution)...")
best_angle_ultra = best_angle_fine
best_score_ultra = best_score_fine

# Search ±0.01 degree around best fine angle
for angle_int in range(int((best_angle_fine - 0.01) * 10000), int((best_angle_fine + 0.01) * 10000) + 1):
    angle = angle_int / 10000.0
    score = compute_score_n1(angle)
    if score < best_score_ultra:
        best_score_ultra = score
        best_angle_ultra = angle

print(f"  Best ultra-fine angle: {best_angle_ultra:.4f}°")
print(f"  Best ultra-fine score: {best_score_ultra:.12f}")

# Compare with baseline
print("\n" + "="*60)
print("COMPARISON WITH BASELINE")
print("="*60)

# Load baseline N=1 angle
def parse_coord(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

baseline_path = "/home/code/experiments/044_extended_subset_extraction/ensemble_044.csv"
df = pd.read_csv(baseline_path)
df['n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
n1_row = df[df['n'] == 1].iloc[0]
baseline_angle = parse_coord(n1_row['deg'])
baseline_score = compute_score_n1(baseline_angle)

print(f"Baseline N=1 angle: {baseline_angle:.6f}°")
print(f"Baseline N=1 score: {baseline_score:.12f}")
print(f"Optimal N=1 angle: {best_angle_ultra:.6f}°")
print(f"Optimal N=1 score: {best_score_ultra:.12f}")
print(f"Improvement: {baseline_score - best_score_ultra:.12f}")

if best_score_ultra < baseline_score - 1e-9:
    print("\n✅ IMPROVEMENT FOUND!")
else:
    print("\n❌ No improvement - baseline is already optimal for N=1")

# Also check the theoretical minimum
tree_width = max(TX) - min(TX)  # 0.7
tree_height = max(TY) - min(TY)  # 1.0
print(f"\nTree dimensions: width={tree_width}, height={tree_height}")
print(f"Minimum possible bbox size (diagonal): {np.sqrt(tree_width**2 + tree_height**2):.6f}")
print(f"At 45°, bbox size should be: {compute_bbox_size_for_angle(45):.6f}")
