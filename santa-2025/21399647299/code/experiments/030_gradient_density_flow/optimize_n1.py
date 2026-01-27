"""
Exhaustive optimization for N=1.

For N=1, the score is simply bbox^2 / 1 = bbox^2.
The bbox is the maximum of (width, height) of the tree at a given angle.

We need to find the angle that minimizes the bounding box of a single tree.
"""

import numpy as np
import math
from shapely import Polygon

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_vertices(angle_deg):
    """Get vertices of tree at origin with given angle."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    vertices = []
    for tx, ty in zip(TX, TY):
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        vertices.append((rx, ry))
    
    return vertices

def compute_bbox_for_angle(angle_deg):
    """Compute bounding box size for a single tree at given angle."""
    vertices = get_tree_vertices(angle_deg)
    
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    return max(width, height)

def main():
    print("=" * 60)
    print("EXHAUSTIVE OPTIMIZATION FOR N=1")
    print("=" * 60)
    
    # Current baseline for N=1
    baseline_angle = 45.0
    baseline_bbox = compute_bbox_for_angle(baseline_angle)
    baseline_score = baseline_bbox ** 2
    
    print(f"\nBaseline: angle={baseline_angle}°, bbox={baseline_bbox:.6f}, score={baseline_score:.6f}")
    
    # Exhaustive search over all angles (0.01 degree increments)
    best_angle = 0
    best_bbox = float('inf')
    
    print("\nSearching all angles from 0 to 360 degrees (0.01° increments)...")
    
    for angle_int in range(36000):
        angle = angle_int / 100.0
        bbox = compute_bbox_for_angle(angle)
        
        if bbox < best_bbox:
            best_bbox = bbox
            best_angle = angle
    
    best_score = best_bbox ** 2
    
    print(f"\nBest found: angle={best_angle}°, bbox={best_bbox:.6f}, score={best_score:.6f}")
    print(f"Improvement: {baseline_score - best_score:.6f}")
    
    # Also check some specific angles
    print("\n" + "=" * 60)
    print("CHECKING SPECIFIC ANGLES")
    print("=" * 60)
    
    specific_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    
    for angle in specific_angles:
        bbox = compute_bbox_for_angle(angle)
        score = bbox ** 2
        print(f"  {angle:3d}°: bbox={bbox:.6f}, score={score:.6f}")
    
    # Fine search around best angle
    print("\n" + "=" * 60)
    print(f"FINE SEARCH AROUND {best_angle}°")
    print("=" * 60)
    
    for delta in np.linspace(-1, 1, 21):
        angle = best_angle + delta
        bbox = compute_bbox_for_angle(angle)
        score = bbox ** 2
        marker = " <-- BEST" if abs(angle - best_angle) < 0.01 else ""
        print(f"  {angle:.2f}°: bbox={bbox:.6f}, score={score:.6f}{marker}")
    
    # Verify the tree dimensions
    print("\n" + "=" * 60)
    print("TREE DIMENSIONS AT DIFFERENT ANGLES")
    print("=" * 60)
    
    for angle in [0, 45, 90, best_angle]:
        vertices = get_tree_vertices(angle)
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        bbox = max(width, height)
        print(f"  {angle:.2f}°: width={width:.6f}, height={height:.6f}, bbox={bbox:.6f}")

if __name__ == "__main__":
    main()
