#!/usr/bin/env python3
"""
Exhaustive search for optimal N=1 configuration.
N=1 is trivial - just one tree. We search ALL angles to find the TRUE optimal.
"""

import numpy as np

# Tree polygon vertices (15 points)
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_VERTICES = np.array(list(zip(TX, TY)))

def rotate_polygon(vertices, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return vertices @ rotation_matrix.T

def get_bounding_box_side(angle_deg):
    """Calculate the bounding box side for a single tree at given angle."""
    rotated = rotate_polygon(TREE_VERTICES, angle_deg)
    min_x, min_y = rotated.min(axis=0)
    max_x, max_y = rotated.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    return max(width, height)

# Current best for N=1
current_angle = 45.0
current_side = get_bounding_box_side(current_angle)
print(f"Current N=1: angle={current_angle}°, side={current_side:.6f}")

# Exhaustive search with fine granularity
best_angle = current_angle
best_side = current_side

# Search from 0 to 180 degrees (180-360 is symmetric)
for angle in np.arange(0, 180, 0.001):
    side = get_bounding_box_side(angle)
    if side < best_side:
        best_side = side
        best_angle = angle

print(f"\nExhaustive search result:")
print(f"Best angle: {best_angle:.6f}°")
print(f"Best side: {best_side:.6f}")
print(f"Improvement: {current_side - best_side:.9f}")

# Fine-tune around the best angle
for angle in np.arange(best_angle - 0.01, best_angle + 0.01, 0.00001):
    side = get_bounding_box_side(angle)
    if side < best_side:
        best_side = side
        best_angle = angle

print(f"\nFine-tuned result:")
print(f"Best angle: {best_angle:.9f}°")
print(f"Best side: {best_side:.9f}")
print(f"Improvement: {current_side - best_side:.12f}")

# Calculate score contribution
current_score = current_side ** 2 / 1
best_score = best_side ** 2 / 1
print(f"\nScore impact:")
print(f"Current score contribution: {current_score:.6f}")
print(f"Best score contribution: {best_score:.6f}")
print(f"Score improvement: {current_score - best_score:.9f}")
