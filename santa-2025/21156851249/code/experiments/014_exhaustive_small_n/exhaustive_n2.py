#!/usr/bin/env python3
"""
Exhaustive search for optimal N=2 configuration.
For N=2, we search all angle combinations and optimize positions.
"""

import numpy as np
from scipy.optimize import minimize
from itertools import product

# Tree polygon vertices (15 points)
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def rotate_polygon(angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.column_stack([
        TX * cos_a - TY * sin_a,
        TX * sin_a + TY * cos_a
    ])

def get_polygon(x, y, angle):
    rotated = rotate_polygon(angle)
    return rotated + np.array([x, y])

def cross(ax, ay, bx, by):
    return ax * by - ay * bx

def segments_intersect(p1, p2, p3, p4):
    d1 = cross(p4[0]-p3[0], p4[1]-p3[1], p1[0]-p3[0], p1[1]-p3[1])
    d2 = cross(p4[0]-p3[0], p4[1]-p3[1], p2[0]-p3[0], p2[1]-p3[1])
    d3 = cross(p2[0]-p1[0], p2[1]-p1[1], p3[0]-p1[0], p3[1]-p1[1])
    d4 = cross(p2[0]-p1[0], p2[1]-p1[1], p4[0]-p1[0], p4[1]-p1[1])
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

def point_in_polygon(px, py, poly):
    n = len(poly)
    count = 0
    for i in range(n):
        j = (i + 1) % n
        y1, y2 = poly[i, 1], poly[j, 1]
        x1, x2 = poly[i, 0], poly[j, 0]
        if (y1 <= py < y2) or (y2 <= py < y1):
            x_int = x1 + (py - y1) / (y2 - y1) * (x2 - x1)
            if px < x_int:
                count += 1
    return count % 2 == 1

def polygons_overlap(poly1, poly2):
    # Bounding box check
    min1, max1 = poly1.min(axis=0), poly1.max(axis=0)
    min2, max2 = poly2.min(axis=0), poly2.max(axis=0)
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False
    
    # Edge intersection check
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        i2 = (i + 1) % n1
        for j in range(n2):
            j2 = (j + 1) % n2
            if segments_intersect(poly1[i], poly1[i2], poly2[j], poly2[j2]):
                return True
    
    # Containment check
    if point_in_polygon(poly1[0, 0], poly1[0, 1], poly2):
        return True
    if point_in_polygon(poly2[0, 0], poly2[0, 1], poly1):
        return True
    
    return False

def calculate_side(positions, angles):
    """Calculate bounding box side for given configuration."""
    all_vertices = []
    for i, (x, y, angle) in enumerate(zip(positions[::2], positions[1::2], angles)):
        poly = get_polygon(x, y, angle)
        all_vertices.append(poly)
    
    all_vertices = np.vstack(all_vertices)
    min_xy = all_vertices.min(axis=0)
    max_xy = all_vertices.max(axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

def check_overlap(positions, angles):
    """Check if any trees overlap."""
    polys = []
    for i, (x, y, angle) in enumerate(zip(positions[::2], positions[1::2], angles)):
        polys.append(get_polygon(x, y, angle))
    
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False

def optimize_positions(angles, initial_positions=None):
    """Optimize positions for given angles."""
    n = len(angles)
    
    if initial_positions is None:
        # Start with trees spread out
        initial_positions = np.zeros(2 * n)
        for i in range(n):
            initial_positions[2*i] = (i - n/2) * 0.5
            initial_positions[2*i + 1] = 0
    
    def objective(positions):
        if check_overlap(positions, angles):
            return 1e9
        return calculate_side(positions, angles)
    
    best_side = 1e9
    best_positions = initial_positions.copy()
    
    # Try multiple starting points
    for trial in range(20):
        if trial == 0:
            x0 = initial_positions.copy()
        else:
            x0 = initial_positions + np.random.randn(2 * n) * 0.3
        
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.fun < best_side:
            best_side = result.fun
            best_positions = result.x.copy()
    
    return best_side, best_positions

# Current best for N=2
print("Current N=2 configuration:")
print("Angles: 23.6° and 203.6° (180° apart)")
print("Side: 0.949504")
print("Score: 0.450779")

# Search all angle pairs
# Due to 180° symmetry, we only need to search angle1 from 0-180 and angle2 from 0-360
# But for double-lattice, angle2 = angle1 + 180

print("\n=== Searching angle pairs ===")
best_side = 0.949504
best_angles = [23.6, 203.6]
best_positions = None

# First, try double-lattice pattern (angle2 = angle1 + 180)
print("\n1. Double-lattice pattern (angle2 = angle1 + 180):")
for angle1 in np.arange(0, 180, 1.0):
    angle2 = angle1 + 180
    angles = [angle1, angle2]
    side, positions = optimize_positions(angles)
    if side < best_side - 1e-6:
        best_side = side
        best_angles = angles.copy()
        best_positions = positions.copy()
        print(f"  Found better: angles={angles}, side={side:.6f}")

# Fine-tune around best
if best_positions is not None:
    print(f"\nFine-tuning around angle1={best_angles[0]:.1f}°...")
    for angle1 in np.arange(best_angles[0] - 2, best_angles[0] + 2, 0.1):
        angle2 = angle1 + 180
        angles = [angle1, angle2]
        side, positions = optimize_positions(angles, best_positions)
        if side < best_side - 1e-9:
            best_side = side
            best_angles = angles.copy()
            best_positions = positions.copy()
            print(f"  Found better: angles={angles}, side={side:.9f}")

# Try non-double-lattice patterns
print("\n2. Non-double-lattice patterns (random angle pairs):")
np.random.seed(42)
for _ in range(100):
    angle1 = np.random.uniform(0, 360)
    angle2 = np.random.uniform(0, 360)
    angles = [angle1, angle2]
    side, positions = optimize_positions(angles)
    if side < best_side - 1e-6:
        best_side = side
        best_angles = angles.copy()
        best_positions = positions.copy()
        print(f"  Found better: angles={angles}, side={side:.6f}")

print(f"\n=== Final Result ===")
print(f"Best angles: {best_angles}")
print(f"Best side: {best_side:.9f}")
print(f"Best score: {best_side**2 / 2:.9f}")
print(f"Current score: {0.949504**2 / 2:.9f}")
print(f"Improvement: {0.949504**2 / 2 - best_side**2 / 2:.12f}")
