"""Tree geometry and bounding box calculations for Santa 2025."""

import numpy as np
from numba import njit

# Tree geometry - 15 vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)

@njit
def get_tree_vertices_numba(x, y, angle_deg):
    """Get tree polygon vertices at position (x,y) with rotation angle_deg.
    
    Returns: (rx, ry) arrays of x and y coordinates for 15 vertices
    """
    rad = np.radians(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    rx = TX * cos_a - TY * sin_a + x
    ry = TX * sin_a + TY * cos_a + y
    
    return rx, ry

def get_tree_vertices(x, y, angle_deg):
    """Get tree polygon vertices as list of (x, y) tuples."""
    rx, ry = get_tree_vertices_numba(x, y, angle_deg)
    return list(zip(rx, ry))

@njit
def calculate_bbox_numba(trees):
    """Calculate bounding box side length for a set of trees.
    
    Args:
        trees: numpy array of shape (n, 3) with columns [x, y, angle_deg]
    
    Returns:
        side: maximum of width and height
    """
    n = len(trees)
    if n == 0:
        return 0.0
    
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    
    for i in range(n):
        x, y, angle = trees[i, 0], trees[i, 1], trees[i, 2]
        rx, ry = get_tree_vertices_numba(x, y, angle)
        
        for j in range(15):
            if rx[j] < min_x:
                min_x = rx[j]
            if rx[j] > max_x:
                max_x = rx[j]
            if ry[j] < min_y:
                min_y = ry[j]
            if ry[j] > max_y:
                max_y = ry[j]
    
    width = max_x - min_x
    height = max_y - min_y
    
    return max(width, height)

def calculate_bbox(trees):
    """Calculate bounding box side length for a list of (x, y, angle) tuples."""
    if len(trees) == 0:
        return 0.0
    trees_arr = np.array(trees, dtype=np.float64)
    return calculate_bbox_numba(trees_arr)

@njit
def calculate_score_numba(trees):
    """Calculate score contribution for N trees.
    
    Score = side^2 / n
    """
    n = len(trees)
    if n == 0:
        return 0.0
    side = calculate_bbox_numba(trees)
    return (side ** 2) / n

def calculate_score(trees):
    """Calculate score contribution for a list of (x, y, angle) tuples."""
    if len(trees) == 0:
        return 0.0
    trees_arr = np.array(trees, dtype=np.float64)
    return calculate_score_numba(trees_arr)
