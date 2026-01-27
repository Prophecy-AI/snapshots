"""Overlap detection for Santa 2025 tree packing."""

import numpy as np
from numba import njit
from shapely.geometry import Polygon
from decimal import Decimal, getcontext

from .tree_geometry import get_tree_vertices, get_tree_vertices_numba, TX, TY

# Set high precision for Decimal operations
getcontext().prec = 30
SCALE = 10**18  # Integer scaling for precise validation

@njit
def point_in_polygon_numba(px, py, poly_x, poly_y):
    """Ray casting algorithm to check if point is inside polygon."""
    n = len(poly_x)
    inside = False
    j = n - 1
    
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside

@njit
def segments_intersect_numba(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
    """Check if line segment a1-a2 intersects b1-b2."""
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    
    return (ccw(a1x, a1y, b1x, b1y, b2x, b2y) != ccw(a2x, a2y, b1x, b1y, b2x, b2y) and 
            ccw(a1x, a1y, a2x, a2y, b1x, b1y) != ccw(a1x, a1y, a2x, a2y, b2x, b2y))

@njit
def polygons_overlap_numba(poly1_x, poly1_y, poly2_x, poly2_y):
    """Check if two 15-vertex polygons overlap using Numba."""
    NV = 15
    
    # Quick bounding box check
    min1x = poly1_x[0]
    max1x = poly1_x[0]
    min1y = poly1_y[0]
    max1y = poly1_y[0]
    for i in range(1, NV):
        if poly1_x[i] < min1x: min1x = poly1_x[i]
        if poly1_x[i] > max1x: max1x = poly1_x[i]
        if poly1_y[i] < min1y: min1y = poly1_y[i]
        if poly1_y[i] > max1y: max1y = poly1_y[i]
    
    min2x = poly2_x[0]
    max2x = poly2_x[0]
    min2y = poly2_y[0]
    max2y = poly2_y[0]
    for i in range(1, NV):
        if poly2_x[i] < min2x: min2x = poly2_x[i]
        if poly2_x[i] > max2x: max2x = poly2_x[i]
        if poly2_y[i] < min2y: min2y = poly2_y[i]
        if poly2_y[i] > max2y: max2y = poly2_y[i]
    
    # Bounding boxes don't overlap
    if max1x < min2x or max2x < min1x or max1y < min2y or max2y < min1y:
        return False
    
    # Point-in-polygon checks
    for i in range(NV):
        if point_in_polygon_numba(poly1_x[i], poly1_y[i], poly2_x, poly2_y):
            return True
        if point_in_polygon_numba(poly2_x[i], poly2_y[i], poly1_x, poly1_y):
            return True
    
    # Edge intersection checks
    for i in range(NV):
        ni = (i + 1) % NV
        for j in range(NV):
            nj = (j + 1) % NV
            if segments_intersect_numba(
                poly1_x[i], poly1_y[i], poly1_x[ni], poly1_y[ni],
                poly2_x[j], poly2_y[j], poly2_x[nj], poly2_y[nj]
            ):
                return True
    
    return False

@njit
def has_any_overlap_numba(trees):
    """Check if any pair of trees overlaps.
    
    Args:
        trees: numpy array of shape (n, 3) with columns [x, y, angle_deg]
    
    Returns:
        True if any overlap exists, False otherwise
    """
    n = len(trees)
    if n <= 1:
        return False
    
    # Pre-compute all polygon vertices
    all_poly_x = np.zeros((n, 15))
    all_poly_y = np.zeros((n, 15))
    
    for i in range(n):
        x, y, angle = trees[i, 0], trees[i, 1], trees[i, 2]
        rx, ry = get_tree_vertices_numba(x, y, angle)
        all_poly_x[i] = rx
        all_poly_y[i] = ry
    
    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_numba(all_poly_x[i], all_poly_y[i], 
                                       all_poly_x[j], all_poly_y[j]):
                return True
    
    return False

def has_overlap(trees):
    """Check if any pair of trees overlaps.
    
    Args:
        trees: list of (x, y, angle) tuples or numpy array
    
    Returns:
        True if any overlap exists, False otherwise
    """
    if len(trees) <= 1:
        return False
    trees_arr = np.array(trees, dtype=np.float64)
    return has_any_overlap_numba(trees_arr)

def validate_no_overlap_shapely(trees):
    """Validate no overlaps using Shapely (more accurate but slower)."""
    if len(trees) <= 1:
        return True, []
    
    polygons = []
    for x, y, angle in trees:
        verts = get_tree_vertices(x, y, angle)
        polygons.append(Polygon(verts))
    
    overlaps = []
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                if not polygons[i].touches(polygons[j]):
                    inter = polygons[i].intersection(polygons[j])
                    if inter.area > 1e-12:
                        overlaps.append((i, j, inter.area))
    
    return len(overlaps) == 0, overlaps
