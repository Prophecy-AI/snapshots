"""
Global search for N=2 with random restarts.
"""
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
import math
from numba import njit
import time
import random

# Tree polygon vertices
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)
    poly = translate(poly, x, y)
    return poly

def check_overlap(x1, y1, a1, x2, y2, a2):
    """Check if two trees overlap."""
    p1 = get_tree_polygon(x1, y1, a1)
    p2 = get_tree_polygon(x2, y2, a2)
    if p1.intersects(p2):
        if not p1.touches(p2):
            return p1.intersection(p2).area > 1e-10
    return False

@njit
def compute_bbox_score_fast(x1, y1, a1, x2, y2, a2, tx, ty):
    """Compute bounding box score for N=2."""
    n = 2
    V = len(tx)
    mnx = 1e300
    mny = 1e300
    mxx = -1e300
    mxy = -1e300
    
    # Tree 1
    r1 = a1 * math.pi / 180.0
    c1 = math.cos(r1)
    s1 = math.sin(r1)
    for j in range(V):
        X = c1 * tx[j] - s1 * ty[j] + x1
        Y = s1 * tx[j] + c1 * ty[j] + y1
        if X < mnx: mnx = X
        if X > mxx: mxx = X
        if Y < mny: mny = Y
        if Y > mxy: mxy = Y
    
    # Tree 2
    r2 = a2 * math.pi / 180.0
    c2 = math.cos(r2)
    s2 = math.sin(r2)
    for j in range(V):
        X = c2 * tx[j] - s2 * ty[j] + x2
        Y = s2 * tx[j] + c2 * ty[j] + y2
        if X < mnx: mnx = X
        if X > mxx: mxx = X
        if Y < mny: mny = Y
        if Y > mxy: mxy = Y
    
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def global_search_n2():
    """Global search with systematic grid."""
    tx = TX.astype(np.float64)
    ty = TY.astype(np.float64)
    
    # Current best
    current_score = 0.450779
    best_score = current_score
    best_config = None
    
    print(f"Current N=2 score: {current_score:.6f}")
    print("Searching all angle combinations with optimal positioning...")
    
    # For each angle combination, find the optimal relative position
    # that minimizes the bounding box
    
    for a1 in range(0, 360, 2):  # 2 degree steps
        for a2 in range(0, 360, 2):
            # For this angle combination, search positions
            # Fix tree 1 at origin
            x1, y1 = 0.0, 0.0
            
            # Search tree 2 positions
            for x2 in np.arange(-1.5, 1.5, 0.02):
                for y2 in np.arange(-1.5, 1.5, 0.02):
                    score = compute_bbox_score_fast(x1, y1, a1, x2, y2, a2, tx, ty)
                    
                    if score < best_score:
                        if not check_overlap(x1, y1, a1, x2, y2, a2):
                            best_score = score
                            best_config = (x1, y1, a1, x2, y2, a2)
                            print(f"  New best: {score:.6f} at a1={a1}, a2={a2}, x2={x2:.2f}, y2={y2:.2f}")
        
        if a1 % 30 == 0:
            print(f"Progress: a1={a1}/360")
    
    print(f"\nFinal best score: {best_score:.6f}")
    print(f"Improvement: {current_score - best_score:.6f}")
    
    return best_score, best_config

if __name__ == "__main__":
    print("=" * 60)
    print("Global Search for N=2")
    print("=" * 60)
    
    best_score, best_config = global_search_n2()
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT:")
    print(f"Current score: 0.450779")
    print(f"Best found:    {best_score:.6f}")
    print(f"Improvement:   {0.450779 - best_score:.6f}")
    if best_config:
        print(f"Config: x1={best_config[0]:.3f}, y1={best_config[1]:.3f}, a1={best_config[2]:.1f}")
        print(f"        x2={best_config[3]:.3f}, y2={best_config[4]:.3f}, a2={best_config[5]:.1f}")
    print("=" * 60)
