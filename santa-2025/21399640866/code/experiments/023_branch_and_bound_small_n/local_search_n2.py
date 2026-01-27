"""
Local search around current N=2 configuration.
"""
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
import math
from numba import njit
import time

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

def local_search_n2():
    """Search around current N=2 configuration."""
    # Current configuration
    x1, y1, a1 = 0.154097, -0.038541, 203.629378
    x2, y2, a2 = -0.154097, -0.561459, 23.629378
    
    tx = TX.astype(np.float64)
    ty = TY.astype(np.float64)
    
    current_score = compute_bbox_score_fast(x1, y1, a1, x2, y2, a2, tx, ty)
    print(f"Current N=2 score: {current_score:.6f}")
    
    best_score = current_score
    best_config = (x1, y1, a1, x2, y2, a2)
    
    # Search parameters
    pos_deltas = np.arange(-0.5, 0.5, 0.01)
    angle_deltas = np.arange(-30, 30, 1)
    
    print(f"Searching {len(pos_deltas)**4 * len(angle_deltas)**2:,} configurations...")
    
    checked = 0
    improved = 0
    start_time = time.time()
    
    for dx1 in pos_deltas:
        for dy1 in pos_deltas:
            for dx2 in pos_deltas:
                for dy2 in pos_deltas:
                    for da1 in angle_deltas:
                        for da2 in angle_deltas:
                            checked += 1
                            
                            nx1 = x1 + dx1
                            ny1 = y1 + dy1
                            na1 = (a1 + da1) % 360
                            nx2 = x2 + dx2
                            ny2 = y2 + dy2
                            na2 = (a2 + da2) % 360
                            
                            score = compute_bbox_score_fast(nx1, ny1, na1, nx2, ny2, na2, tx, ty)
                            
                            if score < best_score:
                                if not check_overlap(nx1, ny1, na1, nx2, ny2, na2):
                                    improved += 1
                                    best_score = score
                                    best_config = (nx1, ny1, na1, nx2, ny2, na2)
                                    print(f"  New best: {score:.6f}")
                            
                            if checked % 10000000 == 0:
                                elapsed = time.time() - start_time
                                print(f"Progress: {checked:,}, improved={improved}, elapsed={elapsed:.0f}s")
    
    elapsed = time.time() - start_time
    print(f"\nSearch complete in {elapsed:.1f}s")
    print(f"Checked: {checked:,}, Improved: {improved}")
    print(f"Current score: {current_score:.6f}")
    print(f"Best score:    {best_score:.6f}")
    print(f"Improvement:   {current_score - best_score:.6f}")
    
    return best_score, best_config

if __name__ == "__main__":
    print("=" * 60)
    print("Local Search for N=2")
    print("=" * 60)
    
    # This is too slow - let's do a smarter search
    # Instead, let's try a gradient-based approach
    
    # Current configuration
    x1, y1, a1 = 0.154097, -0.038541, 203.629378
    x2, y2, a2 = -0.154097, -0.561459, 23.629378
    
    tx = TX.astype(np.float64)
    ty = TY.astype(np.float64)
    
    current_score = compute_bbox_score_fast(x1, y1, a1, x2, y2, a2, tx, ty)
    print(f"Current N=2 score: {current_score:.6f}")
    
    # Try different angle combinations systematically
    print("\nSearching angle combinations...")
    best_score = current_score
    best_config = None
    
    for a1_new in range(0, 360, 5):
        for a2_new in range(0, 360, 5):
            # Keep positions, change angles
            score = compute_bbox_score_fast(x1, y1, a1_new, x2, y2, a2_new, tx, ty)
            if score < best_score:
                if not check_overlap(x1, y1, a1_new, x2, y2, a2_new):
                    best_score = score
                    best_config = (x1, y1, a1_new, x2, y2, a2_new)
                    print(f"  Angles only: {score:.6f} at a1={a1_new}, a2={a2_new}")
    
    # Try different position combinations
    print("\nSearching position combinations...")
    for dx in np.arange(-0.3, 0.3, 0.02):
        for dy in np.arange(-0.3, 0.3, 0.02):
            # Move both trees together (translation)
            score = compute_bbox_score_fast(x1+dx, y1+dy, a1, x2+dx, y2+dy, a2, tx, ty)
            if score < best_score:
                if not check_overlap(x1+dx, y1+dy, a1, x2+dx, y2+dy, a2):
                    best_score = score
                    best_config = (x1+dx, y1+dy, a1, x2+dx, y2+dy, a2)
                    print(f"  Translation: {score:.6f}")
    
    # Try moving tree 2 relative to tree 1
    print("\nSearching relative positions...")
    for dx2 in np.arange(-0.5, 0.5, 0.02):
        for dy2 in np.arange(-0.5, 0.5, 0.02):
            score = compute_bbox_score_fast(x1, y1, a1, x2+dx2, y2+dy2, a2, tx, ty)
            if score < best_score:
                if not check_overlap(x1, y1, a1, x2+dx2, y2+dy2, a2):
                    best_score = score
                    best_config = (x1, y1, a1, x2+dx2, y2+dy2, a2)
                    print(f"  Relative: {score:.6f}")
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT:")
    print(f"Current score: {current_score:.6f}")
    print(f"Best found:    {best_score:.6f}")
    print(f"Improvement:   {current_score - best_score:.6f}")
    print("=" * 60)
