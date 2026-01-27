"""
Smart search for N=2 using geometric insights.

Key insight: For N=2, the optimal configuration has trees arranged
such that their combined bounding box is minimized.

The tree polygon has:
- Width: 0.7 (from -0.35 to 0.35)
- Height: 1.0 (from -0.2 to 0.8)

For N=2, the score is (side^2) / 2, so we want to minimize the side length.
"""
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
import math
from numba import njit

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

def find_optimal_position(a1, a2, tx, ty):
    """Find optimal position for tree 2 given angles."""
    x1, y1 = 0.0, 0.0
    
    best_score = float('inf')
    best_pos = None
    
    # Search positions
    for x2 in np.arange(-1.5, 1.5, 0.01):
        for y2 in np.arange(-1.5, 1.5, 0.01):
            score = compute_bbox_score_fast(x1, y1, a1, x2, y2, a2, tx, ty)
            if score < best_score:
                if not check_overlap(x1, y1, a1, x2, y2, a2):
                    best_score = score
                    best_pos = (x2, y2)
    
    return best_score, best_pos

if __name__ == "__main__":
    tx = TX.astype(np.float64)
    ty = TY.astype(np.float64)
    
    print("=" * 60)
    print("Smart Search for N=2")
    print("=" * 60)
    
    current_score = 0.450779
    print(f"Current N=2 score: {current_score:.6f}")
    
    # Try specific angle combinations that might work well
    # Based on tree symmetry, try angles that are 180 degrees apart
    # or complementary angles
    
    best_score = current_score
    best_config = None
    
    # Test specific promising angle combinations
    test_angles = [
        (0, 180),    # Opposite directions
        (45, 225),   # 45 degree offset
        (90, 270),   # Perpendicular
        (0, 0),      # Same direction
        (45, 45),    # Both at 45
        (203.6, 23.6),  # Current angles
        (180, 0),
        (135, 315),
        (22.5, 202.5),
        (67.5, 247.5),
    ]
    
    print("\nTesting specific angle combinations:")
    for a1, a2 in test_angles:
        score, pos = find_optimal_position(a1, a2, tx, ty)
        if score and score < best_score:
            best_score = score
            best_config = (0, 0, a1, pos[0], pos[1], a2)
            print(f"  a1={a1:.1f}, a2={a2:.1f}: {score:.6f} at x2={pos[0]:.3f}, y2={pos[1]:.3f} ✓")
        elif score:
            print(f"  a1={a1:.1f}, a2={a2:.1f}: {score:.6f}")
    
    # Now do a finer search around the best angles
    if best_config:
        print(f"\nFine-tuning around best angles...")
        a1_best, a2_best = best_config[2], best_config[5]
        
        for da1 in range(-10, 11, 1):
            for da2 in range(-10, 11, 1):
                a1 = a1_best + da1
                a2 = a2_best + da2
                score, pos = find_optimal_position(a1, a2, tx, ty)
                if score and score < best_score:
                    best_score = score
                    best_config = (0, 0, a1, pos[0], pos[1], a2)
                    print(f"  Fine: a1={a1:.1f}, a2={a2:.1f}: {score:.6f}")
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT:")
    print(f"Current score: {current_score:.6f}")
    print(f"Best found:    {best_score:.6f}")
    print(f"Improvement:   {current_score - best_score:.6f}")
    if best_config:
        print(f"Config: x1={best_config[0]:.3f}, y1={best_config[1]:.3f}, a1={best_config[2]:.1f}")
        print(f"        x2={best_config[3]:.3f}, y2={best_config[4]:.3f}, a2={best_config[5]:.1f}")
    print("=" * 60)
