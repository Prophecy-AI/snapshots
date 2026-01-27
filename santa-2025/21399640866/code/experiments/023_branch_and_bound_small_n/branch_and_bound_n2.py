"""
Branch-and-bound exhaustive search for N=2.
This is NOT simulated annealing - it's a systematic search.
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

def branch_and_bound_n2(current_score, angle_step=5, pos_step=0.05, pos_range=1.5):
    """
    Exhaustive search for N=2 with pruning.
    
    Strategy:
    - Fix tree 1 at (0, 0) with angle 0 (we can always translate/rotate the final solution)
    - Search all positions and angles for tree 2
    - Use bounding box pruning to skip infeasible regions
    """
    best_score = current_score
    best_config = None
    
    tx = TX.astype(np.float64)
    ty = TY.astype(np.float64)
    
    # Fix tree 1 at origin with angle 0
    x1, y1, a1 = 0.0, 0.0, 0.0
    
    # Search space for tree 2
    angles = np.arange(0, 360, angle_step)
    positions = np.arange(-pos_range, pos_range + pos_step, pos_step)
    
    total_configs = len(angles) * len(positions) * len(positions)
    print(f"Searching {total_configs:,} configurations...")
    print(f"Current best score: {current_score:.6f}")
    
    checked = 0
    valid = 0
    improved = 0
    
    start_time = time.time()
    
    for a2 in angles:
        for x2 in positions:
            for y2 in positions:
                checked += 1
                
                # Quick bounding box check (pruning)
                score = compute_bbox_score_fast(x1, y1, a1, x2, y2, a2, tx, ty)
                
                # Only check overlap if score is better
                if score < best_score:
                    if not check_overlap(x1, y1, a1, x2, y2, a2):
                        valid += 1
                        if score < best_score:
                            improved += 1
                            best_score = score
                            best_config = (x1, y1, a1, x2, y2, a2)
                            print(f"  New best: {score:.6f} at a2={a2}, x2={x2:.2f}, y2={y2:.2f}")
        
        # Progress update
        if checked % 100000 == 0:
            elapsed = time.time() - start_time
            rate = checked / elapsed
            remaining = (total_configs - checked) / rate
            print(f"Progress: {checked:,}/{total_configs:,} ({checked/total_configs*100:.1f}%), "
                  f"valid={valid}, improved={improved}, ETA: {remaining:.0f}s")
    
    elapsed = time.time() - start_time
    print(f"\nSearch complete in {elapsed:.1f}s")
    print(f"Checked: {checked:,}, Valid: {valid}, Improved: {improved}")
    print(f"Best score: {best_score:.6f}")
    
    return best_score, best_config

if __name__ == "__main__":
    # Current N=2 score
    current_score = 0.450779
    
    print("=" * 60)
    print("Branch-and-Bound Search for N=2")
    print("=" * 60)
    
    # First pass: coarse search
    print("\nPhase 1: Coarse search (5 degree, 0.1 position step)")
    best_score, best_config = branch_and_bound_n2(
        current_score, 
        angle_step=5, 
        pos_step=0.1, 
        pos_range=1.5
    )
    
    if best_config:
        # Second pass: fine search around best config
        print("\nPhase 2: Fine search around best config")
        x1, y1, a1, x2, y2, a2 = best_config
        
        # Search in smaller region around best
        fine_angles = np.arange(max(0, a2-10), min(360, a2+10), 1)
        fine_x = np.arange(x2-0.2, x2+0.2, 0.01)
        fine_y = np.arange(y2-0.2, y2+0.2, 0.01)
        
        tx = TX.astype(np.float64)
        ty = TY.astype(np.float64)
        
        for a2_fine in fine_angles:
            for x2_fine in fine_x:
                for y2_fine in fine_y:
                    score = compute_bbox_score_fast(x1, y1, a1, x2_fine, y2_fine, a2_fine, tx, ty)
                    if score < best_score:
                        if not check_overlap(x1, y1, a1, x2_fine, y2_fine, a2_fine):
                            best_score = score
                            best_config = (x1, y1, a1, x2_fine, y2_fine, a2_fine)
                            print(f"  Fine: {score:.6f} at a2={a2_fine}, x2={x2_fine:.3f}, y2={y2_fine:.3f}")
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT:")
    print(f"Current score: {current_score:.6f}")
    print(f"Best found:    {best_score:.6f}")
    print(f"Improvement:   {current_score - best_score:.6f}")
    if best_config:
        print(f"Config: x1={best_config[0]:.3f}, y1={best_config[1]:.3f}, a1={best_config[2]:.1f}")
        print(f"        x2={best_config[3]:.3f}, y2={best_config[4]:.3f}, a2={best_config[5]:.1f}")
    print("=" * 60)
