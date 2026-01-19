import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.prepared import prep
from scipy.optimize import differential_evolution
import time

# Tree definition
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))
TREE_POLY = Polygon(TREE_COORDS)
TREE_AREA = TREE_POLY.area

# Prepared geometries for faster checks
# We only need one base tree (Up) and one rotated tree (Down)
TREE_UP = TREE_POLY
TREE_DOWN = affinity.rotate(TREE_POLY, 180, origin=(0,0))

# We will translate these during check
# But creating Polygon objects is slow.
# Maybe we can just transform coordinates?
# Shapely affinity.translate creates new Polygon.
# Let's try to optimize.
# Actually, for DE, we need robustness. Shapely is robust.
# Let's use shapely but minimize object creation if possible.
# Or just use it, DE is parallelizable.

def get_poly(base_poly, x, y):
    return affinity.translate(base_poly, x, y)

def check_overlap(params):
    v1_x, v1_y, v2_x, v2_y, off_x, off_y = params
    
    # Lattice vectors
    v1 = np.array([v1_x, v1_y])
    v2 = np.array([v2_x, v2_y])
    offset = np.array([off_x, off_y])
    
    # Area of unit cell (parallelogram)
    # Cross product in 2D is determinant
    area = abs(v1_x * v2_y - v1_y * v2_x)
    
    # Penalty if area is too small (density constraint)
    # Max density is 1.0 (impossible). Tree area is ~0.2.
    # If area < 0.2, it's impossible.
    if area < 0.25: # Density > 0.8 constraint implicitly
        return 1000 + (0.25 - area) * 10000
        
    # Check overlaps
    # We check Tree1 at (0,0) against neighbors
    
    # Neighbors to check:
    # Tree1s at n*v1 + m*v2
    # Tree2s at offset + n*v1 + m*v2
    
    # Range of n, m: -1 to 1 is usually enough for first shell
    # But if vectors are short, might need more.
    # Let's assume vectors are reasonable (> 0.5 length).
    
    t1_0 = TREE_UP # At 0,0
    
    # List of polygons to check against t1_0
    others = []
    
    # Tree1 neighbors (excluding 0,0)
    for n in range(-1, 2):
        for m in range(-1, 2):
            if n == 0 and m == 0:
                continue
            pos = n * v1 + m * v2
            # Quick bounding box check?
            # Tree radius is approx 0.5.
            # If dist > 1.5, unlikely to overlap.
            if np.dot(pos, pos) > 4.0:
                continue
            others.append(get_poly(TREE_UP, pos[0], pos[1]))
            
    # Tree2 neighbors
    for n in range(-1, 2):
        for m in range(-1, 2):
            pos = offset + n * v1 + m * v2
            if np.dot(pos, pos) > 4.0:
                continue
            others.append(get_poly(TREE_DOWN, pos[0], pos[1]))
            
    # Check intersection
    # Using STRtree might be overkill for 18 polygons.
    # Just loop.
    for p in others:
        if t1_0.intersects(p):
            # Intersection area penalty
            return 1000 + t1_0.intersection(p).area * 1000
            
    # Also check Tree2 at offset against Tree2 neighbors?
    # By symmetry, checking Tree1 vs all neighbors is sufficient if lattice is infinite.
    # Tree1 vs Tree2(off) is same as Tree2(off) vs Tree1(0).
    # Tree1 vs Tree2(off+v) is same as Tree2(off) vs Tree1(-v).
    # So checking Tree1(0) vs all is enough.
    
    return area

def callback(xk, convergence):
    # Optional printing
    pass

def optimize():
    print("Starting optimization...")
    start_time = time.time()
    
    # Bounds
    # v1_x: [0.4, 1.5]
    # v1_y: [-0.5, 0.5]
    # v2_x: [-0.5, 1.5]
    # v2_y: [0.4, 1.5]
    # off_x: [0, 1.5]
    # off_y: [0, 1.5]
    
    bounds = [
        (0.4, 1.2),   # v1_x
        (-0.5, 0.5),  # v1_y
        (-0.5, 0.8),  # v2_x
        (0.4, 1.2),   # v2_y
        (0.0, 1.0),   # off_x
        (0.0, 1.0)    # off_y
    ]
    
    result = differential_evolution(
        check_overlap,
        bounds,
        strategy='best1bin',
        maxiter=1000,
        popsize=20,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        workers=-1, # Parallel
        disp=True
    )
    
    print("\nOptimization complete!")
    print(f"Time: {time.time() - start_time:.2f}s")
    print(f"Best parameters: {result.x}")
    print(f"Best Area: {result.fun}")
    
    # Calculate density
    # Unit cell contains 2 trees.
    density = (2 * TREE_AREA) / result.fun
    print(f"Density: {density:.4f}")
    
    return result.x

if __name__ == "__main__":
    optimize()
