import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from scipy.optimize import minimize

# Tree definition
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))
TREE_POLY = Polygon(TREE_COORDS)

def get_tree(x, y, angle):
    t = affinity.rotate(TREE_POLY, angle, origin=(0,0))
    t = affinity.translate(t, x, y)
    return t

def objective(params):
    dx, dy = params
    # Tree 1: Upright at (0,0)
    t1 = get_tree(0, 0, 0)
    # Tree 2: Rotated 180 at (dx, dy)
    t2 = get_tree(dx, dy, 180)
    
    # Check overlap
    if t1.intersects(t2):
        # Penalty for overlap
        intersection = t1.intersection(t2).area
        return 1000 + intersection * 1000
    
    # Bounding box of the pair
    minx = min(t1.bounds[0], t2.bounds[0])
    miny = min(t1.bounds[1], t2.bounds[1])
    maxx = max(t1.bounds[2], t2.bounds[2])
    maxy = max(t1.bounds[3], t2.bounds[3])
    
    w = maxx - minx
    h = maxy - miny
    
    # We want to minimize the area of the unit cell effectively occupied
    # But actually we want to minimize the spacing for a lattice.
    # If we pack them in a grid, the density is what matters.
    # Let's minimize the bounding box area of the pair for now.
    return w * h

def optimize():
    # Initial guess: place t2 somewhere above/next to t1
    # Tree is about 0.7 wide and 1.0 high.
    # Try placing it at (0.35, 0.5)
    x0 = [0.35, 0.5]
    
    # We need a better objective for lattice packing.
    # We want to find dx, dy such that we can tile the plane with this pair.
    # But for now, let's just find the tightest packing of two trees with 180 rotation.
    
    # Using COBYLA or similar derivative-free method might be better due to sharp edges
    res = minimize(objective, x0, method='Nelder-Mead', tol=1e-4)
    
    print(f"Optimal offset: dx={res.x[0]:.6f}, dy={res.x[1]:.6f}")
    print(f"Objective: {res.fun:.6f}")
    
    # Verify overlap
    t1 = get_tree(0, 0, 0)
    t2 = get_tree(res.x[0], res.x[1], 180)
    if t1.intersects(t2):
        print("Warning: Result has overlap!")
        print(f"Intersection area: {t1.intersection(t2).area}")
    else:
        print("Valid packing found.")
        
    return res.x

if __name__ == "__main__":
    optimize()
