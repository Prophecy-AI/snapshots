import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

# Tree definition
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))
TREE_POLY = Polygon(TREE_COORDS)

def get_tree(x, y, angle):
    t = affinity.rotate(TREE_POLY, angle, origin=(0,0))
    t = affinity.translate(t, x, y)
    return t

def generate_lattice_submission(filename="submission.csv"):
    rows = []
    
    # Lattice parameters (derived from optimization + safety margin)
    # Optimization found dx=0.41, dy=0.50
    # We assume a lattice where Up trees are at (i*Lx, j*Ly)
    # Down trees are at (i*Lx + dx, j*Ly + dy)
    # We set Lx = 2*dx, Ly = 2*dy roughly.
    
    Lx = 0.83  # Slightly larger than 2*0.41 = 0.82
    Ly = 1.02  # Slightly larger than 2*0.50 = 1.00 (height of tree)
    
    dx = Lx / 2.0
    dy = Ly / 2.0
    
    for n in range(1, 201):
        # We need to pack n trees.
        # We will fill a square-ish region.
        # Area per tree approx Lx * Ly / 2 = 0.42
        # Total area approx n * 0.42
        # Side length approx sqrt(n * 0.42)
        
        # We generate a grid large enough
        side_count = int(np.ceil(np.sqrt(n))) + 2
        
        trees = []
        
        # Generate candidates
        candidates = []
        for j in range(side_count * 2): # More rows
            for i in range(side_count * 2): # More cols
                # Up tree
                candidates.append({'x': i * Lx, 'y': j * Ly, 'deg': 0})
                # Down tree
                candidates.append({'x': i * Lx + dx, 'y': j * Ly + dy, 'deg': 180})
        
        # Sort candidates by distance to center to form a compact cluster
        # Center of grid roughly
        cx = side_count * Lx / 2
        cy = side_count * Ly / 2
        
        candidates.sort(key=lambda p: (p['x']-cx)**2 + (p['y']-cy)**2)
        
        # Take first n
        selected = candidates[:n]
        
        # Center the cluster at 0,0 for the submission format (optional but good practice)
        # Actually the packer handles global shift, but let's center it.
        min_x = min(p['x'] for p in selected)
        max_x = max(p['x'] for p in selected)
        min_y = min(p['y'] for p in selected)
        max_y = max(p['y'] for p in selected)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        for i, p in enumerate(selected):
            rows.append({
                "id": f"{n:03d}_{i}",
                "x": f"s{p['x'] - center_x}",
                "y": f"s{p['y'] - center_y}",
                "deg": f"s{p['deg']}"
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {len(df)} rows.")

if __name__ == "__main__":
    generate_lattice_submission()
