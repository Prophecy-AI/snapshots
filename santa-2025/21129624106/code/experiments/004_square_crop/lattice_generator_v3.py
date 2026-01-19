import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

# Tree definition
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
TREE_COORDS = list(zip(TX, TY))
TREE_POLY = Polygon(TREE_COORDS)

def generate_lattice_submission(filename="submission.csv"):
    rows = []
    
    # Optimized parameters from Experiment 003
    # Best parameters: [ 0.53770751 -0.16283756 -0.49997463  1.1981019   0.3152888   0.83522419]
    v1 = np.array([0.53770751, -0.16283756])
    v2 = np.array([-0.49997463, 1.1981019])
    offset = np.array([0.3152888, 0.83522419])
    
    # Slightly expand to be safe
    scale = 1.001
    v1 *= scale
    v2 *= scale
    offset *= scale
    
    for n in range(1, 201):
        # Generate candidates
        candidates = []
        
        # Estimate grid size
        K = int(np.ceil(np.sqrt(n))) + 3
        
        for i in range(-K, K+1):
            for j in range(-K, K+1):
                # Up tree
                pos_up = i * v1 + j * v2
                candidates.append({'x': pos_up[0], 'y': pos_up[1], 'deg': 0})
                
                # Down tree
                pos_down = offset + i * v1 + j * v2
                candidates.append({'x': pos_down[0], 'y': pos_down[1], 'deg': 180})
        
        # Sort by Chebyshev distance (L-infinity norm) to create a SQUARE crop
        # We want to minimize max(|x|, |y|)
        # But wait, the lattice is rotated.
        # The bounding box is axis-aligned.
        # So we should sort by max(|x|, |y|) in the global coordinate system.
        # Yes, p['x'] and p['y'] are global coordinates.
        
        # Center of the cluster is roughly (0,0) because we generate i,j in [-K, K]
        # But the lattice is skewed, so (0,0) might not be the geometric center.
        # Let's calculate the center of mass of all candidates first?
        # Or just sort by distance to (0,0) using Chebyshev.
        
        # Better: Sort by Chebyshev distance to (0,0).
        candidates.sort(key=lambda p: max(abs(p['x']), abs(p['y'])))
        
        # Take first n
        selected = candidates[:n]
        
        # Center the cluster exactly
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
