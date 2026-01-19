import pandas as pd
import numpy as np

def generate_initial_submission(filename="submission.csv"):
    rows = []
    # Tree bounding box is roughly 1x1 (actually smaller, but let's be safe)
    # Tree coordinates are relative to center.
    # Max extent is roughly +/- 0.35 in X and -0.2 to 0.8 in Y.
    # So width ~0.7, height ~1.0.
    # Let's use 1.0 spacing to be safe.
    
    for n in range(1, 201):
        # Arrange n trees in a grid
        side = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            row = i // side
            col = i % side
            
            # Position
            x = col * 1.0
            y = row * 1.5 # More vertical space
            
            rows.append({
                "id": f"{n:03d}_{i}",
                "x": f"s{x}",
                "y": f"s{y}",
                "deg": "s0"
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {len(df)} rows.")

if __name__ == "__main__":
    generate_initial_submission()
