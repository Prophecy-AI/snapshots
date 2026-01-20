"""
Ensemble approach: Take the best N from multiple snapshot sources.
For each N from 1 to 200, find which snapshot has the best (lowest) score for that N.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
from shapely.geometry import Polygon
from shapely import affinity

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
BASE_TREE = Polygon(zip(TX, TY))

def parse_value(val):
    """Parse a value that may have 's' prefix."""
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def ensure_s_prefix(val):
    """Ensure value has 's' prefix."""
    if isinstance(val, str):
        if val.startswith('s'):
            return val
        return f's{val}'
    return f's{val}'

def create_tree(x, y, deg):
    """Create a tree polygon at position (x, y) with rotation deg."""
    tree = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
    tree = affinity.translate(tree, x, y)
    return tree

def get_bounding_box_side(trees):
    """Get the side length of the bounding box containing all trees."""
    if not trees:
        return 0
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for tree in trees:
        bounds = tree.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    
    return max(max_x - min_x, max_y - min_y)

def get_n_score(df, n):
    """Calculate score for a specific N in a dataframe."""
    prefix = f"{n:03d}_"
    n_rows = df[df['id'].str.startswith(prefix)]
    
    if len(n_rows) != n:
        return float('inf'), None
    
    try:
        trees = []
        for _, row in n_rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            trees.append(create_tree(x, y, deg))
        
        side = get_bounding_box_side(trees)
        score = side**2 / n
        return score, n_rows
    except Exception as e:
        return float('inf'), None

def load_all_snapshots():
    """Load all snapshot submissions."""
    snapshot_dir = Path('/home/nonroot/snapshots/santa-2025')
    snapshots = {}
    
    for snapshot_path in sorted(snapshot_dir.iterdir()):
        if snapshot_path.is_dir():
            submission_path = snapshot_path / 'submission' / 'submission.csv'
            if submission_path.exists():
                try:
                    df = pd.read_csv(submission_path)
                    # Check required columns
                    if all(col in df.columns for col in ['id', 'x', 'y', 'deg']):
                        snapshots[snapshot_path.name] = df
                        print(f"Loaded {snapshot_path.name} ({len(df)} rows)")
                    else:
                        print(f"Skipping {snapshot_path.name}: missing columns")
                except Exception as e:
                    print(f"Failed to load {snapshot_path.name}: {e}")
    
    return snapshots

def create_ensemble(snapshots):
    """Create ensemble by taking best N from each snapshot."""
    best_rows = []
    best_sources = {}
    total_score = 0
    
    for n in range(1, 201):
        best_score = float('inf')
        best_n_rows = None
        best_source = None
        
        for source_name, df in snapshots.items():
            score, n_rows = get_n_score(df, n)
            if score < best_score:
                best_score = score
                best_n_rows = n_rows
                best_source = source_name
        
        if best_n_rows is not None:
            best_rows.append(best_n_rows)
            best_sources[n] = (best_source, best_score)
            total_score += best_score
            if n <= 10 or n % 20 == 0:
                print(f"N={n}: best from {best_source}, score={best_score:.6f}")
        else:
            print(f"WARNING: No valid configuration for N={n}")
    
    # Combine all best rows
    ensemble_df = pd.concat(best_rows, ignore_index=True)
    
    # Ensure all values have 's' prefix
    ensemble_df['x'] = ensemble_df['x'].apply(ensure_s_prefix)
    ensemble_df['y'] = ensemble_df['y'].apply(ensure_s_prefix)
    ensemble_df['deg'] = ensemble_df['deg'].apply(ensure_s_prefix)
    
    return ensemble_df, total_score, best_sources

def main():
    print("Loading all snapshots...")
    snapshots = load_all_snapshots()
    print(f"\nLoaded {len(snapshots)} snapshots")
    
    print("\nCreating ensemble...")
    ensemble_df, total_score, best_sources = create_ensemble(snapshots)
    
    print(f"\nEnsemble total score: {total_score:.6f}")
    
    # Count how many N values came from each source
    source_counts = {}
    for n, (source, score) in best_sources.items():
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\nSource distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {source}: {count} N values")
    
    # Save ensemble
    output_path = '/home/code/experiments/002_ensemble_snapshots/ensemble.csv'
    ensemble_df.to_csv(output_path, index=False)
    print(f"\nSaved ensemble to {output_path}")
    
    # Also save to submission folder
    ensemble_df.to_csv('/home/submission/submission.csv', index=False)
    print("Saved to /home/submission/submission.csv")
    
    return total_score

if __name__ == "__main__":
    score = main()
