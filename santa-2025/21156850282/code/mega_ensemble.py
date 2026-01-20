"""
Mega Ensemble: Combine ALL available sources (public datasets + snapshots)
Take best N from each source with overlap checking.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from shapely import affinity
import glob

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
BASE_TREE = Polygon(zip(TX, TY))

def parse_value(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return float(val[1:])
        return float(val)
    return float(val)

def ensure_s_prefix(val):
    if isinstance(val, str):
        if val.startswith('s'):
            return val
        return f's{val}'
    return f's{val}'

def create_tree(x, y, deg):
    tree = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
    tree = affinity.translate(tree, x, y)
    return tree

def get_bounding_box_side(trees):
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

def check_overlaps(trees, tolerance=1e-9):
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].intersects(trees[j]):
                intersection = trees[i].intersection(trees[j])
                if intersection.area > tolerance:
                    return True
    return False

def get_n_score(df, n):
    prefix = f"{n:03d}_"
    n_rows = df[df['id'].str.startswith(prefix)]
    
    if len(n_rows) != n:
        return float('inf'), None, True
    
    try:
        trees = []
        for _, row in n_rows.iterrows():
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            trees.append(create_tree(x, y, deg))
        
        # Check for overlaps
        has_overlap = check_overlaps(trees)
        if has_overlap:
            return float('inf'), None, True
        
        side = get_bounding_box_side(trees)
        score = side**2 / n
        return score, n_rows, False
    except Exception as e:
        return float('inf'), None, True

def load_all_sources():
    """Load all available CSV sources."""
    sources = {}
    
    # 1. Public sources
    public_csvs = [
        '/home/code/public_sources/71.97.csv',
        '/home/code/public_sources/72.49.csv',
        '/home/code/public_sources/submission.csv',
        '/home/code/public_sources/submission_best.csv',
        '/home/code/public_sources/santa-2025.csv',
        '/home/code/public_sources/bucket/submission.csv',
    ]
    
    for csv_path in public_csvs:
        if Path(csv_path).exists():
            try:
                df = pd.read_csv(csv_path)
                if all(col in df.columns for col in ['id', 'x', 'y', 'deg']):
                    name = Path(csv_path).stem
                    sources[f'public_{name}'] = df
                    print(f"Loaded public source: {name}")
            except Exception as e:
                print(f"Failed to load {csv_path}: {e}")
    
    # 2. All snapshots (not just clean ones)
    snapshot_dir = Path('/home/nonroot/snapshots/santa-2025')
    for snapshot_path in sorted(snapshot_dir.iterdir()):
        if snapshot_path.is_dir():
            submission_path = snapshot_path / 'submission' / 'submission.csv'
            if submission_path.exists():
                try:
                    df = pd.read_csv(submission_path)
                    if all(col in df.columns for col in ['id', 'x', 'y', 'deg']):
                        sources[f'snapshot_{snapshot_path.name}'] = df
                except Exception as e:
                    pass
    
    print(f"\nTotal sources loaded: {len(sources)}")
    return sources

def create_mega_ensemble(sources):
    """Create ensemble by taking best N from all sources (with overlap checking)."""
    best_rows = []
    best_sources = {}
    total_score = 0
    
    for n in range(1, 201):
        best_score = float('inf')
        best_n_rows = None
        best_source = None
        
        for source_name, df in sources.items():
            score, n_rows, has_overlap = get_n_score(df, n)
            if not has_overlap and score < best_score:
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
    
    ensemble_df = pd.concat(best_rows, ignore_index=True)
    
    # Ensure all values have 's' prefix
    ensemble_df['x'] = ensemble_df['x'].apply(ensure_s_prefix)
    ensemble_df['y'] = ensemble_df['y'].apply(ensure_s_prefix)
    ensemble_df['deg'] = ensemble_df['deg'].apply(ensure_s_prefix)
    
    return ensemble_df, total_score, best_sources

def main():
    print("Loading all sources...")
    sources = load_all_sources()
    
    print("\nCreating mega ensemble...")
    ensemble_df, total_score, best_sources = create_mega_ensemble(sources)
    
    print(f"\nMega ensemble total score: {total_score:.6f}")
    
    # Count sources
    source_counts = {}
    for n, (source, score) in best_sources.items():
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\nSource distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {source}: {count} N values")
    
    # Save
    output_path = '/home/code/experiments/004_mega_ensemble/ensemble.csv'
    Path('/home/code/experiments/004_mega_ensemble').mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    ensemble_df.to_csv('/home/submission/submission.csv', index=False)
    print("Saved to /home/submission/submission.csv")
    
    return total_score, best_sources

if __name__ == "__main__":
    score, sources = main()
