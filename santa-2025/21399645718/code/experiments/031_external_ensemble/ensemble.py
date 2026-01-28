"""
Ensemble current best with external data sources (Saspav santa-2025.csv).
Expected improvement: ~0.008 points.
"""

import pandas as pd
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate, translate
from decimal import Decimal, getcontext
import os

getcontext().prec = 30
SCALE = 10**18

# Tree polygon vertices
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def get_tree_polygon(x, y, angle):
    coords = list(zip(TX, TY))
    poly = Polygon(coords)
    poly = rotate(poly, angle, origin=(0, 0))
    poly = translate(poly, x, y)
    return poly

def get_bbox_size(trees):
    all_coords = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_coords.extend(poly.exterior.coords)
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))

def calculate_score(trees, n):
    bbox = get_bbox_size(trees)
    return bbox**2 / n

def parse_value(s):
    """Parse 's' prefixed values"""
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def check_overlap_strict(trees):
    """Check for overlaps using strict integer arithmetic"""
    polygons = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        # Scale to integers
        coords = [(int(Decimal(str(c[0])) * SCALE), 
                   int(Decimal(str(c[1])) * SCALE)) 
                  for c in poly.exterior.coords]
        polygons.append(Polygon(coords))
    
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 0:
                    return True
    return False

def load_submission(filepath):
    """Load a submission file and return per-N data"""
    df = pd.read_csv(filepath)
    
    # Handle different formats
    if 'id' in df.columns:
        df['n'] = df['id'].apply(lambda x: int(str(x).split('_')[0]))
        df['i'] = df['id'].apply(lambda x: int(str(x).split('_')[1]))
    
    # Parse values
    for col in ['x', 'y', 'deg']:
        if col in df.columns:
            df[f'{col}_val'] = df[col].apply(parse_value)
    
    return df

def get_trees_for_n(df, n):
    """Get trees for a specific N value"""
    group = df[df['n'] == n].sort_values('i')
    return [(row['x_val'], row['y_val'], row['deg_val']) for _, row in group.iterrows()]

def get_raw_data_for_n(df, n):
    """Get raw data (x, y, deg strings) for a specific N value"""
    group = df[df['n'] == n].sort_values('i')
    return [(row['x'], row['y'], row['deg']) for _, row in group.iterrows()]

def main():
    print("=== EXTERNAL DATA ENSEMBLE ===\n")
    
    # Load current best
    print("Loading current best submission...")
    current_df = load_submission('/home/submission/submission.csv')
    
    # Load external data
    print("Loading external data (Saspav santa-2025.csv)...")
    external_df = load_submission('/home/code/data/saspav/santa-2025.csv')
    
    # Calculate current scores
    current_scores = {}
    for n in range(1, 201):
        trees = get_trees_for_n(current_df, n)
        current_scores[n] = calculate_score(trees, n)
    
    current_total = sum(current_scores.values())
    print(f"Current total score: {current_total:.6f}")
    
    # Calculate external scores
    external_scores = {}
    for n in range(1, 201):
        trees = get_trees_for_n(external_df, n)
        if len(trees) == n:
            external_scores[n] = calculate_score(trees, n)
    
    external_total = sum(external_scores.values())
    print(f"External total score: {external_total:.6f}")
    
    # Find improvements
    improvements = []
    for n in range(1, 201):
        if n in external_scores:
            diff = current_scores[n] - external_scores[n]
            if diff > 1e-6:
                improvements.append((n, diff))
    
    print(f"\nFound {len(improvements)} N values with potential improvements")
    improvements.sort(key=lambda x: -x[1])
    
    # Build ensemble
    print("\n=== BUILDING ENSEMBLE ===")
    ensemble_data = []
    total_improvement = 0
    improvements_applied = 0
    
    for n in range(1, 201):
        use_external = False
        
        if n in external_scores and current_scores[n] - external_scores[n] > 1e-6:
            # Check for overlaps in external solution
            external_trees = get_trees_for_n(external_df, n)
            if not check_overlap_strict(external_trees):
                use_external = True
                improvement = current_scores[n] - external_scores[n]
                total_improvement += improvement
                improvements_applied += 1
                if improvement > 0.0001:
                    print(f"  N={n:3d}: Using external (improvement: {improvement:.6f})")
            else:
                print(f"  N={n:3d}: External has overlaps, keeping current")
        
        # Get data for this N
        if use_external:
            raw_data = get_raw_data_for_n(external_df, n)
        else:
            raw_data = get_raw_data_for_n(current_df, n)
        
        for i, (x, y, deg) in enumerate(raw_data):
            ensemble_data.append({
                'id': f'{n:03d}_{i}',
                'x': x,
                'y': y,
                'deg': deg
            })
    
    print(f"\nApplied {improvements_applied} improvements")
    print(f"Total improvement: {total_improvement:.6f}")
    
    # Calculate final score
    ensemble_df = pd.DataFrame(ensemble_data)
    ensemble_df['n'] = ensemble_df['id'].apply(lambda x: int(x.split('_')[0]))
    ensemble_df['x_val'] = ensemble_df['x'].apply(parse_value)
    ensemble_df['y_val'] = ensemble_df['y'].apply(parse_value)
    ensemble_df['deg_val'] = ensemble_df['deg'].apply(parse_value)
    
    final_total = 0
    for n in range(1, 201):
        trees = get_trees_for_n(ensemble_df, n)
        final_total += calculate_score(trees, n)
    
    print(f"\nFinal ensemble score: {final_total:.6f}")
    print(f"Improvement over current: {current_total - final_total:.6f}")
    
    # Save submission
    output_df = ensemble_df[['id', 'x', 'y', 'deg']]
    output_df.to_csv('submission.csv', index=False)
    print(f"\nSaved to submission.csv")
    
    # Validate
    print("\n=== VALIDATION ===")
    test_df = pd.read_csv('submission.csv')
    print(f"Rows: {len(test_df)}")
    print(f"Expected: {sum(range(1, 201))}")
    
    # Check for overlaps in a few N values
    print("\nSpot-checking for overlaps...")
    for n in [2, 10, 50, 100, 150, 200]:
        trees = get_trees_for_n(ensemble_df, n)
        has_overlap = check_overlap_strict(trees)
        print(f"  N={n}: {'OVERLAP!' if has_overlap else 'OK'}")
    
    return final_total

if __name__ == '__main__':
    score = main()
    
    # Save results
    import json
    with open('results.json', 'w') as f:
        json.dump({
            'score': score,
            'approach': 'external_data_ensemble',
            'sources': ['current_best', 'saspav_santa-2025.csv']
        }, f, indent=2)
