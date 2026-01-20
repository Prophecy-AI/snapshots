"""
Build ensemble from ALL available CSV files in snapshots.
For each N, take the configuration with the smallest bounding box that has NO overlaps.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import glob
import os
import time

# Tree geometry
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def parse_value(val):
    if isinstance(val, str) and val.startswith('s'):
        return val[1:]
    return str(val)

def build_polygon(x, y, angle):
    angle_rad = float(angle) * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    vertices = [(TX[i] * cos_a - TY[i] * sin_a + float(x),
                 TX[i] * sin_a + TY[i] * cos_a + float(y)) for i in range(15)]
    return Polygon(vertices)

def get_group_data(df, n):
    """Get data for group N"""
    prefix = f"{n:03d}_"
    rows = df[df['id'].str.startswith(prefix)].copy()
    return rows

def get_bounding_box_side(rows):
    """Get bounding box side for a group"""
    all_points = []
    for _, row in rows.iterrows():
        x = float(parse_value(row['x']))
        y = float(parse_value(row['y']))
        deg = float(parse_value(row['deg']))
        poly = build_polygon(x, y, deg)
        all_points.extend(list(poly.exterior.coords))
    
    all_points = np.array(all_points)
    side = max(all_points.max(axis=0) - all_points.min(axis=0))
    return side

def has_overlap(rows):
    """Check if group has overlapping trees"""
    polygons = []
    for _, row in rows.iterrows():
        x = float(parse_value(row['x']))
        y = float(parse_value(row['y']))
        deg = float(parse_value(row['deg']))
        polygons.append(build_polygon(x, y, deg))
    
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-12:
                    return True
    return False

def main():
    # Find all CSV files
    csv_files = []
    csv_files += glob.glob('/home/nonroot/snapshots/santa-2025/*/code/**/*.csv', recursive=True)
    csv_files += glob.glob('/home/nonroot/snapshots/santa-2025/*/code/preoptimized/**/*.csv', recursive=True)
    csv_files = list(set(csv_files))
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Best configuration for each N
    best = {n: {'score': float('inf'), 'rows': None, 'source': None} for n in range(1, 201)}
    
    # Process each file
    start_time = time.time()
    valid_files = 0
    
    for file_idx, csv_path in enumerate(csv_files):
        if file_idx % 50 == 0:
            print(f"Processing file {file_idx}/{len(csv_files)}...")
        
        try:
            df = pd.read_csv(csv_path)
            if 'id' not in df.columns or len(df) != 20100:
                continue
            
            valid_files += 1
            
            for n in range(1, 201):
                rows = get_group_data(df, n)
                if len(rows) != n:
                    continue
                
                # Check for overlaps
                if has_overlap(rows):
                    continue
                
                # Calculate score
                side = get_bounding_box_side(rows)
                score = (side ** 2) / n
                
                if score < best[n]['score']:
                    best[n]['score'] = score
                    best[n]['rows'] = rows.copy()
                    best[n]['source'] = csv_path
                    
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {valid_files} valid files in {elapsed:.1f}s")
    
    # Build ensemble
    all_rows = []
    sources_used = {}
    total_score = 0
    
    for n in range(1, 201):
        if best[n]['rows'] is not None:
            all_rows.append(best[n]['rows'])
            total_score += best[n]['score']
            source = best[n]['source'].split('/')[-1]
            sources_used[source] = sources_used.get(source, 0) + 1
    
    if not all_rows:
        print("No valid configurations found!")
        return
    
    # Create ensemble dataframe
    ensemble_df = pd.concat(all_rows, ignore_index=True)
    
    # Sort by id
    ensemble_df['n'] = ensemble_df['id'].str.split('_').str[0].astype(int)
    ensemble_df['i'] = ensemble_df['id'].str.split('_').str[1].astype(int)
    ensemble_df = ensemble_df.sort_values(['n', 'i']).drop(columns=['n', 'i'])
    
    # Save
    output_path = '/home/code/experiments/004_fractional_translation/ensemble.csv'
    ensemble_df.to_csv(output_path, index=False)
    
    print(f"\nEnsemble score: {total_score:.6f}")
    print(f"Target: 68.919154")
    print(f"Gap: {total_score - 68.919154:.6f}")
    print(f"\nTop sources used:")
    for source, count in sorted(sources_used.items(), key=lambda x: -x[1])[:10]:
        print(f"  {source}: {count} N values")
    
    print(f"\nSaved to {output_path}")
    
    return total_score

if __name__ == "__main__":
    main()
