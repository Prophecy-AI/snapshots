"""
Fix angle normalization issue in submission.
Many angles are outside 0-360 range which may cause Kaggle validation to fail.
"""

import pandas as pd
import numpy as np
import math
import os

def parse_s(x):
    if pd.isna(x):
        return float('nan')
    s = str(x).strip()
    if s.startswith('s'):
        s = s[1:]
    return float(s)

def to_s(v, precision=18):
    if pd.isna(v) or not np.isfinite(v):
        v = 0.0
    return "s" + format(float(v), f".{precision}f").rstrip("0").rstrip(".")

def normalize_angle(deg):
    """Normalize angle to [0, 360) range."""
    deg = deg % 360.0
    if deg < 0:
        deg += 360.0
    return deg

def main():
    print("=" * 60)
    print("FIXING ANGLE NORMALIZATION")
    print("=" * 60)
    
    # Load submission
    df = pd.read_csv('/home/submission/submission.csv')
    print(f"Loaded {len(df)} rows")
    
    # Parse values
    df['x_val'] = df['x'].apply(parse_s)
    df['y_val'] = df['y'].apply(parse_s)
    df['deg_val'] = df['deg'].apply(parse_s)
    
    # Count anomalies before
    anomalies_before = len(df[(df['deg_val'] < 0) | (df['deg_val'] >= 360)])
    print(f"Angles outside [0, 360) before: {anomalies_before}")
    
    # Normalize angles
    df['deg_normalized'] = df['deg_val'].apply(normalize_angle)
    
    # Count anomalies after
    anomalies_after = len(df[(df['deg_normalized'] < 0) | (df['deg_normalized'] >= 360)])
    print(f"Angles outside [0, 360) after: {anomalies_after}")
    
    # Show some examples of fixed angles
    changed = df[df['deg_val'] != df['deg_normalized']]
    print(f"\nFixed {len(changed)} angles:")
    for _, row in changed.head(10).iterrows():
        print(f"  {row['id']}: {row['deg_val']:.2f} -> {row['deg_normalized']:.2f}")
    
    # Create output dataframe
    output = pd.DataFrame({
        'id': df['id'],
        'x': df['x_val'].apply(lambda v: to_s(v)),
        'y': df['y_val'].apply(lambda v: to_s(v)),
        'deg': df['deg_normalized'].apply(lambda v: to_s(v))
    })
    
    # Save
    os.makedirs('/home/code/experiments/033_fix_angles', exist_ok=True)
    output.to_csv('/home/code/experiments/033_fix_angles/submission_fixed.csv', index=False)
    output.to_csv('/home/submission/submission.csv', index=False)
    
    print(f"\nSaved fixed submission")
    
    # Verify
    df_check = pd.read_csv('/home/submission/submission.csv')
    df_check['deg_val'] = df_check['deg'].apply(parse_s)
    anomalies_final = len(df_check[(df_check['deg_val'] < 0) | (df_check['deg_val'] >= 360)])
    print(f"Final check - angles outside [0, 360): {anomalies_final}")
    
    return anomalies_final == 0

if __name__ == "__main__":
    success = main()
    print(f"\nSuccess: {success}")
