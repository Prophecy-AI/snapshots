"""Utility functions for Santa 2025."""

import pandas as pd
import numpy as np
from collections import defaultdict

def parse_submission(df):
    """Parse submission CSV into dict of n -> list of (x, y, angle) tuples."""
    configs = defaultdict(list)
    
    for _, row in df.iterrows():
        parts = row['id'].split('_')
        n = int(parts[0])
        x = float(str(row['x']).replace('s', ''))
        y = float(str(row['y']).replace('s', ''))
        deg = float(str(row['deg']).replace('s', ''))
        configs[n].append((x, y, deg))
    
    return dict(configs)

def load_submission(filepath):
    """Load and parse a submission CSV file."""
    df = pd.read_csv(filepath)
    return parse_submission(df)

def format_submission(configs):
    """Format configurations as submission DataFrame.
    
    Args:
        configs: dict of n -> list of (x, y, angle) tuples
    
    Returns:
        DataFrame with columns [id, x, y, deg]
    """
    rows = []
    for n in range(1, 201):
        for i, (x, y, deg) in enumerate(configs[n]):
            rows.append({
                'id': f'{n:03d}_{i}',
                'x': f's{x:.20f}',
                'y': f's{y:.20f}',
                'deg': f's{deg:.20f}'
            })
    return pd.DataFrame(rows)

def save_submission(configs, filepath):
    """Save configurations to submission CSV file."""
    df = format_submission(configs)
    df.to_csv(filepath, index=False)
    return df

def calculate_total_score(configs, score_fn):
    """Calculate total score for all N configurations.
    
    Args:
        configs: dict of n -> list of (x, y, angle) tuples
        score_fn: function to calculate score for a list of trees
    
    Returns:
        total_score: sum of scores for all N
        scores_by_n: dict of n -> score
    """
    scores_by_n = {}
    total = 0.0
    
    for n in range(1, 201):
        score = score_fn(configs[n])
        scores_by_n[n] = score
        total += score
    
    return total, scores_by_n

def compare_configs(baseline_configs, new_configs, score_fn, threshold=1e-6):
    """Compare two configurations and find improvements.
    
    Args:
        baseline_configs: dict of n -> list of (x, y, angle) tuples
        new_configs: dict of n -> list of (x, y, angle) tuples
        score_fn: function to calculate score for a list of trees
        threshold: minimum improvement to report
    
    Returns:
        improvements: list of (n, baseline_score, new_score, improvement) tuples
    """
    improvements = []
    
    for n in range(1, 201):
        baseline_score = score_fn(baseline_configs[n])
        new_score = score_fn(new_configs[n])
        improvement = baseline_score - new_score
        
        if improvement > threshold:
            improvements.append((n, baseline_score, new_score, improvement))
    
    return improvements
