"""
Multi-Seed C++ SA Optimization

Run the sa_fast_v2 binary with different random seeds to explore
different local optima. Then ensemble the best per-N solutions.

The evolver has determined that this is the only remaining path forward
since all our snapshots converge to the same local optimum.
"""

import subprocess
import os
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
import time
import json
import glob

getcontext().prec = 30

# Paths
SA_BINARY = "/home/nonroot/snapshots/santa-2025/21328309254/code/sa_fast_v2"
BASELINE = "/home/nonroot/snapshots/santa-2025/21337353543/submission/submission.csv"
OUTPUT_DIR = "/home/code/experiments/016_multi_seed_sa"

# Tree polygon vertices for score computation
TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

def compute_score(csv_path):
    """Compute total score for a submission."""
    df = pd.read_csv(csv_path)
    total = 0
    
    for n in range(1, 201):
        group = df[df['id'].str.startswith(f'{n:03d}_')]
        if len(group) != n:
            continue
        
        # Get tree positions
        trees = []
        for _, row in group.iterrows():
            x = float(str(row['x']).replace('s', ''))
            y = float(str(row['y']).replace('s', ''))
            angle = float(str(row['deg']).replace('s', ''))
            trees.append((x, y, angle))
        
        # Compute bounding box
        all_coords = []
        for x, y, angle in trees:
            rad = angle * np.pi / 180
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            for tx, ty in zip(TX, TY):
                rx = tx * cos_a - ty * sin_a + x
                ry = tx * sin_a + ty * cos_a + y
                all_coords.append((rx, ry))
        
        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        side = max(max(xs) - min(xs), max(ys) - min(ys))
        score = (side ** 2) / n
        total += score
    
    return total

def run_sa_with_seed(seed, iterations=50000):
    """Run SA with a specific seed."""
    output_path = os.path.join(OUTPUT_DIR, f"output_seed{seed}.csv")
    
    cmd = [
        SA_BINARY,
        "-i", BASELINE,
        "-o", output_path,
        "-iter", str(iterations),
        "-seed", str(seed),
        "-threads", "4"
    ]
    
    print(f"Running SA with seed={seed}, iter={iterations}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            return None, None
        
        if os.path.exists(output_path):
            score = compute_score(output_path)
            print(f"  Completed in {elapsed:.2f}s, score={score:.6f}")
            return output_path, score
        else:
            print(f"  Output file not created")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after 300s")
        return None, None
    except Exception as e:
        print(f"  Exception: {e}")
        return None, None

def main():
    print("=" * 60)
    print("MULTI-SEED C++ SA OPTIMIZATION")
    print("=" * 60)
    
    # Compute baseline score
    baseline_score = compute_score(BASELINE)
    print(f"\nBaseline score: {baseline_score:.6f}")
    
    # Run SA with different seeds
    results = []
    num_seeds = 5  # Start with 5 seeds
    iterations = 50000  # 50K iterations per seed
    
    print(f"\nRunning SA with {num_seeds} different seeds, {iterations} iterations each...")
    
    for seed in range(num_seeds):
        output_path, score = run_sa_with_seed(seed, iterations)
        if output_path and score:
            results.append({
                'seed': seed,
                'path': output_path,
                'score': score,
                'improvement': baseline_score - score
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        best_result = min(results, key=lambda x: x['score'])
        print(f"\nBaseline score: {baseline_score:.6f}")
        print(f"Best SA score: {best_result['score']:.6f} (seed={best_result['seed']})")
        print(f"Improvement: {best_result['improvement']:.10f}")
        
        print("\nAll results:")
        for r in sorted(results, key=lambda x: x['score']):
            print(f"  Seed {r['seed']}: {r['score']:.6f} (improvement: {r['improvement']:.10f})")
        
        # Save metrics
        metrics = {
            'baseline_score': baseline_score,
            'best_score': best_result['score'],
            'improvement': best_result['improvement'],
            'num_seeds': num_seeds,
            'iterations': iterations,
            'results': results
        }
        
        with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Copy best result to submission
        if best_result['improvement'] > 1e-8:
            import shutil
            shutil.copy(best_result['path'], '/home/submission/submission.csv')
            print(f"\nCopied best result to /home/submission/submission.csv")
        else:
            # Copy baseline
            import shutil
            shutil.copy(BASELINE, '/home/submission/submission.csv')
            print(f"\nNo improvement found, using baseline")
        
        return best_result['improvement']
    else:
        print("No successful SA runs")
        return 0

if __name__ == "__main__":
    improvement = main()
