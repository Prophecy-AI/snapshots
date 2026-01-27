"""
Genetic Algorithm Optimizer
Attempts to escape local optima through crossover and mutation.
Tests on small N values first before scaling up.
"""

import sys
import os
sys.path.insert(0, '/home/code')

import numpy as np
import pandas as pd
import json
import random
import time
from decimal import Decimal, getcontext
from shapely.geometry import Polygon

from code.tree_geometry import TX, TY, calculate_score, get_tree_vertices_numba
from code.utils import parse_submission, save_submission

getcontext().prec = 30
SCALE = 10**18
MIN_IMPROVEMENT = 0.001

def get_tree_polygon(x, y, angle_deg):
    """Get tree polygon."""
    rx, ry = get_tree_vertices_numba(x, y, angle_deg)
    return Polygon(zip(rx, ry))

def check_overlap_pair(tree1, tree2):
    """Check if two trees overlap."""
    poly1 = get_tree_polygon(tree1[0], tree1[1], tree1[2])
    poly2 = get_tree_polygon(tree2[0], tree2[1], tree2[2])
    if poly1.intersects(poly2):
        if not poly1.touches(poly2):
            inter = poly1.intersection(poly2)
            if inter.area > 1e-12:
                return True
    return False

def has_any_overlap(config):
    """Check if any trees in config overlap."""
    n = len(config)
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap_pair(config[i], config[j]):
                return True
    return False

def calculate_bbox_score(trees):
    """Calculate bounding box score."""
    if len(trees) == 0:
        return 0.0
    
    all_xs = []
    all_ys = []
    for x, y, angle in trees:
        rx, ry = get_tree_vertices_numba(x, y, angle)
        all_xs.extend(rx)
        all_ys.extend(ry)
    
    width = max(all_xs) - min(all_xs)
    height = max(all_ys) - min(all_ys)
    side = max(width, height)
    n = len(trees)
    return (side ** 2) / n

def crossover(parent1, parent2, n):
    """Swap a subset of trees between two configurations."""
    child = [list(t) for t in parent1]
    swap_count = max(1, n // 4)
    swap_indices = random.sample(range(n), swap_count)
    for i in swap_indices:
        child[i] = list(parent2[i])
    return [(t[0], t[1], t[2]) for t in child]

def mutate(config, n, sigma=0.02):
    """Small perturbation to one tree."""
    config = [list(t) for t in config]
    i = random.randint(0, n-1)
    x, y, angle = config[i]
    config[i] = [x + random.gauss(0, sigma),
                 y + random.gauss(0, sigma),
                 angle + random.gauss(0, 2)]
    return [(t[0], t[1], t[2]) for t in config]

def repair_overlaps(config, n, max_attempts=50):
    """Try to fix overlaps by small perturbations."""
    config = [list(t) for t in config]
    
    for attempt in range(max_attempts):
        if not has_any_overlap(config):
            return [(t[0], t[1], t[2]) for t in config], True
        
        # Find overlapping pair and perturb one
        found = False
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_pair(config[i], config[j]):
                    # Perturb tree j
                    x, y, angle = config[j]
                    config[j] = [x + random.gauss(0, 0.05),
                                 y + random.gauss(0, 0.05),
                                 angle]
                    found = True
                    break
            if found:
                break
    
    return [(t[0], t[1], t[2]) for t in config], False

def run_ga_for_n(baseline_config, n, pop_size=20, generations=50):
    """Run genetic algorithm for a single N value."""
    baseline_score = calculate_bbox_score(baseline_config)
    
    # Initialize population from baseline + random perturbations
    population = [baseline_config]
    for _ in range(pop_size - 1):
        mutated = mutate(baseline_config, n, sigma=0.1)
        repaired, ok = repair_overlaps(mutated, n)
        if ok:
            population.append(repaired)
        else:
            population.append(baseline_config)  # Fallback
    
    best_score = baseline_score
    best_config = baseline_config
    
    for gen in range(generations):
        # Evaluate fitness
        scores = [calculate_bbox_score(p) for p in population]
        
        # Selection (tournament)
        new_pop = []
        for _ in range(len(population)):
            i, j = random.sample(range(len(population)), 2)
            winner = population[i] if scores[i] < scores[j] else population[j]
            new_pop.append(winner)
        
        # Crossover
        for i in range(0, len(new_pop)-1, 2):
            if random.random() < 0.7:
                child1 = crossover(new_pop[i], new_pop[i+1], n)
                child2 = crossover(new_pop[i+1], new_pop[i], n)
                child1, ok1 = repair_overlaps(child1, n)
                child2, ok2 = repair_overlaps(child2, n)
                if ok1:
                    new_pop[i] = child1
                if ok2:
                    new_pop[i+1] = child2
        
        # Mutation
        for i in range(len(new_pop)):
            if random.random() < 0.3:
                mutated = mutate(new_pop[i], n)
                repaired, ok = repair_overlaps(mutated, n)
                if ok:
                    new_pop[i] = repaired
        
        population = new_pop
        
        # Track best
        for p in population:
            if not has_any_overlap(p):
                score = calculate_bbox_score(p)
                if score < best_score:
                    best_score = score
                    best_config = p
    
    return best_config, best_score, baseline_score

def main():
    print("=" * 70)
    print("GENETIC ALGORITHM OPTIMIZER")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load baseline (exp_016)
    print("\nLoading exp_016 baseline...")
    baseline_df = pd.read_csv('/home/code/experiments/016_mega_ensemble_external/submission.csv')
    baseline_configs = parse_submission(baseline_df)
    baseline_scores = {n: calculate_score(baseline_configs[n]) for n in range(1, 201)}
    baseline_total = sum(baseline_scores.values())
    print(f"Baseline total: {baseline_total:.6f}")
    
    # Test on small N values first
    test_n_values = [5, 10, 15, 20, 25, 30]
    print(f"\nTesting GA on N values: {test_n_values}")
    print("=" * 70)
    
    improvements = []
    
    for n in test_n_values:
        print(f"\n=== N={n} ===")
        best_config, best_score, orig_score = run_ga_for_n(
            baseline_configs[n], n, pop_size=20, generations=50
        )
        
        improvement = orig_score - best_score
        print(f"  Baseline: {orig_score:.6f}")
        print(f"  GA Best:  {best_score:.6f}")
        print(f"  Improvement: {improvement:.6f}")
        
        if improvement >= MIN_IMPROVEMENT:
            improvements.append((n, improvement, best_config))
            print(f"  ✅ SIGNIFICANT IMPROVEMENT!")
        else:
            print(f"  ❌ No significant improvement (< {MIN_IMPROVEMENT})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"N values tested: {len(test_n_values)}")
    print(f"N values with improvements >= {MIN_IMPROVEMENT}: {len(improvements)}")
    
    if improvements:
        print("\nImprovements found:")
        for n, imp, _ in improvements:
            print(f"  N={n}: {imp:.6f}")
    else:
        print("\nNo significant improvements found on test N values.")
        print("GA cannot escape the local optimum for these N values.")
    
    # If we found improvements, run on all N values
    if improvements:
        print("\n" + "=" * 70)
        print("RUNNING GA ON ALL N VALUES")
        print("=" * 70)
        
        best_per_n = {}
        all_improvements = []
        
        for n in range(1, 201):
            best_config, best_score, orig_score = run_ga_for_n(
                baseline_configs[n], n, pop_size=15, generations=30
            )
            
            improvement = orig_score - best_score
            
            if improvement >= MIN_IMPROVEMENT and not has_any_overlap(best_config):
                best_per_n[n] = best_config
                all_improvements.append((n, improvement))
                print(f"✅ N={n}: IMPROVED by {improvement:.6f}")
            else:
                best_per_n[n] = baseline_configs[n]
        
        # Calculate final score
        final_total = sum(calculate_score(best_per_n[n]) for n in range(1, 201))
        total_improvement = baseline_total - final_total
        
        print(f"\nBaseline total: {baseline_total:.6f}")
        print(f"Final total: {final_total:.6f}")
        print(f"Total improvement: {total_improvement:.6f}")
        
        # Save submission
        save_submission(best_per_n, 'submission.csv')
        print("\nSaved submission.csv")
        
        # Save metrics
        metrics = {
            'cv_score': final_total,
            'baseline_score': baseline_total,
            'improvement': total_improvement,
            'num_improvements': len(all_improvements),
            'notes': 'Genetic algorithm optimization'
        }
    else:
        # No improvements - use baseline
        final_total = baseline_total
        
        # Save baseline as submission
        save_submission(baseline_configs, 'submission.csv')
        print("\nNo improvements found - saved baseline as submission.csv")
        
        # Save metrics
        metrics = {
            'cv_score': final_total,
            'baseline_score': baseline_total,
            'improvement': 0.0,
            'num_improvements': 0,
            'notes': 'GA found no improvements on test N values - using baseline'
        }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFinal CV Score: {final_total:.6f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Copy to submission folder
    import shutil
    shutil.copy('submission.csv', '/home/submission/submission.csv')
    print("Copied submission to /home/submission/")
    
    return final_total

if __name__ == '__main__':
    os.chdir('/home/code/experiments/018_genetic_algorithm')
    random.seed(42)
    np.random.seed(42)
    main()
