#!/usr/bin/env python3
"""
Ensemble Optimizer
https://www.kaggle.com/code/hvanphucs112/bbox3-ensemble-update
"""

import csv
import math
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# Tree shape constants (from C++ code)
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

def get_polygon_bounds(cx: float, cy: float, deg: float) -> Tuple[float, float, float, float]:
   """Calculate bounding box of rotated tree polygon"""
   rad = deg * math.pi / 180.0
   s = math.sin(rad)
   c = math.cos(rad)
   
   x_coords = []
   y_coords = []
   
   for i in range(len(TX)):
       x = TX[i] * c - TY[i] * s + cx
       y = TX[i] * s + TY[i] * c + cy
       x_coords.append(x)
       y_coords.append(y)
   
   return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

def calculate_score(trees: List[Tuple[int, float, float, float]]) -> Tuple[float, float, float, float]:
   """
   Calculate score for a configuration
   Returns: (score, side, width, height)
   """
   if not trees:
       return float('inf'), 0, 0, 0
   
   global_x_min = float('inf')
   global_x_max = float('-inf')
   global_y_min = float('inf')
   global_y_max = float('-inf')
   
   for idx, cx, cy, deg in trees:
       x_min, x_max, y_min, y_max = get_polygon_bounds(cx, cy, deg)
       global_x_min = min(global_x_min, x_min)
       global_x_max = max(global_x_max, x_max)
       global_y_min = min(global_y_min, y_min)
       global_y_max = max(global_y_max, y_max)
   
   width = global_x_max - global_x_min
   height = global_y_max - global_y_min
   side = max(width, height)
   score = side * side / len(trees)
   
   return score, side, width, height

def load_submission(filepath: str) -> Dict[int, List[Tuple[int, float, float, float]]]:
   """
   Load submission file
   Returns: dict mapping n -> list of (idx, x, y, deg)
   """
   configurations = defaultdict(list)
   
   try:
       with open(filepath, 'r') as f:
           reader = csv.DictReader(f)
           for row in reader:
               # Parse id
               id_parts = row['id'].split('_')
               n = int(id_parts[0])
               idx = int(id_parts[1])
               
               # Parse coordinates (remove 's' prefix if present)
               x = float(row['x'].replace('s', ''))
               y = float(row['y'].replace('s', ''))
               deg = float(row['deg'].replace('s', ''))
               
               configurations[n].append((idx, x, y, deg))
       
       # Sort by index
       for n in configurations:
           configurations[n].sort(key=lambda t: t[0])
       
       return dict(configurations)
   
   except Exception as e:
       print(f"Error loading {filepath}: {e}")
       return {}

def analyze_submission(filepath: str, configurations: Dict[int, List]) -> Dict[int, Tuple]:
   """
   Analyze a submission file
   Returns: dict mapping n -> (score, side, width, height)
   """
   results = {}
   
   for n, trees in configurations.items():
       if len(trees) != n:
           print(f"  WARNING: n={n} has {len(trees)} trees (expected {n})")
           continue
       
       score, side, width, height = calculate_score(trees)
       results[n] = (score, side, width, height)
   
   return results

def create_ensemble(submissions: Dict[str, Dict[int, List]]) -> Dict[int, Tuple[List, str, float]]:
   """
   Create ensemble by selecting best configuration for each n
   Returns: dict mapping n -> (best_trees, source_file, score)
   """
   ensemble = {}
   
   # Get all n values
   all_n = set()
   for configs in submissions.values():
       all_n.update(configs.keys())
   
   # For each n, find best configuration
   for n in sorted(all_n):
       best_score = float('inf')
       best_trees = None
       best_source = None
       
       for filepath, configs in submissions.items():
           if n not in configs:
               continue
           
           trees = configs[n]
           if len(trees) != n:
               continue
           
           score, side, width, height = calculate_score(trees)
           
           if score < best_score:
               best_score = score
               best_trees = trees
               best_source = filepath
       
       if best_trees:
           ensemble[n] = (best_trees, best_source, best_score)
   
   return ensemble

def save_ensemble(ensemble: Dict[int, Tuple], output_path: str):
   """Save ensemble submission"""
   with open(output_path, 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(['id', 'x', 'y', 'deg'])
       
       for n in sorted(ensemble.keys()):
           trees, _, _ = ensemble[n]
           for idx, x, y, deg in trees:
               row_id = f"{n:03d}_{idx}"
               writer.writerow([row_id, f's{x:.17f}', f's{y:.17f}', f's{deg:.17f}'])

def print_comparison(submissions: Dict[str, Dict[int, List]], ensemble: Dict[int, Tuple]):
   """Print detailed comparison"""
   print("\n" + "="*80)
   print("DETAILED COMPARISON BY N")
   print("="*80)
   
   all_n = sorted(set(
       n for configs in submissions.values() 
       for n in configs.keys()
   ))
   
   # Prepare data for each n
   for n in all_n:
       print(f"\n{'─'*80}")
       print(f"n = {n}")
       print(f"{'─'*80}")
       
       # Collect scores from all submissions
       scores_data = []
       for filepath, configs in submissions.items():
           if n in configs and len(configs[n]) == n:
               score, side, width, height = calculate_score(configs[n])
               basename = os.path.basename(filepath)
               scores_data.append((basename, score, side))
       
       # Sort by score
       scores_data.sort(key=lambda x: x[1])
       
       # Print table
       print(f"{'Source':<30} {'Score':<20} {'Side':<20} {'Status'}")
       print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*10}")
       
       for i, (source, score, side) in enumerate(scores_data):
           status = "✅ BEST" if i == 0 else ""
           print(f"{source:<30} {score:<20.15f} {side:<20.15f} {status}")
       
       # Show ensemble choice
       if n in ensemble:
           _, best_source, best_score = ensemble[n]
           print(f"\n→ Ensemble choice: {os.path.basename(best_source)} (score: {best_score:.15f})")
       
       # Calculate improvement range
       if len(scores_data) > 1:
           worst_score = scores_data[-1][1]
           best_score = scores_data[0][1]
           improvement = (worst_score - best_score) / worst_score * 100
           print(f"→ Improvement range: {improvement:.4f}%")

def print_summary(submissions: Dict[str, Dict[int, List]], ensemble: Dict[int, Tuple]):
   """Print summary statistics"""
   print("\n" + "="*80)
   print("SUMMARY STATISTICS")
   print("="*80)
   
   # Per-file statistics
   print("\nPer-file statistics:")
   print(f"{'File':<30} {'Total n':<10} {'Avg Score':<20} {'Best Count'}")
   print(f"{'-'*30} {'-'*10} {'-'*20} {'-'*10}")
   
   for filepath, configs in sorted(submissions.items()):
       basename = os.path.basename(filepath)
       
       # Calculate average score
       total_score = 0
       count = 0
       for n, trees in configs.items():
           if len(trees) == n:
               score, _, _, _ = calculate_score(trees)
               total_score += score
               count += 1
       
       avg_score = total_score / count if count > 0 else 0
       
       # Count how many times this file was chosen as best
       best_count = sum(1 for _, source, _ in ensemble.values() 
                       if source == filepath)
       
       print(f"{basename:<30} {count:<10} {avg_score:<20.10f} {best_count}")
   
   # Ensemble statistics
   print("\n" + "-"*80)
   print("Ensemble statistics:")
   
   total_score = sum(score for _, _, score in ensemble.values())
   avg_score = total_score / len(ensemble) if ensemble else 0

  
   print(f"  Total n values: {len(ensemble)}")
   print(f"  Total score: {total_score}")
   print(f"  Average score:  {avg_score:.10f}")
   
   # Count improvements
   print("\nSource distribution in ensemble:")
   source_counts = defaultdict(int)
   for _, source, _ in ensemble.values():
       basename = os.path.basename(source)
       source_counts[basename] += 1
   
   for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
       pct = count / len(ensemble) * 100
       print(f"  {source:<30} {count:>3} / {len(ensemble)} ({pct:>5.1f}%)")

def print_highlights(ensemble: Dict[int, Tuple]):
   """Print highlights - best and worst scores"""
   print("\n" + "="*80)
   print("HIGHLIGHTS")
   print("="*80)
   
   # Sort by score
   sorted_n = sorted(ensemble.items(), key=lambda x: x[1][2])
   
   print("\n🏆 TOP 10 BEST SCORES:")
   print(f"{'n':<5} {'Score':<20} {'Source'}")
   print(f"{'-'*5} {'-'*20} {'-'*40}")
   for i, (n, (_, source, score)) in enumerate(sorted_n[:10]):
       basename = os.path.basename(source)
       print(f"{n:<5} {score:<20.15f} {basename}")
   
   print("\n⚠️  TOP 10 WORST SCORES:")
   print(f"{'n':<5} {'Score':<20} {'Source'}")
   print(f"{'-'*5} {'-'*20} {'-'*40}")
   for i, (n, (_, source, score)) in enumerate(sorted_n[-10:]):
       basename = os.path.basename(source)
       print(f"{n:<5} {score:<20.15f} {basename}")

def main():
   import argparse
   
   parser = argparse.ArgumentParser(description='Ensemble multiple submissions')
   parser.add_argument('-d', '--dir', default='submissions', 
                      help='Directory containing submission files')
   parser.add_argument('-o', '--output', default='submission_ensemble.csv',
                      help='Output ensemble file')
   parser.add_argument('--verbose', action='store_true',
                      help='Show detailed comparison')
   
   args = parser.parse_args()    
   # Find all CSV files
   csv_files = glob.glob(os.path.join(args.dir, '*.csv'))
   
   if not csv_files:
       print(f"❌ No CSV files found in {args.dir}")
       return
   
   print(f"📁 Found {len(csv_files)} submission files:")
   for f in csv_files:
       size = os.path.getsize(f) / 1024
       print(f"   - {os.path.basename(f):<30} ({size:>8.1f} KB)")
   
   # Load all submissions
   print(f"\n📊 Loading submissions...")
   submissions = {}
   for filepath in csv_files:
       basename = os.path.basename(filepath)
       print(f"   Loading {basename}...", end=' ')
       configs = load_submission(filepath)
       if configs:
           submissions[filepath] = configs
           print(f"✅ ({len(configs)} groups)")
       else:
           print("❌ Failed")
   
   if not submissions:
       print("\n❌ No valid submissions loaded")
       return
   
   print(f"\n✅ Loaded {len(submissions)} submissions successfully")
   
   # Create ensemble
   print(f"\n🔧 Creating ensemble (selecting best for each n)...")
   ensemble = create_ensemble(submissions)
   
   print(f"✅ Ensemble created with {len(ensemble)} groups")
   
   # Save ensemble
   print(f"\n💾 Saving to {args.output}...")
   save_ensemble(ensemble, args.output)
   print(f"✅ Saved!")
   
   # Print statistics
   print_summary(submissions, ensemble)
   print_highlights(ensemble)
   
   if args.verbose:
       print_comparison(submissions, ensemble)
   else:
       print("\n💡 Use --verbose flag to see detailed comparison for each n")
   
   # Final summary
   print("\n" + "="*80)
   print("✅ ENSEMBLE COMPLETE!")
   print("="*80)
   print(f"\n📄 Output: {args.output}")
   print(f"📊 Total groups: {len(ensemble)}")
   
   # Calculate overall improvement
   total_improvement = 0
   count = 0
   for n in ensemble.keys():
       scores = []
       for filepath, configs in submissions.items():
           if n in configs and len(configs[n]) == n:
               score, _, _, _ = calculate_score(configs[n])
               scores.append(score)
       
       if len(scores) > 1:
           best = min(scores)
           worst = max(scores)
           if worst > 0:
               improvement = (worst - best) / worst * 100
               total_improvement += improvement
               count += 1
   
   if count > 0:
       avg_improvement = total_improvement / count
       print(f"📈 Average improvement per group: {avg_improvement:.4f}%")
   
   print("\n🎯 Next steps:")
   print(f"Review the ensemble: {args.output}")
   print()

if __name__ == '__main__':
   main()
