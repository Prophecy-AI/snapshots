# Experiment 014: Per-Target + Optuna Optimization

## Approach
- Per-target models: HGB for SM, ExtraTrees for Products
- Optuna hyperparameter optimization (50 trials each)
- GroupKFold (5-fold) INTERNALLY for Optuna
- LOO for final submission

## Optuna Results
Best single solvent params:
- hgb_depth: 3, hgb_lr: 0.094, hgb_iter: 326
- etr_depth: 20, etr_n_estimators: 494, etr_min_samples: 8
- GroupKFold CV: 0.078348

Best full data params:
- hgb_depth: 4, hgb_lr: 0.197, hgb_iter: 208
- etr_depth: 6, etr_n_estimators: 188, etr_min_samples: 3
- GroupKFold CV: 0.083789

## Final Results (LOO)
- Single Solvent CV MAE: 0.0719 +/- 0.0334
- Full Data CV MAE: 0.0895 +/- 0.0251
- Combined CV MAE: 0.0834

## Comparison
- Best LOO CV (exp_004): 0.0623
- Best LB (exp_004): 0.0956
- This experiment (LOO): 0.0834

## Analysis
- LOO CV (0.0834) is worse than exp_004 (0.0623)
- But GroupKFold CV during Optuna was more realistic (~0.08)
- The Optuna-found params may generalize better to unseen solvents
- Key insight: Optuna found shallow HGB (depth 3-4) works best

## Template Compliance
✅ Last 3 cells match template exactly
✅ LOO validation with correct fold counts (24/13)
✅ 'row' column included

