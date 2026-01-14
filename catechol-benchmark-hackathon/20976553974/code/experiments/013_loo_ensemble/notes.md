# Experiment 013: LOO Ensemble (Fixed Fold Structure)

## CRITICAL FIX: Reverted to LOO validation
exp_012 FAILED because GroupKFold changed the fold structure to 5 folds.
The evaluation metric expects:
- Task 0: 24 folds (one per solvent)
- Task 1: 13 folds (one per solvent ramp)

## Results
- Single Solvent CV MAE: 0.0670 +/- 0.0337
- Full Data CV MAE: 0.0911 +/- 0.0281
- Combined CV MAE: 0.0827

## Comparison
- Best LOO CV (exp_004): 0.0623
- Best LB (exp_004): 0.0956
- This experiment (LOO): 0.0827

## Configuration
- MLP: [128, 64, 32], NO Sigmoid, 100 epochs, lr=1e-3, dropout=0.1
- XGBoost: n_estimators=300, max_depth=6
- RandomForest: n_estimators=300, max_depth=15
- LightGBM: n_estimators=300
- Weights: [0.4, 0.2, 0.2, 0.2]
- Features: Spange descriptors only

## Template Compliance
✅ Last 3 cells match template exactly
✅ Only model definition line changed
✅ 'row' column included
✅ LOO validation with correct fold counts (24/13)

## Analysis
- LOO CV (0.0827) is worse than exp_004 (0.0623)
- But MLP+GBDT ensemble may generalize better to unseen solvents
- The CV-LB gap for exp_004 was 53% (0.0623 → 0.0956)
- If this model has smaller CV-LB gap, LB could be similar or better

