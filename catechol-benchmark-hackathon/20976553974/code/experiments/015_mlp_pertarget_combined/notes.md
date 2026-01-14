# Experiment 015: Per-Target + MLP Hybrid with COMBINED Features

## Key Improvements
1. COMBINED features (0.8*ACS_PCA + 0.2*Spange) like exp_004
2. DEEP models (depth=None) - NOT shallow like exp_014
3. MLP component for non-linear patterns
4. Optuna for ensemble WEIGHTS

## Optuna Results
- Best MLP weight: 0.5012
- Best GroupKFold CV: 0.0702

## Final Results (LOO)
- Single Solvent CV MAE: 0.0638 +/- 0.0325 (EXCELLENT - nearly matches exp_004!)
- Full Data CV MAE: 0.1027 +/- 0.0260 (worse than exp_004)
- Combined CV MAE: 0.0891

## Comparison
- Best LOO CV (exp_004): 0.0623
- Best LB (exp_004): 0.0956
- This experiment (LOO): 0.0891

## Analysis
- Single solvent performance (0.0638) nearly matches exp_004 (0.0623)!
- Full data performance (0.1027) is worse - may need different tuning
- MLP weight of 0.5 suggests MLP and GBDT contribute equally
- COMBINED features + DEEP models + MLP hybrid works well for single solvents

## Configuration
- Features: COMBINED (0.8*ACS_PCA + 0.2*Spange)
- MLP: [128, 64, 32], BatchNorm, ReLU, Dropout(0.2), Sigmoid
- HGB: depth=None (unlimited), lr=0.1, max_iter=200
- ETR: depth=None (unlimited), n_estimators=200
- Weights: MLP=0.50, HGB=0.25, ETR=0.25

## Template Compliance
✅ Last 3 cells match template exactly
✅ LOO validation with correct fold counts (24/13)
✅ 'row' column included

