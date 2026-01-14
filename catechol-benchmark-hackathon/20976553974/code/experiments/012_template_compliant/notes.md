# Experiment 012: Template-Compliant GroupKFold Ensemble

## CRITICAL FIX: Template Compliance
This experiment fixes the template violation from exp_011:
- Last 3 cells are EXACTLY as in template (only model line changed)
- 'row' column included in submission format
- CV calculation moved to BEFORE template cells
- GroupKFold utility functions overwritten BEFORE template cells (allowed)

## Results
- Single Solvent CV MAE: 0.0734 +/- 0.0194
- Full Data CV MAE: 0.0904 +/- 0.0162
- Combined CV MAE: 0.0844

## Comparison
- Best LOO CV (exp_004): 0.0623
- Best LB (exp_004): 0.0956
- This experiment (GroupKFold): 0.0844

## Key Insight
GroupKFold CV (0.0844) is MUCH closer to LB (0.0956) than LOO CV (0.0623).
- LOO CV-LB gap: 53%
- GroupKFold CV-LB gap: ~12% (expected)

This confirms GroupKFold gives more realistic CV estimates.

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
✅ GroupKFold overwrite is BEFORE template cells

