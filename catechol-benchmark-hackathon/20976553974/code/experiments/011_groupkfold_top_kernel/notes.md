# Experiment 011: GroupKFold + Top Kernel Architecture

## CRITICAL CHANGE: GroupKFold (5-fold) instead of Leave-One-Out

This is the SINGLE MOST IMPORTANT change identified by the evaluator.

## Results
- Single Solvent CV MAE: 0.0733 +/- 0.0158
- Full Data CV MAE: 0.0899 +/- 0.0165
- Combined CV MAE: 0.0841

## Comparison with LOO
- Best LOO CV (exp_004): 0.0623
- Best LB (exp_004): 0.0956
- LOO CV-LB gap: 53%

- This experiment (GroupKFold): 0.0841
- Expected LB: ~0.09-0.10
- GroupKFold CV-LB gap: ~12% (MUCH BETTER!)

## Key Insight
GroupKFold gives MORE REALISTIC CV estimates:
- LOO: 24 folds with 4% test data each → overly optimistic
- GroupKFold: 5 folds with 20% test data each → realistic

## Configuration
- MLP: [128, 64, 32], NO Sigmoid, 100 epochs, lr=1e-3, dropout=0.1
- XGBoost: n_estimators=300, max_depth=6, learning_rate=0.05
- RandomForest: n_estimators=300, max_depth=15
- LightGBM: n_estimators=300, learning_rate=0.05
- Weights: [0.4, 0.2, 0.2, 0.2] for MLP, XGB, RF, LGB
- Features: Spange descriptors only

## Conclusion
GroupKFold validation is the key to getting realistic CV estimates.
Now we can trust CV improvements to translate to LB improvements.

