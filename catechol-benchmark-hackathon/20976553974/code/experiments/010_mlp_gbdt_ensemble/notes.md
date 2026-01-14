# Experiment 010: MLP + GBDT Ensemble (Like Top Kernel)

## Results
- Single Solvent CV MAE: 0.0686 +/- 0.0366
- Full Data CV MAE: 0.0660 +/- 0.0221
- Combined CV MAE: 0.0669

## Comparison
- Best CV (exp_004): 0.0623
- This experiment: 0.0669 (7.4% worse)

## Configuration
- MLP: [128, 64, 32] hidden dims, BatchNorm + ReLU + Dropout(0.2), Sigmoid output
- XGBoost: n_estimators=200, max_depth=6, learning_rate=0.05
- RandomForest: n_estimators=200, max_depth=10
- LightGBM: n_estimators=200, max_depth=6, learning_rate=0.05
- Ensemble weights: [0.35, 0.25, 0.25, 0.15] for MLP, XGB, RF, LGB
- Features: Spange descriptors only

## Analysis
- MLP + GBDT ensemble is slightly worse than per-target HGB+ETR
- The MLP may need more training epochs or different architecture
- Fixed weights may be suboptimal
- Next: Try Optuna for weight optimization or different MLP architecture

