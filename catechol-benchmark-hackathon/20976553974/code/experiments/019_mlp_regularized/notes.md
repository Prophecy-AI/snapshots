# Experiment 019: MLP with Strong Regularization + exp_004 Ensemble

## Approach
- Combine exp_004's proven architecture (HGB+ETR per-target) with regularized MLP
- MLP: [256, 128, 64] with dropout=0.4, weight_decay=1e-3
- Ensemble: 0.7 * exp_004_pred + 0.3 * mlp_pred
- Features: Spange + ACS_PCA + Arrhenius kinetics

## Results
- Single Solvent CV MAE: 0.0632 +/- 0.0302
- Full Data CV MAE: 0.0620 +/- 0.0228
- Combined CV MAE: 0.0624

## Comparison
| Experiment | Single | Full | Combined |
|------------|--------|------|----------|
| exp_004/017 | 0.0659 | 0.0603 | 0.0623 |
| exp_018 (DRFP) | 0.0711 | 0.0665 | 0.0681 |
| exp_019 (MLP) | 0.0632 | 0.0620 | 0.0624 |

## Analysis
- CV 0.0624 is essentially the same as exp_004/017's 0.0623
- Single solvent improved slightly (0.0632 vs 0.0659)
- Full data slightly worse (0.0620 vs 0.0603)
- MLP hybrid maintains performance while potentially improving generalization
- The ensemble diversity may help reduce CV-LB gap

## Key Insight
- MLP with strong regularization (dropout=0.4, weight_decay=1e-3) doesn't hurt CV
- The 0.7/0.3 ensemble weights balance exp_004's strength with MLP's potential generalization
- This is a safe submission candidate - same CV as best, potentially better LB

## Template Compliance
✅ Last 3 cells match template exactly
✅ LOO validation with correct fold counts (24/13)
✅ 'row' column included

