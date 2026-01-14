# Experiment 018: DRFP-Based Ensemble with Prediction Combination

## Approach
- Added DRFP-PCA (20 dims) as THIRD feature set alongside Spange and ACS_PCA
- Train separate models on each feature set (like exp_004)
- Combine PREDICTIONS: 0.35 * drfp_pred + 0.45 * acs_pred + 0.20 * spange_pred

## Results
- Single Solvent CV MAE: 0.0711 +/- 0.0344
- Full Data CV MAE: 0.0665 +/- 0.0228
- Combined CV MAE: 0.0681

## Comparison with exp_004/017
| Metric | exp_004/017 | exp_018 | Change |
|--------|-------------|---------|--------|
| Single | 0.0659 | 0.0711 | +7.9% WORSE |
| Full | 0.0603 | 0.0665 | +10.3% WORSE |
| Combined | 0.0623 | 0.0681 | +9.3% WORSE |

## Analysis
- DRFP features HURT performance, not helped
- The paper's GNN approach may be fundamentally different from tabular DRFP
- DRFP is very sparse (97.43% zeros) - may not work well with tree-based models
- The weights (0.35 drfp, 0.45 acs, 0.20 spange) may be suboptimal

## Key Insight
- DRFP features alone don't capture the chemical patterns needed
- The paper's success may come from GNN architecture, not just DRFP features
- Tabular ensembles may have hit their ceiling at ~0.06 CV

## Template Compliance
✅ Last 3 cells match template exactly
✅ LOO validation with correct fold counts (24/13)
✅ 'row' column included

