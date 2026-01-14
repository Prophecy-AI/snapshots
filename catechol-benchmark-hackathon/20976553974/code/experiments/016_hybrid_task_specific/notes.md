# Experiment 016: Hybrid Model with Task-Specific Configurations

## Approach
- Single solvent: Deep models + MLP + COMBINED features (exp_015 approach)
- Full data: Shallow models + NO MLP + Arrhenius features (exp_004 approach)

## Results
- Single Solvent CV MAE: 0.0647 +/- 0.0328
- Full Data CV MAE: 0.0928 +/- 0.0246
- Combined CV MAE: 0.0830

## Comparison
| Metric | exp_004 | exp_015 | exp_016 |
|--------|---------|---------|---------|
| Single | 0.0659 | 0.0638 | 0.0647 |
| Full | 0.0603 | 0.1027 | 0.0928 |
| Combined | 0.0623 | 0.0891 | 0.0830 |

## Analysis
- Full data improved from exp_015 (0.1027 → 0.0928) but still worse than exp_004 (0.0603)
- Arrhenius features + shallow models helped but not enough
- The exp_004 full data approach may have other factors we're missing

## Configuration
Single solvent:
- Deep models (depth=None) + MLP (weight=0.5)
- COMBINED features (0.8*ACS_PCA + 0.2*Spange)

Full data:
- Shallow models (HGB depth=7, ETR depth=10) + NO MLP
- COMBINED features + Arrhenius kinetics (1/T, ln(t), t*T)

## Template Compliance
✅ Last 3 cells match template exactly
✅ LOO validation with correct fold counts (24/13)
✅ 'row' column included

