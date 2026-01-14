# Experiment 017: Replicate exp_004's EXACT Architecture

## CRITICAL FIX
exp_016 used FEATURE combination, but exp_004 uses PREDICTION combination.

## Results
- Single Solvent CV MAE: 0.0659 +/- 0.0321
- Full Data CV MAE: 0.0603 +/- 0.0219
- Combined CV MAE: 0.0623

## Comparison with exp_004
| Metric | exp_004 | exp_017 | Match? |
|--------|---------|---------|--------|
| Single | 0.0659 | 0.0659 | ✅ YES |
| Full | 0.0603 | 0.0603 | ✅ YES |
| Combined | 0.0623 | 0.0623 | ✅ YES |

## Key Architecture (EXACT replication)
1. Train SEPARATE models on spange and acs_pca features for EACH target
2. Combine PREDICTIONS: 0.8 * acs_pred + 0.2 * spange_pred
3. HGB for SM (depth=7, iter=700, lr=0.04)
4. ETR for Products (n_estimators=500, depth=10, min_samples_leaf=2)
5. Arrhenius kinetics features (inv_temp, log_time, interaction)
6. NO TTA

## Why exp_016 Failed
exp_016 used FEATURE combination:
- combined_features = 0.8*acs + 0.2*spange
- Train single model on combined_features
- Result: Full CV 0.0928 (54% worse!)

exp_004/017 uses PREDICTION combination:
- Train separate models on each feature set
- pred = 0.8*acs_pred + 0.2*spange_pred
- Result: Full CV 0.0603 (CORRECT!)

## Template Compliance
✅ Last 3 cells match template exactly
✅ LOO validation with correct fold counts (24/13)
✅ 'row' column included

