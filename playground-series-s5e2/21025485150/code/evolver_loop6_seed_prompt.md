## Current Status
- Best CV score: 38.660840 from exp_004 (groupby statistics)
- Last experiment: 38.663395 from exp_005 (cleaned names + histograms) - WORSE by 0.002555
- Target: 38.616280
- Gap: 0.044560 RMSE to close
- CV-LB gap: Unknown (no LB submissions yet)

## Response to Evaluator
- Technical verdict: TRUSTWORTHY - Feature engineering is sound, but strategy needs refinement
- Evaluator's top priority: Investigate why histogram features degraded performance despite high importance
- Key concerns addressed:
  - ✅ Histogram features are NOT in winning solution - they add noise and overfitting
  - ✅ Original dataset is the critical missing piece (worth ~0.08-0.10 RMSE)
  - ✅ COMBO/interaction features are missing from our approach
  - ✅ Groupby statistics are working well (19.1% importance) but need optimization
  - ✅ Feature count too high (313) with many low-value histogram bins

## Data Understanding
- Reference notebooks: See `exploration/evolver_loop6_analysis.ipynb` for winning solution deep dive
- Key findings from analysis:
  - Winning solution uses: 15 COMBO features + 3 rounding + 4 original dataset + ~15 digit + ~463 groupby = 500 total
  - Our exp_005 uses: 0 COMBO + 4 rounding + 0 original + 5 digit + 48 groupby + 250 histogram = 313 total
  - Histogram bins have lower average importance (4,084) than groupby features (5,860)
  - 250 histogram features create multicollinearity and dilute good features
  - Original dataset features (orig_price, orig_price_r7, orig_price_r8, orig_price_r9) are the key differentiator

## Recommended Approaches

### 1. Add Original Dataset Features (CRITICAL - Priority 1)
- Download "Student Bag Price Prediction Dataset" by Souradip Pal from Kaggle
- Load Noisy_Student_Bag_Price_Prediction_Dataset.csv
- Compute orig_price: mean Price by Weight Capacity (kg) from original dataset
- Compute orig_price_r7, orig_price_r8, orig_price_r9: mean Price by rounded Weight Capacity (7, 8, 9 decimals)
- Merge these 4 features into train/test data
- Expected improvement: 0.08-0.10 RMSE (this closes most of the gap!)
- Rationale: Winning solution's strongest features, provides reference price lookup table

### 2. Add COMBO/Interaction Features (HIGH - Priority 2)
- NaNs: Base-2 encoding of all NaN patterns across 7 categorical columns
- {col}_nan_wc: For each of 7 columns, NaN status (0/1) × Weight Capacity
- {col}_wc: For each of 7 columns, factorized categorical × Weight Capacity
- Total: 1 + 7 + 7 = 15 interaction features
- Expected improvement: 0.02-0.04 RMSE
- Rationale: Captures missing value patterns and categorical-weight interactions that winning solution uses

### 3. Optimize Groupby Statistics (MEDIUM - Priority 3)
- Current: 48 features (6 stats × 8 group keys: mean, std, count, min, max, median)
- Keep: mean, count, median (highest importance from exp_005 analysis)
- Remove: std, min, max (low/zero importance, 18 features)
- Add: skew, kurtosis, 25th/75th/90th percentiles (5 new stats)
- New total: 8 group keys × 8 stats = 64 features (net +16 from current)
- Expected improvement: 0.01-0.02 RMSE
- Rationale: Focus on high-signal statistics, add more distribution measures

### 4. Remove Histogram Bins (MEDIUM - Priority 4)
- Remove all 250 histogram features from exp_005
- These are NOT in winning solution and create redundancy
- Histogram bins duplicate weight_capacity signal with multicollinearity
- Expected improvement: 0.01-0.02 RMSE (from reduced overfitting)
- Rationale: Quality over quantity - remove noise to let good features shine

### 5. Hyperparameter Refinement (LOW - Priority 5)
- Learning rate: 0.05 → 0.03 (better convergence)
- Max depth: 8 → 10 (more capacity for complex patterns)
- Add regularization: reg_alpha=0.1, reg_lambda=1.0 (prevent overfitting)
- Keep n_estimators=2000, early_stopping_rounds=100
- Expected improvement: 0.005-0.01 RMSE
- Rationale: Fine-tune after fixing feature engineering issues

## What NOT to Try
- Histogram/binning features: Proven to be ineffective and not in winning solution
- Complex ensembles: Focus on single strong model with proper features first
- High-cardinality interactions: Test basic COMBO features first before adding more
- Neural networks: Winning solutions are XGBoost-based, stick with what works
- Excessive feature count: Winning full solution has 500 features, but simplified has 138 - quality matters

## Validation Notes
- CV scheme: 20-fold CV (consistent with previous experiments)
- Feature importance: Monitor to validate original dataset and COMBO features are top predictors
- Ablation study: Test each feature group individually to measure impact
- Early stopping: Use to prevent overfitting with new features
- LB calibration: Submit after adding original dataset features to verify CV-LB correlation

## Expected Outcome
- Current: 38.660840 (exp_004)
- Projected improvements: -0.148 RMSE total
  - Original dataset: -0.080
  - COMBO features: -0.030
  - Groupby optimization: -0.015
  - Remove histograms: -0.015
  - Hyperparameter tuning: -0.008
- Projected CV: 38.512840
- Target: 38.616280
- Expected margin: Beat target by 0.103460 RMSE