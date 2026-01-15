## Current Status
- **Best CV score**: 0.02047 from exp_000 (baseline XGBoost)
- **Target score**: 0.058410 (we need to bridge a significant gap)
- **Key issue**: Our CV is TOO GOOD compared to winning solutions (0.058-0.059), indicating synthetic data is too simple OR we need much more sophisticated modeling

## Response to Evaluator
No evaluator feedback yet - this is the first iteration.

## Data Understanding
- Competition: Playground Series S5E5 - Predict Calorie Expenditure
- Metric: RMSLE (Root Mean Squared Logarithmic Error)
- Target: Calories (continuous)
- Features: Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
- **Critical insight from winners**: CV-LB correlation is unstable. Trust CV optimization over LB feedback.

Reference analysis: See `exploration/evolver_loop1_analysis.ipynb` for detailed winning solution analysis.

## Key Patterns to Exploit (from winning solutions)
1. **Ensemble Diversity**: Winners used 7-12+ diverse models (XGBoost, CatBoost, LGBM, Neural Networks, Linear Regression, Autogluon)
2. **Advanced Feature Engineering**:
   - Target encoding (with proper cross-validation to avoid leakage)
   - Binned features for CatBoost
   - GroupBy z-score features
   - Product features (all pairs of numerical features)
   - Residual modeling (NN on LinearRegression residuals, XGB on NN residuals)
3. **Ensembling Methods**: Hill climbing (Chris Deotte 1st place) and Ridge regression (AngelosMar 4th place)
4. **Computational Approach**: GPU acceleration for hill climbing with hundreds of models

## Recommended Approaches (Priority Order)

### Priority 1: Generate Diverse Base Models
**Goal**: Create 7-10 diverse models with different algorithms and feature sets

1. **CatBoost Model**
   - Use binned features (create binned versions of numerical features)
   - Handle categorical features natively
   - Train with different hyperparameters than XGBoost
   - Expected CV: ~0.055-0.065

2. **LightGBM Model**
   - Use histogram-based approach
   - Try different feature subsets
   - Experiment with GOSS (Gradient-based One-Side Sampling)
   - Expected CV: ~0.055-0.065

3. **Neural Network (MLP)**
   - Use original features + basic interactions
   - 2-3 hidden layers with dropout
   - Early stopping to prevent overfitting
   - Expected CV: ~0.060-0.070

4. **Linear Regression with Advanced Features**
   - Create 200-400 engineered features
   - Include target-encoded features (with cross-validation)
   - Include groupby z-score features
   - Include product/ratio features
   - Use Ridge regularization
   - Expected CV: ~0.058-0.068

5. **XGBoost with Target Encoding**
   - Re-train XGBoost with target-encoded features
   - Compare performance to baseline XGBoost
   - Expected CV: ~0.050-0.060

### Priority 2: Advanced Feature Engineering
**Goal**: Implement sophisticated features used by winners

1. **Target Encoding** (Critical - must use cross-validation)
   - Encode categorical features (Sex) using target mean
   - Use scikit-learn's TargetEncoder with internal cross-fitting
   - Add smoothing to prevent overfitting on rare categories
   - CRITICAL: Must compute encoding on out-of-fold data only

2. **Binned Features for CatBoost**
   - Create binned versions of: Age, Height, Weight, Duration, Heart_Rate, Body_Temp
   - Use 10-20 bins per feature
   - These help CatBoost handle numerical features more effectively

3. **GroupBy Z-Score Features**
   - Group by Sex and compute z-scores for numerical features
   - Example: (Weight - mean(Weight by Sex)) / std(Weight by Sex)
   - Creates relative positioning features

4. **Product Features (All Pairs)**
   - Create interaction terms: Height*Weight, Weight*Duration, Duration*Heart_Rate, etc.
   - Include all pairwise products of numerical features
   - Winners found these very effective

5. **Residual Modeling Features**
   - Train Linear Regression first, get residuals
   - Train Neural Network on residuals
   - Train XGBoost on NN residuals
   - This sequential approach captures different patterns

### Priority 3: Create OOF Predictions for Ensemble
**Goal**: Generate out-of-fold predictions for all models

1. Use 5-fold CV with seed 42 (consistent with winners)
2. Save OOF predictions for each model
3. Save test predictions for each model
4. Organize predictions for easy loading in ensemble step

### Priority 4: Implement Ensemble Methods
**Goal**: Replicate winning ensemble strategies

1. **Hill Climbing (Primary - Chris Deotte's approach)**
   - Load all OOF predictions
   - Start with best single model
   - Iteratively add models that improve ensemble CV
   - Use GPU acceleration for speed
   - Allow negative weights if beneficial
   - Clip final predictions to training data range

2. **Ridge Regression Ensemble (Secondary - AngelosMar's approach)**
   - Use OOF predictions as features
   - Train Ridge regression to find optimal weights
   - Compare performance to hill climbing
   - May combine both approaches

### Priority 5: Validation and Submission
**Goal**: Ensure robust evaluation and create final submission

1. **Cross-Validation Scheme**
   - Use 5-fold CV with seed 42
   - Compute RMSLE metric properly
   - Track CV stability across folds

2. **Prediction Clipping**
   - Clip final predictions to [train_min, train_max] range
   - Prevents unrealistic predictions

3. **Submission Preparation**
   - Create submission CSV with proper format
   - Verify predictions are in correct scale

## What NOT to Try

1. **Don't rely on LB feedback**: Winners explicitly ignored public LB due to unstable correlation
2. **Don't use simple averaging**: Hill climbing and Ridge regression are superior
3. **Don't skip target encoding**: This was a key technique for winners
4. **Don't use single model**: Ensemble diversity is critical for top performance
5. **Don't overfit to synthetic data**: Our current CV is too good - focus on methodology that works on real data

## Validation Notes

- **CV Scheme**: 5-fold CV with seed 42 (consistent with winners)
- **Metric**: RMSLE (Root Mean Squared Logarithmic Error)
- **Target Transformation**: Use log1p for training, expm1 for predictions
- **Clipping**: Clip predictions to training data min/max range
- **Trust CV**: Do not chase LB feedback - focus on CV improvement

## Expected Timeline

1. **Loop 1-2**: Implement 3-4 diverse models with basic feature engineering
2. **Loop 3-4**: Add advanced feature engineering (target encoding, binned features)
3. **Loop 5-6**: Generate OOF predictions and implement hill climbing
4. **Loop 7+**: Refine ensemble and optimize final model

## Success Criteria

- Generate at least 7 diverse models with CV scores in 0.055-0.070 range
- Implement hill climbing ensemble that beats best single model by >0.001
- Achieve final CV score < 0.058410 (target)
- Create robust pipeline that works on real competition data