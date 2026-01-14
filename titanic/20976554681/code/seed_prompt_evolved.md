## Current Status
- Best CV score: 0.817 from exp_000 (baseline RandomForest)
- Best LB score: None (no submissions yet)
- CV-LB gap: Unknown - need LB feedback to calibrate
- Analysis shows validated features can improve CV to ~0.8305 (+0.0135)

## Response to Evaluator
- **Technical verdict was TRUSTWORTHY** - Agree. The baseline execution is solid with proper stratified CV and no leakage.
- **Evaluator's top priority: Hyperparameter tuning + XGBoost** - Partially agree. These are important, but we have validated features that show +0.0135 CV improvement that haven't been incorporated into a full experiment yet. We should implement these first, THEN tune.
- **Key concerns raised:**
  - *No hyperparameter tuning*: Valid concern. Will address after incorporating validated features.
  - *Missing advanced features*: Already validated TicketFreq, CabinSide, NameLength, FareBin5, and interactions in analysis notebooks. Need to implement in full experiment.
  - *Only one model type*: Agree. Will test XGBoost after getting LB feedback with improved features.
  - *No error analysis*: Completed in evolver_loop3_analysis.ipynb - identified 3rd class females, 2nd class males, and young males as problematic groups. Interaction features specifically address these.

## Data Understanding
- **Reference notebooks**: 
  - `exploration/evolver_loop1_analysis.ipynb` - EDA with TicketFreq (+24.7pp), CabinSide (+15.0pp), NameLength (r=0.332)
  - `exploration/evolver_loop2_analysis.ipynb` - Feature validation showing +0.0079 improvement
  - `exploration/evolver_loop3_analysis.ipynb` - Interaction features showing +0.0068 improvement (0.8237 → 0.8305)
- **Key patterns to exploit**:
  - Ticket frequency captures family/group survival patterns (51.7% vs 27.0% survival)
  - Cabin side (odd/even) shows location-based survival differences (76.1% vs 61.1%)
  - Name length correlates with social status (r=0.332)
  - Fare binning with 5 categories captures granular wealth effects
  - Interaction features (Pclass_Sex, AgeGroup_Sex, FareBin5_Sex) address class-gender misclassifications

## Recommended Approaches
Priority-ordered list:

1. **Run experiment with ALL validated features** (exp_001)
   - Include: TicketFreq, CabinSide, NameLength, FareBin5, Pclass_Sex, AgeGroup_Sex, FareBin5_Sex
   - Keep baseline features: Title, FamilySize, IsAlone, AgeGroup, FarePerPerson, Deck
   - Expected CV: ~0.8305 based on analysis
   - Reason: These features are validated and show consistent improvement. Highest ROI before tuning.

2. **Submit to LB for calibration** (candidate_001)
   - Reason: Critical to understand CV-LB gap before investing in hyperparameter tuning
   - If gap is small (<0.01), CV is reliable for optimization
   - If gap is large (>0.02), may indicate distribution shift requiring different approach

3. **Hyperparameter tuning on RandomForest** (exp_002)
   - Parameters: n_estimators (200-500), max_depth (5-15), min_samples_split (2-20), min_samples_leaf (1-10)
   - Use Optuna or GridSearchCV with 5-fold stratified CV
   - Reason: Evaluator correctly identified this as high-leverage. Could gain 2-5% improvement.

4. **Implement XGBoost model** (exp_003)
   - Use same feature set as exp_001
   - Parameters to tune: max_depth (3-9), min_child_weight (1-7), subsample (0.6-1.0), colsample_bytree (0.6-1.0), learning_rate (0.01-0.3)
   - Reason: XGBoost often outperforms RF on tabular data. Provides model diversity for ensembling.

5. **Feature importance analysis + error analysis** (analysis notebook)
   - Identify which validated features contribute most
   - Analyze misclassifications to guide further feature engineering
   - Reason: Understand what's working and where to focus next efforts

6. **Simple ensemble (if time permits)**
   - Average predictions from tuned RF and XGBoost
   - Reason: Ensembling typically provides 1-2% improvement with minimal effort

## What NOT to Try
- **Neural networks**: Titanic is tabular data with <1000 samples. Tree models are superior here.
- **Complex feature engineering beyond validated ones**: The current validated set is strong. Focus on tuning before adding more complexity.
- **Advanced ensembling (stacking)**: Too early. Need multiple strong base models first.
- **More feature engineering without LB feedback**: Risk of overfitting to CV without knowing LB behavior.

## Validation Notes
- **CV scheme**: Stratified 5-fold CV (same as baseline)
- **Metric**: Accuracy (minimize - the competition metric)
- **Submission threshold**: Submit exp_001 to get LB feedback before heavy tuning
- **Success criteria**: CV improvement of +0.01+ with validated features, then maintain/improve with tuning