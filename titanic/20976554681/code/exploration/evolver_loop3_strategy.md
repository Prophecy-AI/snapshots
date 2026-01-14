# Evolver Loop 3 Strategic Analysis

## Current State Assessment

**Best CV Score**: 83.84% from exp_001 (XGBoost, leakage fixed)
**Best LB Score**: None yet (0/10 submissions used)
**CV-LB Gap**: Unknown - CRITICAL INFORMATION GAP
**Remaining Submissions**: 9/10

## Evaluator Feedback Response

**Technical Verdict**: TRUSTWORTHY ✓
- Leakage successfully fixed with sklearn pipelines
- CV score is reliable and trustworthy
- Code quality is excellent

**Evaluator's Top Priority**: Systematic hyperparameter tuning with RandomizedSearchCV

**My Response**: I agree hyperparameter tuning is high-ROI, BUT we have a strategic decision to make first:

### Critical Decision Point: Submit NOW vs Tune First

**Arguments for Submitting Now:**
1. **CV-LB correlation is UNKNOWN** - This is critical missing information
2. **Pipeline is TRUSTWORTHY** - No technical concerns blocking submission
3. **Score is COMPETITIVE** - 83.84% is solid for Titanic
4. **9 submissions remaining** - Can afford calibration submission
5. **LB feedback will inform tuning** - Better to know gap before extensive tuning

**Arguments for Tuning First:**
1. Could gain +0.5-1% before submitting
2. Evaluator identified this as top priority
3. Might be more efficient

**My Decision**: SUBMIT NOW for these strategic reasons:

1. **Information Value**: CV-LB gap is crucial for all future decisions
2. **Low Risk**: Worst case, we get valuable feedback
3. **Parallel Processing**: Can tune while waiting for LB score
4. **Resource Abundance**: 9 submissions is plenty
5. **Psychological**: Getting on LB provides motivation and validation

## Data Understanding

**Reference Notebooks:**
- `exploration/evolver_loop1_analysis.ipynb` - Leakage impact analysis
- `exploration/evolver_loop2_analysis.ipynb` - Post-fix analysis
- `research/kernels/` - Winning solution patterns

**Key Findings from Data:**
- Title_Other at 5.7% importance → Splitting could yield +0.3-0.7%
- Optimization roadmap: 83.84% → 87.84% potential
  - Hyperparameter tuning: +1.0%
  - Simple ensemble: +0.7%
  - Title refinement: +0.5%
  - Age optimization: +0.3%
  - Advanced stacking: +1.5%

**Feature Importance Hierarchy:**
1. Title_Mr: 19.4%
2. Sex_female: 14.9%
3. Sex_male: 10.0%
4. Fare: 8.2%
5. Pclass: 7.1%
6. Age: 6.5%
7. Title_Other: 5.7% ← REFINEMENT OPPORTUNITY

## Recommended Next Steps (Priority Order)

### IMMEDIATE (Before Next Experiment):
1. **SUBMIT exp_001 to LB** - Establish CV-LB correlation
2. **Analyze results** - Determine if CV is optimistic/pessimistic

### Priority 1: Hyperparameter Tuning (Evaluator's Top Priority)
**Why First**: Highest ROI, pipeline is ready, can run while analyzing LB feedback

**Specific Approach:**
- Use RandomizedSearchCV with 30-50 iterations
- Search space:
  - n_estimators: [300, 500, 800, 1000]
  - max_depth: [3, 4, 5, 6, 7]
  - learning_rate: [0.01, 0.05, 0.1]
  - subsample: [0.7, 0.8, 0.9, 1.0]
  - colsample_bytree: [0.7, 0.8, 0.9, 1.0]
- Keep other parameters fixed initially
- Expected gain: +0.5-1.0%

### Priority 2: Title Refinement
**Why Second**: Clear signal (5.7% importance), easy to implement, +0.3-0.7% expected

**Specific Mapping:**
Current: Other (29 rare titles lumped)
Proposed:
- Keep: Mr, Mrs, Miss, Master
- New: Dr (professional status)
- New: Military (Col, Major, Capt - authority structure)
- New: Noble (Countess, Lady, Sir, Don, Dona, Jonkheer - social class)
- New: Clergy (Rev - different social role)
- Group: Remaining truly rare titles

**Implementation:** Simple mapping function, low risk

### Priority 3: Simple Ensemble (XGBoost + Logistic Regression)
**Why Third**: Proven pattern, +0.7% expected, builds on tuned models

**Approach:**
- Train XGBoost (tuned from Priority 1)
- Train Logistic Regression (same preprocessing)
- Weighted blend: 75% XGBoost + 25% LR
- Use pipeline to ensure consistency

### Priority 4: Age Optimization
**Why Fourth**: Smaller expected gain (+0.3%), more complex to optimize

**Approaches to Try:**
1. Data-driven binning (find natural breakpoints)
2. Different threshold sets via CV
3. Use as numeric feature (XGBoost handles non-linearity)
4. Polynomial features (Age², Age³)

### Priority 5: Advanced Stacking
**Why Fifth**: Highest complexity, +1.5% potential but requires well-tuned base models

**Save for Later**: After we have:
- Tuned XGBoost ✓ (Priority 1)
- Tuned LR ✓ (Priority 3)
- CV-LB correlation established ✓ (Immediate)

## What NOT to Try (Yet)

1. **Complex feature interactions** - Wait for LB feedback
2. **Neural networks** - Overkill for this problem
3. **Target encoding** - Leakage risk with small dataset
4. **Cabin deck letters** - Statistically insignificant (p=0.172)
5. **Ticket parsing** - Low ROI, high complexity

## Validation Strategy

**CV Scheme**: 5-fold stratified CV (proven effective)
**Holdout**: Create 20% stratified holdout after LB submission
**LB Tracking**: Monitor CV-LB gap after each submission

## Success Metrics

**Short-term (Next 2 experiments):**
1. LB submission with feedback
2. Hyperparameter tuning: +0.5% CV improvement
3. Title refinement: +0.3% CV improvement

**Medium-term (Next 4 experiments):**
1. Ensemble implementation: +0.7% CV improvement
2. CV-LB gap < 2%
3. Score > 85% CV

## Risk Assessment

**Low Risk:**
- Hyperparameter tuning (proven approach)
- Title refinement (clear signal)
- LB submission (pipeline trustworthy)

**Medium Risk:**
- Ensemble (requires well-tuned components)

**Mitigation:**
- Establish CV-LB correlation first
- Incremental improvements
- Keep 5+ submissions in reserve for final ensemble

## Decision Rationale

**Why Submit Now:**
1. **Information Asymmetry**: CV-LB gap is unknown, affecting all decisions
2. **Trust Established**: Pipeline is verified trustworthy
3. **Competitive Position**: 83.84% is submission-worthy
4. **Resource Management**: 9 submissions allows calibration
5. **Strategic Value**: LB feedback will optimize subsequent tuning

**Why Hyperparameter Tuning Next:**
1. **Evaluator Priority**: Directly addresses top concern
2. **High ROI**: +1% potential gain
3. **Clean Pipeline**: Ready for systematic search
4. **Parallelizable**: Can run while analyzing LB results
5. **Foundation for Ensembling**: Well-tuned models needed first

**Why Title Refinement Third:**
1. **Clear Signal**: 5.7% importance indicates value
2. **Easy Implementation**: Simple mapping function
3. **Expected Gain**: +0.3-0.7%
4. **Low Risk**: Doesn't break existing pipeline
5. **Complements Tuning**: Independent improvement

This strategy balances immediate action (submission) with systematic improvement (tuning → refinement → ensemble) while maintaining strategic flexibility based on LB feedback.