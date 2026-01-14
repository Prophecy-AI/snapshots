## Current Status
- Best CV: 0.0718 from exp_006 (ResNet50 + fine-tuning + TTA)
- Gap to gold: 0.0330 (46% improvement needed)
- Experiments above gold: 0
- **Critical finding**: Only 2.4% improvement vs expected 30-40% indicates severe optimization issues

## Response to Evaluator
- Technical verdict was **TRUSTWORTHY but concerning**. I fully agree with the evaluator's assessment that training dynamics show critical optimization issues preventing the model from reaching its potential.
- **Evaluator's top priority: Fix optimization hyperparameters and training schedule**. This is absolutely correct and my analysis confirms it.
- **Key concerns raised**:
  - Early stopping too aggressive (patience=3) → Confirmed: 4/5 folds stopped early, avg only 2.0 Phase 2 epochs
  - Learning rates too high → Confirmed: Validation degrades after initial improvement, best scores in epochs 1-3
  - Training instability → Confirmed: 3/5 folds degrade from best to final (avg +0.0053 degradation)
  - Batch size too small → Confirmed: 32 is conservative for A100, need 64-128
  - No LR warmup → Confirmed: Causes optimization instability

## Data Understanding
- **Reference notebooks**: See `exploration/evolver_loop2_analysis.ipynb` for detailed training dynamics analysis
- **Key patterns discovered**:
  - Fold 4 achieved 0.0621 (15.6% improvement!) but degraded to 0.0735 due to poor optimization
  - Training follows pattern: initial improvement (Phase 2 epochs 1-3) then divergence
  - Early stopping triggered prematurely in 80% of folds
  - ResNet50 CAN beat baseline significantly when properly trained

## Recommended Approaches (Priority Order)

### 1. FIX LEARNING RATES (CRITICAL - HIGHEST PRIORITY)
**What**: Reduce learning rates by 5x and implement proper scheduling
**Why**: Current LRs (backbone=0.0001, head=0.001) cause divergence after initial fit. Evidence: Fold 4 best=0.0621 but final=0.0735 (+0.0114 degradation). Best scores occur in epochs 1-3, then degrade.
**How**:
- Backbone LR: 0.00002 (5x lower than current)
- Head LR: 0.0002 (5x lower than current)
- Use 10:1 ratio (backbone:head) for differential learning rates
- Implement LR warmup: linear warmup from 0 to target LR over 2 epochs
- Use cosine annealing schedule instead of ReduceLROnPlateau
- Remove early stopping, train for fixed schedule

### 2. EXTEND TRAINING DURATION (CRITICAL)
**What**: Train for 15-18 total epochs (3 head-only + 12-15 fine-tuning)
**Why**: Current 8 epoch max with early stopping (avg 4.8 epochs) is insufficient. Fine-tuning requires longer adaptation. Evidence: Best scores in early epochs suggest model needs more time to stabilize at lower LRs.
**How**:
- Phase 1: 3 epochs (frozen backbone) - keep this
- Phase 2: 12-15 epochs (fine-tuning) - remove early stopping
- Monitor but don't stop early; let cosine schedule handle LR decay
- Save best model based on validation score

### 3. INCREASE BATCH SIZE (HIGH PRIORITY)
**What**: Increase from 32 to 64-128
**Why**: Larger batches provide stable gradients, better GPU utilization. A100 has 80GB memory, can easily handle 128+ batch size. Current small batches cause high variance across folds (std=0.0026).
**How**:
- Start with batch_size=64 (2x increase)
- If stable, try 96 or 128
- Adjust learning rates linearly (LR ∝ batch size)

### 4. ENHANCE REGULARIZATION (MEDIUM PRIORITY)
**What**: Add Cutout/RandomErasing and consider Mixup/CutMix
**Why**: Model shows overfitting (validation loss increases after initial fit). Current label smoothing insufficient. Need stronger regularization to maintain improvements.
**How**:
- Add Cutout: 8x8 patches, probability=0.3
- Add RandomErasing: probability=0.25, scale=(0.02, 0.15)
- Try Mixup: alpha=0.2 (mix images and labels)
- Keep label smoothing at 0.1

### 5. OPTIMIZER & SCHEDULE REFINEMENT (MEDIUM PRIORITY)
**What**: Switch to AdamW with proper weight decay and cosine schedule
**Why**: Current optimizer config contributes to instability. Cosine annealing provides smooth LR decay vs ReduceLROnPlateau's abrupt changes.
**How**:
- Optimizer: AdamW with weight_decay=0.05
- Schedule: Cosine annealing with warmup
- Warmup: Linear increase over 2 epochs
- T_max: Total fine-tuning epochs (12-15)
- Eta_min: 1e-6

### 6. ARCHITECTURE ASSESSMENT (LOW PRIORITY - DEFER)
**What**: Keep ResNet50, don't switch architectures yet
**Why**: Fold 4 proved ResNet50 can achieve 0.0621 (15.6% improvement) with current setup. Problem is optimization, not architecture. Fix training first before exploring EfficientNet or other architectures.
**When to reconsider**: If after fixing optimization we still don't see 15-20% improvement

## What NOT to Try
- **New architectures**: Don't switch to EfficientNet yet - ResNet50 is capable when properly trained
- **Complex ensembles**: Need strong single model first (currently underperforming)
- **AutoML/hyperparameter sweeps**: Manual tuning based on training dynamics is more efficient
- **Advanced TTA**: Current TTA is fine; focus on base model quality first
- **Progressive unfreezing variations**: Current 2-phase approach is correct, just needs better hyperparameters

## Validation Notes
- **CV Scheme**: Continue 5-fold stratified CV (proven stable)
- **Confidence**: High - fixes address root cause of optimization instability
- **Risk**: Low - changes are incremental and based on clear evidence
- **Target**: Aim for 0.055-0.060 in next experiment (16-25% improvement from current)
- **Success metric**: Should see best validation scores maintained/improved throughout training, not just early epochs

## Expected Outcomes
With proper optimization fixes:
- ResNet50 should achieve 0.055-0.060 (15-20% improvement from baseline)
- Training curves should show stable convergence without degradation
- All folds should complete full training schedule
- TTA should provide meaningful boost (3-5%) on stronger base model