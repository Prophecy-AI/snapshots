"""
Experiment 063: CatBoost + XGBoost Ensemble (matthewmaree style)

Based on the matthewmaree kernel approach:
1. Uses ALL feature sources: spange, acs_pca, drfps, fragprints
2. Correlation filtering (threshold 0.90) to remove redundant features
3. CatBoost with MultiRMSE loss
4. XGBoost with reg:squarederror
5. Different weights for single vs full: single (7:6), full (1:2)
6. Clip predictions to [0, inf), normalize to sum <= 1
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("Experiment 063: CatBoost + XGBoost Ensemble (matthewmaree style)")
print("="*60)

# Data paths
DATA_PATH = '/home/data'

# Load data
full_data = pd.read_csv(f'{DATA_PATH}/catechol_full_data_yields.csv')
single_data = pd.read_csv(f'{DATA_PATH}/catechol_single_solvent_yields.csv')
spange = pd.read_csv(f'{DATA_PATH}/spange_descriptors_lookup.csv')
drfp = pd.read_csv(f'{DATA_PATH}/drfps_catechol_lookup.csv')
acs_pca = pd.read_csv(f'{DATA_PATH}/acs_pca_descriptors_lookup.csv')
fragprints = pd.read_csv(f'{DATA_PATH}/fragprints_lookup.csv')

print(f"Full data: {full_data.shape}")
print(f"Single solvent data: {single_data.shape}")
print(f"Spange: {spange.shape}")
print(f"DRFP: {drfp.shape}")
print(f"ACS PCA: {acs_pca.shape}")
print(f"Fragprints: {fragprints.shape}")

# Rename solvent column for consistency
spange = spange.rename(columns={'SOLVENT NAME': 'Solvent'})
drfp = drfp.rename(columns={'SOLVENT NAME': 'Solvent'})
acs_pca = acs_pca.rename(columns={'SOLVENT NAME': 'Solvent'})
fragprints = fragprints.rename(columns={'SOLVENT NAME': 'Solvent'})

# Get column names
spange_cols = [c for c in spange.columns if c != 'Solvent']
drfp_cols = [c for c in drfp.columns if c != 'Solvent']
acs_cols = [c for c in acs_pca.columns if c != 'Solvent']
frag_cols = [c for c in fragprints.columns if c != 'Solvent']

# Filter DRFP to high-variance columns
drfp_variance = drfp[drfp_cols].var()
drfp_filtered_cols = drfp_variance[drfp_variance > 0].index.tolist()

# Filter fragprints to high-variance columns
frag_variance = fragprints[frag_cols].var()
frag_filtered_cols = frag_variance[frag_variance > 0].index.tolist()

print(f"\nSpange features: {len(spange_cols)}")
print(f"DRFP filtered: {len(drfp_filtered_cols)} (from {len(drfp_cols)})")
print(f"ACS PCA features: {len(acs_cols)}")
print(f"Fragprints filtered: {len(frag_filtered_cols)} (from {len(frag_cols)})")

# Prepare single solvent data
single_data['Solvent'] = single_data['SOLVENT NAME']

single_merged = single_data.merge(spange, on='Solvent', how='left')
single_merged = single_merged.merge(drfp[['Solvent'] + drfp_filtered_cols], on='Solvent', how='left')
single_merged = single_merged.merge(acs_pca, on='Solvent', how='left')
single_merged = single_merged.merge(fragprints[['Solvent'] + frag_filtered_cols], on='Solvent', how='left')

# Add Arrhenius features
single_merged['inv_temp'] = 1.0 / (single_merged['Temperature'] + 273.15)
single_merged['log_time'] = np.log1p(single_merged['Residence Time'])

feature_cols = spange_cols + drfp_filtered_cols + acs_cols + frag_filtered_cols + ['inv_temp', 'log_time']

# Remove highly correlated features (threshold 0.90)
def remove_correlated_features(df, feature_cols, threshold=0.90):
    """Remove features with correlation > threshold"""
    X = df[feature_cols].values
    corr_matrix = np.corrcoef(X.T)
    
    # Find pairs with high correlation
    to_remove = set()
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            if abs(corr_matrix[i, j]) > threshold:
                # Remove the feature with higher mean correlation
                if np.mean(np.abs(corr_matrix[i, :])) > np.mean(np.abs(corr_matrix[j, :])):
                    to_remove.add(feature_cols[i])
                else:
                    to_remove.add(feature_cols[j])
    
    return [c for c in feature_cols if c not in to_remove]

# Apply correlation filtering
feature_cols_filtered = remove_correlated_features(single_merged, feature_cols, threshold=0.90)
print(f"\nFeatures after correlation filtering: {len(feature_cols_filtered)} (from {len(feature_cols)})")

X_single = single_merged[feature_cols_filtered].values
Y_single = single_merged[['SM', 'Product 2', 'Product 3']].values

print(f"Single solvent features: {X_single.shape}")

# Prepare mixture data
full_data_mix = full_data[full_data['SolventB%'] > 0].copy()
full_data_mix['Solvent'] = full_data_mix['SOLVENT A NAME'] + '.' + full_data_mix['SOLVENT B NAME']

# Get features for solvent A and B
def prepare_mixture_features(data, spange, drfp, acs_pca, fragprints, 
                             spange_cols, drfp_cols, acs_cols, frag_cols):
    """Prepare features for mixture data"""
    # Solvent A features
    spange_a = spange.copy()
    spange_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in spange_cols]
    drfp_a = drfp.copy()
    drfp_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in drfp_cols]
    acs_a = acs_pca.copy()
    acs_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in acs_cols]
    frag_a = fragprints.copy()
    frag_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in frag_cols]
    
    # Solvent B features
    spange_b = spange.copy()
    spange_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in spange_cols]
    drfp_b = drfp.copy()
    drfp_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in drfp_cols]
    acs_b = acs_pca.copy()
    acs_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in acs_cols]
    frag_b = fragprints.copy()
    frag_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in frag_cols]
    
    # Merge
    data = data.merge(spange_a, on='SOLVENT A NAME', how='left')
    data = data.merge(spange_b, on='SOLVENT B NAME', how='left')
    data = data.merge(drfp_a, on='SOLVENT A NAME', how='left')
    data = data.merge(drfp_b, on='SOLVENT B NAME', how='left')
    data = data.merge(acs_a, on='SOLVENT A NAME', how='left')
    data = data.merge(acs_b, on='SOLVENT B NAME', how='left')
    data = data.merge(frag_a, on='SOLVENT A NAME', how='left')
    data = data.merge(frag_b, on='SOLVENT B NAME', how='left')
    
    # Average features weighted by ratio
    ratio_b = data['SolventB%'].values / 100.0
    ratio_a = 1.0 - ratio_b
    
    for col in spange_cols:
        data[col] = ratio_a * data[f'{col}_A'].values + ratio_b * data[f'{col}_B'].values
    for col in drfp_cols:
        data[col] = ratio_a * data[f'{col}_A'].values + ratio_b * data[f'{col}_B'].values
    for col in acs_cols:
        data[col] = ratio_a * data[f'{col}_A'].values + ratio_b * data[f'{col}_B'].values
    for col in frag_cols:
        data[col] = ratio_a * data[f'{col}_A'].values + ratio_b * data[f'{col}_B'].values
    
    data['inv_temp'] = 1.0 / (data['Temperature'] + 273.15)
    data['log_time'] = np.log1p(data['Residence Time'])
    
    return data

mix_merged = prepare_mixture_features(
    full_data_mix, spange, 
    drfp[['Solvent'] + drfp_filtered_cols],
    acs_pca,
    fragprints[['Solvent'] + frag_filtered_cols],
    spange_cols, drfp_filtered_cols, acs_cols, frag_filtered_cols
)

X_mix = mix_merged[feature_cols_filtered].values
Y_mix = mix_merged[['SM', 'Product 2', 'Product 3']].values

print(f"Mixture features: {X_mix.shape}")

# CatBoost + XGBoost Ensemble
class CatBoostXGBoostEnsemble:
    def __init__(self, input_dim, catboost_weight=0.5, xgboost_weight=0.5):
        self.input_dim = input_dim
        self.catboost_weight = catboost_weight
        self.xgboost_weight = xgboost_weight
        self.catboost_models = []
        self.xgboost_models = []
        self.scaler = StandardScaler()
    
    def fit(self, X_train, Y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train CatBoost (one per target)
        self.catboost_models = []
        for i in range(Y_train.shape[1]):
            catboost = CatBoostRegressor(
                iterations=1050,
                learning_rate=0.07,
                depth=3,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
            catboost.fit(X_scaled, Y_train[:, i])
            self.catboost_models.append(catboost)
        
        # Train XGBoost (one per target)
        self.xgboost_models = []
        for i in range(Y_train.shape[1]):
            xgboost = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            xgboost.fit(X_scaled, Y_train[:, i])
            self.xgboost_models.append(xgboost)
    
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        
        # CatBoost predictions
        catboost_pred = np.column_stack([m.predict(X_scaled) for m in self.catboost_models])
        
        # XGBoost predictions
        xgboost_pred = np.column_stack([m.predict(X_scaled) for m in self.xgboost_models])
        
        # Weighted ensemble
        pred = self.catboost_weight * catboost_pred + self.xgboost_weight * xgboost_pred
        
        # Clip to [0, inf) and normalize if sum > 1
        pred = np.clip(pred, 0, None)
        row_sums = pred.sum(axis=1, keepdims=True)
        pred = np.where(row_sums > 1, pred / row_sums, pred)
        
        return np.clip(pred, 0, 1)

# CV functions
def generate_leave_one_solvent_out_splits(data):
    solvents = data['Solvent'].unique()
    for solvent in solvents:
        test_mask = data['Solvent'] == solvent
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx, solvent

def generate_leave_one_ramp_out_splits(data):
    ramps = data['Solvent'].unique()
    for ramp in ramps:
        test_mask = data['Solvent'] == ramp
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx, ramp

# Run CV for single solvents with weights (7:6) = (0.538, 0.462)
print("\n" + "="*60)
print("Running Single Solvent CV with CatBoost+XGBoost (7:6 weights)...")
print("="*60)

single_errors = {}
all_preds = []
all_true = []

for fold_idx, (train_idx, test_idx, solvent) in enumerate(generate_leave_one_solvent_out_splits(single_merged)):
    X_train = X_single[train_idx]
    Y_train = Y_single[train_idx]
    X_test = X_single[test_idx]
    Y_test = Y_single[test_idx]
    
    # Single: CatBoost 7:6 XGBoost = 0.538:0.462
    model = CatBoostXGBoostEnsemble(input_dim=len(feature_cols_filtered), 
                                     catboost_weight=7/13, xgboost_weight=6/13)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    
    mse = np.mean((pred - Y_test) ** 2)
    single_errors[solvent] = mse
    
    all_preds.append(pred)
    all_true.append(Y_test)
    
    print(f"Fold {fold_idx+1:2d}: {solvent:50s} MSE = {mse:.6f}")

all_preds = np.vstack(all_preds)
all_true = np.vstack(all_true)
single_mse = np.mean((all_preds - all_true) ** 2)
single_std = np.std(list(single_errors.values()))

print(f"\nCatBoost+XGBoost Single Solvent CV MSE: {single_mse:.6f} +/- {single_std:.6f}")

# Run CV for mixtures with weights (1:2) = (0.333, 0.667)
print("\n" + "="*60)
print("Running Mixture CV with CatBoost+XGBoost (1:2 weights)...")
print("="*60)

mix_errors = {}
mix_preds_list = []
mix_true_list = []

for fold_idx, (train_idx, test_idx, mixture) in enumerate(generate_leave_one_ramp_out_splits(mix_merged)):
    X_train = X_mix[train_idx]
    Y_train = Y_mix[train_idx]
    X_test = X_mix[test_idx]
    Y_test = Y_mix[test_idx]
    
    # Full: CatBoost 1:2 XGBoost = 0.333:0.667
    model = CatBoostXGBoostEnsemble(input_dim=len(feature_cols_filtered),
                                     catboost_weight=1/3, xgboost_weight=2/3)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    
    mse = np.mean((pred - Y_test) ** 2)
    mix_errors[mixture] = mse
    
    mix_preds_list.append(pred)
    mix_true_list.append(Y_test)
    
    print(f"Fold {fold_idx+1:2d}: {mixture:50s} MSE = {mse:.6f}")

mix_preds = np.vstack(mix_preds_list)
mix_true = np.vstack(mix_true_list)
mix_mse = np.mean((mix_preds - mix_true) ** 2)
mix_std = np.std(list(mix_errors.values()))

print(f"\nCatBoost+XGBoost Mixture CV MSE: {mix_mse:.6f} +/- {mix_std:.6f}")

# Calculate overall CV score
print("\n" + "="*60)
print("CatBoost+XGBoost Ensemble Overall Results")
print("="*60)

n_single = len(all_true)
n_mix = len(mix_true)
n_total = n_single + n_mix

overall_mse = (n_single * single_mse + n_mix * mix_mse) / n_total

print(f"\nSingle Solvent CV MSE: {single_mse:.6f} +/- {single_std:.6f} (n={n_single})")
print(f"Mixture CV MSE: {mix_mse:.6f} +/- {mix_std:.6f} (n={n_mix})")
print(f"Overall CV MSE: {overall_mse:.6f}")

print(f"\nBaseline (exp_030): CV = 0.008298")
print(f"Best CV (exp_032): CV = 0.008194")
improvement = (overall_mse - 0.008194) / 0.008194 * 100
print(f"Improvement vs best CV: {improvement:+.1f}%")

if overall_mse < 0.008194:
    print("\n✓ BETTER than best CV!")
elif overall_mse < 0.008298:
    print("\n✓ BETTER than baseline!")
else:
    print("\n✗ WORSE than baseline.")

# Final Summary
print("\n" + "="*60)
print("EXPERIMENT 063 SUMMARY")
print("="*60)

print(f"\nCatBoost + XGBoost Ensemble (matthewmaree style):")
print(f"  Features: Spange + DRFP + ACS PCA + Fragprints + Arrhenius")
print(f"  After correlation filtering: {len(feature_cols_filtered)} features")
print(f"  Single weights: CatBoost 7:6 XGBoost")
print(f"  Mixture weights: CatBoost 1:2 XGBoost")
print(f"  Clipping + normalization applied")
print(f"\n  Single Solvent CV: {single_mse:.6f}")
print(f"  Mixture CV: {mix_mse:.6f}")
print(f"  Overall CV: {overall_mse:.6f}")
print(f"  vs Best CV (exp_032): {improvement:+.1f}%")

print(f"\nRemaining submissions: 4")
print(f"Best model: exp_032 (CV 0.008194, LB 0.0873)")
