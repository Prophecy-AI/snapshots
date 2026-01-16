"""
Experiment 061: Mixall-Style Ensemble with CORRECT Features

Key insight: Recent experiments (058-060) had a BUG - they used all 2048 DRFP features
instead of filtering to 122 high-variance ones like the baseline exp_030 does.

This experiment:
1. Uses CORRECT feature filtering (122 DRFP features with non-zero variance)
2. Uses MLP + XGBoost + RandomForest + LightGBM ensemble (like mixall kernel)
3. Tests if this combination has a different CV-LB relationship
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("Experiment 061: Mixall-Style Ensemble with CORRECT Features")
print("="*60)

# Data paths
DATA_PATH = '/home/data'

# Load data
full_data = pd.read_csv(f'{DATA_PATH}/catechol_full_data_yields.csv')
single_data = pd.read_csv(f'{DATA_PATH}/catechol_single_solvent_yields.csv')
spange = pd.read_csv(f'{DATA_PATH}/spange_descriptors_lookup.csv', index_col=0)
drfp = pd.read_csv(f'{DATA_PATH}/drfps_catechol_lookup.csv', index_col=0)
acs_pca = pd.read_csv(f'{DATA_PATH}/acs_pca_descriptors_lookup.csv', index_col=0)

print(f"Full data: {full_data.shape}")
print(f"Single solvent data: {single_data.shape}")

# Filter DRFP to high-variance columns (CRITICAL FIX!)
drfp_variance = drfp.var()
nonzero_variance_cols = drfp_variance[drfp_variance > 0].index.tolist()
drfp_filtered = drfp[nonzero_variance_cols]

print(f"Spange: {spange.shape}")
print(f"DRFP filtered: {drfp_filtered.shape} (from {drfp.shape[1]} original)")
print(f"ACS PCA: {acs_pca.shape}")

# Prepare single solvent data
single_data['Solvent'] = single_data['SOLVENT NAME']

# Merge features
spange_cols = spange.columns.tolist()
drfp_cols = drfp_filtered.columns.tolist()
acs_cols = acs_pca.columns.tolist()

# Reset index for merging
spange_reset = spange.reset_index().rename(columns={'index': 'Solvent'})
drfp_reset = drfp_filtered.reset_index().rename(columns={'index': 'Solvent'})
acs_reset = acs_pca.reset_index().rename(columns={'index': 'Solvent'})

single_merged = single_data.merge(spange_reset, on='Solvent', how='left')
single_merged = single_merged.merge(drfp_reset, on='Solvent', how='left')
single_merged = single_merged.merge(acs_reset, on='Solvent', how='left')

# Add Arrhenius features
single_merged['inv_temp'] = 1.0 / (single_merged['Temperature'] + 273.15)
single_merged['log_time'] = np.log1p(single_merged['Residence Time'])

feature_cols = spange_cols + drfp_cols + acs_cols + ['inv_temp', 'log_time']
X_single = single_merged[feature_cols].values
Y_single = single_merged[['SM', 'Product 2', 'Product 3']].values

print(f"\nSingle solvent features: {X_single.shape}")
print(f"Number of features: {len(feature_cols)} (13 Spange + {len(drfp_cols)} DRFP + {len(acs_cols)} ACS + 2 Arrhenius)")

# Prepare mixture data
full_data_mix = full_data[full_data['SolventB%'] > 0].copy()
full_data_mix['Solvent'] = full_data_mix['SOLVENT A NAME'] + '.' + full_data_mix['SOLVENT B NAME']

# Get features for solvent A and B
spange_a = spange_reset.copy()
spange_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in spange_cols]
spange_b = spange_reset.copy()
spange_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in spange_cols]

drfp_a = drfp_reset.copy()
drfp_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in drfp_cols]
drfp_b = drfp_reset.copy()
drfp_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in drfp_cols]

acs_a = acs_reset.copy()
acs_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in acs_cols]
acs_b = acs_reset.copy()
acs_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in acs_cols]

mix_merged = full_data_mix.merge(spange_a, on='SOLVENT A NAME', how='left')
mix_merged = mix_merged.merge(spange_b, on='SOLVENT B NAME', how='left')
mix_merged = mix_merged.merge(drfp_a, on='SOLVENT A NAME', how='left')
mix_merged = mix_merged.merge(drfp_b, on='SOLVENT B NAME', how='left')
mix_merged = mix_merged.merge(acs_a, on='SOLVENT A NAME', how='left')
mix_merged = mix_merged.merge(acs_b, on='SOLVENT B NAME', how='left')

# Average features weighted by ratio
ratio_b = mix_merged['SolventB%'].values / 100.0
ratio_a = 1.0 - ratio_b

for col in spange_cols:
    mix_merged[col] = ratio_a * mix_merged[f'{col}_A'].values + ratio_b * mix_merged[f'{col}_B'].values

for col in drfp_cols:
    mix_merged[col] = ratio_a * mix_merged[f'{col}_A'].values + ratio_b * mix_merged[f'{col}_B'].values

for col in acs_cols:
    mix_merged[col] = ratio_a * mix_merged[f'{col}_A'].values + ratio_b * mix_merged[f'{col}_B'].values

mix_merged['inv_temp'] = 1.0 / (mix_merged['Temperature'] + 273.15)
mix_merged['log_time'] = np.log1p(mix_merged['Residence Time'])

X_mix = mix_merged[feature_cols].values
Y_mix = mix_merged[['SM', 'Product 2', 'Product 3']].values

print(f"Mixture features: {X_mix.shape}")

# MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=3, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_mlp(X_train, Y_train, input_dim, epochs=200, lr=0.001, batch_size=32):
    model = MLPModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.FloatTensor(Y_train)
    
    model.train()
    for epoch in range(epochs):
        indices = torch.randperm(len(X_tensor))
        for i in range(0, len(X_tensor), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_tensor[batch_idx]
            Y_batch = Y_tensor[batch_idx]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
    
    return model

# Mixall-Style Ensemble: MLP + XGBoost + RandomForest + LightGBM
class MixallEnsemble:
    def __init__(self, input_dim, weights=[0.4, 0.2, 0.2, 0.2]):
        self.input_dim = input_dim
        self.weights = weights  # [MLP, XGB, RF, LGBM]
        self.mlp = None
        self.xgb_models = []
        self.rf_models = []
        self.lgbm_models = []
        self.scaler = StandardScaler()
    
    def fit(self, X_train, Y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train MLP
        self.mlp = train_mlp(X_scaled, Y_train, self.input_dim, epochs=200)
        
        # Train XGBoost (one per target)
        self.xgb_models = []
        for i in range(Y_train.shape[1]):
            xgb = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            xgb.fit(X_scaled, Y_train[:, i])
            self.xgb_models.append(xgb)
        
        # Train RandomForest (one per target)
        self.rf_models = []
        for i in range(Y_train.shape[1]):
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_scaled, Y_train[:, i])
            self.rf_models.append(rf)
        
        # Train LightGBM (one per target)
        self.lgbm_models = []
        for i in range(Y_train.shape[1]):
            lgbm = LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            lgbm.fit(X_scaled, Y_train[:, i])
            self.lgbm_models.append(lgbm)
    
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        
        # MLP predictions
        self.mlp.eval()
        with torch.no_grad():
            mlp_pred = self.mlp(torch.FloatTensor(X_scaled)).numpy()
        
        # XGBoost predictions
        xgb_pred = np.column_stack([m.predict(X_scaled) for m in self.xgb_models])
        
        # RandomForest predictions
        rf_pred = np.column_stack([m.predict(X_scaled) for m in self.rf_models])
        
        # LightGBM predictions
        lgbm_pred = np.column_stack([m.predict(X_scaled) for m in self.lgbm_models])
        
        # Weighted ensemble
        pred = (self.weights[0] * mlp_pred + 
                self.weights[1] * xgb_pred + 
                self.weights[2] * rf_pred + 
                self.weights[3] * lgbm_pred)
        
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

# Run CV for single solvents
print("\n" + "="*60)
print("Running Single Solvent CV with Mixall Ensemble...")
print("="*60)

single_errors = {}
all_preds = []
all_true = []

for fold_idx, (train_idx, test_idx, solvent) in enumerate(generate_leave_one_solvent_out_splits(single_merged)):
    X_train = X_single[train_idx]
    Y_train = Y_single[train_idx]
    X_test = X_single[test_idx]
    Y_test = Y_single[test_idx]
    
    model = MixallEnsemble(input_dim=len(feature_cols))
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

print(f"\nMixall Ensemble Single Solvent CV MSE: {single_mse:.6f} +/- {single_std:.6f}")

# Run CV for mixtures
print("\n" + "="*60)
print("Running Mixture CV with Mixall Ensemble...")
print("="*60)

mix_errors = {}
mix_preds_list = []
mix_true_list = []

for fold_idx, (train_idx, test_idx, mixture) in enumerate(generate_leave_one_ramp_out_splits(mix_merged)):
    X_train = X_mix[train_idx]
    Y_train = Y_mix[train_idx]
    X_test = X_mix[test_idx]
    Y_test = Y_mix[test_idx]
    
    model = MixallEnsemble(input_dim=len(feature_cols))
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

print(f"\nMixall Ensemble Mixture CV MSE: {mix_mse:.6f} +/- {mix_std:.6f}")

# Calculate overall CV score
print("\n" + "="*60)
print("Mixall Ensemble Overall Results")
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
print("EXPERIMENT 061 SUMMARY")
print("="*60)

print(f"\nMixall Ensemble (MLP + XGB + RF + LGBM):")
print(f"  Features: Spange + DRFP (filtered) + ACS PCA + Arrhenius ({len(feature_cols)} features)")
print(f"  Weights: MLP=0.4, XGB=0.2, RF=0.2, LGBM=0.2")
print(f"\n  Single Solvent CV: {single_mse:.6f}")
print(f"  Mixture CV: {mix_mse:.6f}")
print(f"  Overall CV: {overall_mse:.6f}")
print(f"  vs Best CV (exp_032): {improvement:+.1f}%")

print(f"\nKey insights:")
print(f"1. Uses CORRECT feature filtering (122 DRFP instead of 2048)")
print(f"2. Mixall-style ensemble (MLP + XGB + RF + LGBM)")
print(f"3. May have different CV-LB relationship")

print(f"\nRemaining submissions: 4")
print(f"Best model: exp_032 (CV 0.008194, LB 0.0873)")
