"""
Experiment 059: Auxiliary Multi-Task Learning (NEW - NOT YET TRIED)

Hypothesis: Research shows that jointly training with auxiliary self-supervised tasks 
can improve generalization by up to 7.7%. This is fundamentally different from our 
previous approaches.

Key changes from baseline (exp_030):
1. Add auxiliary task: Reconstruct input features (autoencoder)
2. Add auxiliary task: Predict solvent properties from embeddings
3. Joint training forces model to learn more generalizable representations
4. Keep GP + MLP + LGBM ensemble structure but with multi-task MLP
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("Experiment 059: Auxiliary Multi-Task Learning")
print("="*60)

# Data paths
DATA_PATH = '/home/data'

# Load data
full_data = pd.read_csv(f'{DATA_PATH}/catechol_full_data_yields.csv')
single_data = pd.read_csv(f'{DATA_PATH}/catechol_single_solvent_yields.csv')
spange = pd.read_csv(f'{DATA_PATH}/spange_descriptors_lookup.csv')
drfp = pd.read_csv(f'{DATA_PATH}/drfps_catechol_lookup.csv')

print(f"Full data: {full_data.shape}")
print(f"Single solvent data: {single_data.shape}")
print(f"Spange descriptors: {spange.shape}")
print(f"DRFP features: {drfp.shape}")

# Rename columns for consistency
spange = spange.rename(columns={'SOLVENT NAME': 'Solvent'})
drfp = drfp.rename(columns={'SOLVENT NAME': 'Solvent'})

spange_cols = [c for c in spange.columns if c != 'Solvent']
drfp_cols = [c for c in drfp.columns if c != 'Solvent']

print(f"Spange features: {len(spange_cols)}")
print(f"DRFP features: {len(drfp_cols)}")

# Prepare single solvent data
single_data['Solvent'] = single_data['SOLVENT NAME']
single_merged = single_data.merge(spange, on='Solvent', how='left')
single_merged = single_merged.merge(drfp, on='Solvent', how='left')
single_merged['inv_temp'] = 1.0 / (single_merged['Temperature'] + 273.15)
single_merged['log_time'] = np.log1p(single_merged['Residence Time'])

feature_cols = spange_cols + drfp_cols + ['inv_temp', 'log_time']
X_single = single_merged[feature_cols].values
Y_single = single_merged[['SM', 'Product 2', 'Product 3']].values

# Also extract Spange properties as auxiliary targets
spange_target_cols = ['dielectric constant', 'ET(30)', 'alpha', 'beta', 'pi*']
Y_spange_single = single_merged[spange_target_cols].values

print(f"\nSingle solvent features: {X_single.shape}")
print(f"Single solvent targets: {Y_single.shape}")
print(f"Spange auxiliary targets: {Y_spange_single.shape}")
print(f"Number of features: {len(feature_cols)}")

# Prepare mixture data
full_data_mix = full_data[full_data['SolventB%'] > 0].copy()
full_data_mix['Solvent'] = full_data_mix['SOLVENT A NAME'] + '.' + full_data_mix['SOLVENT B NAME']

# Get features for solvent A and B
spange_a = spange.copy()
spange_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in spange_cols]
spange_b = spange.copy()
spange_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in spange_cols]

drfp_a = drfp.copy()
drfp_a.columns = ['SOLVENT A NAME'] + [f'{c}_A' for c in drfp_cols]
drfp_b = drfp.copy()
drfp_b.columns = ['SOLVENT B NAME'] + [f'{c}_B' for c in drfp_cols]

mix_merged = full_data_mix.merge(spange_a, on='SOLVENT A NAME', how='left')
mix_merged = mix_merged.merge(spange_b, on='SOLVENT B NAME', how='left')
mix_merged = mix_merged.merge(drfp_a, on='SOLVENT A NAME', how='left')
mix_merged = mix_merged.merge(drfp_b, on='SOLVENT B NAME', how='left')

# Average features weighted by ratio
ratio_b = mix_merged['SolventB%'].values / 100.0
ratio_a = 1.0 - ratio_b

for col in spange_cols:
    mix_merged[col] = ratio_a * mix_merged[f'{col}_A'].values + ratio_b * mix_merged[f'{col}_B'].values

for col in drfp_cols:
    mix_merged[col] = ratio_a * mix_merged[f'{col}_A'].values + ratio_b * mix_merged[f'{col}_B'].values

mix_merged['inv_temp'] = 1.0 / (mix_merged['Temperature'] + 273.15)
mix_merged['log_time'] = np.log1p(mix_merged['Residence Time'])

X_mix = mix_merged[feature_cols].values
Y_mix = mix_merged[['SM', 'Product 2', 'Product 3']].values
Y_spange_mix = mix_merged[spange_target_cols].values

print(f"\nMixture features: {X_mix.shape}")
print(f"Mixture targets: {Y_mix.shape}")
print(f"Spange auxiliary targets: {Y_spange_mix.shape}")

# Multi-Task MLP Model with Auxiliary Tasks
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=3, aux_dim=5, dropout=0.3):
        super().__init__()
        
        # Shared encoder
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
        self.encoder = nn.Sequential(*layers)
        
        # Main task head: Predict yields
        self.yield_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Sigmoid()
        )
        
        # Auxiliary task 1: Reconstruct input features (autoencoder)
        self.reconstruction_head = nn.Linear(hidden_dims[-1], input_dim)
        
        # Auxiliary task 2: Predict solvent properties
        self.solvent_property_head = nn.Linear(hidden_dims[-1], aux_dim)
    
    def forward(self, x, return_aux=False):
        z = self.encoder(x)
        yields = self.yield_head(z)
        
        if return_aux:
            reconstruction = self.reconstruction_head(z)
            solvent_props = self.solvent_property_head(z)
            return yields, reconstruction, solvent_props
        return yields

def train_multitask_mlp(X_train, Y_train, Y_spange_train, input_dim, 
                        epochs=300, lr=0.001, batch_size=32,
                        aux_weight_recon=0.1, aux_weight_props=0.1):
    """Train multi-task MLP with auxiliary objectives"""
    model = MultiTaskMLP(input_dim, hidden_dims=[128, 64], aux_dim=Y_spange_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.FloatTensor(Y_train)
    Y_spange_tensor = torch.FloatTensor(Y_spange_train)
    
    # Normalize auxiliary targets
    spange_mean = Y_spange_tensor.mean(dim=0)
    spange_std = Y_spange_tensor.std(dim=0) + 1e-8
    Y_spange_norm = (Y_spange_tensor - spange_mean) / spange_std
    
    model.train()
    for epoch in range(epochs):
        indices = torch.randperm(len(X_tensor))
        total_loss = 0
        
        for i in range(0, len(X_tensor), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_tensor[batch_idx]
            Y_batch = Y_tensor[batch_idx]
            Y_spange_batch = Y_spange_norm[batch_idx]
            
            optimizer.zero_grad()
            
            # Forward pass with auxiliary outputs
            yields, reconstruction, solvent_props = model(X_batch, return_aux=True)
            
            # Main task loss
            main_loss = mse_loss(yields, Y_batch)
            
            # Auxiliary task 1: Reconstruction loss
            recon_loss = mse_loss(reconstruction, X_batch)
            
            # Auxiliary task 2: Solvent property prediction loss
            props_loss = mse_loss(solvent_props, Y_spange_batch)
            
            # Combined loss
            loss = main_loss + aux_weight_recon * recon_loss + aux_weight_props * props_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model

# GP + MultiTask MLP + LGBM Ensemble
class MultiTaskEnsemble:
    def __init__(self, input_dim, weights=[0.15, 0.55, 0.30]):
        self.input_dim = input_dim
        self.weights = weights  # [GP, MLP, LGBM]
        self.gp_models = []
        self.mlp = None
        self.lgbm_models = []
        self.scaler = StandardScaler()
        self.spange_scaler = StandardScaler()
    
    def fit(self, X_train, Y_train, Y_spange_train):
        X_scaled = self.scaler.fit_transform(X_train)
        Y_spange_scaled = self.spange_scaler.fit_transform(Y_spange_train)
        
        # Train GP (one per target)
        self.gp_models = []
        for i in range(Y_train.shape[1]):
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
            gp.fit(X_scaled, Y_train[:, i])
            self.gp_models.append(gp)
        
        # Train Multi-Task MLP with auxiliary objectives
        self.mlp = train_multitask_mlp(
            X_scaled, Y_train, Y_spange_scaled, self.input_dim,
            epochs=300, aux_weight_recon=0.1, aux_weight_props=0.1
        )
        
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
        
        # GP predictions
        gp_pred = np.column_stack([m.predict(X_scaled) for m in self.gp_models])
        
        # MLP predictions (main task only)
        self.mlp.eval()
        with torch.no_grad():
            mlp_pred = self.mlp(torch.FloatTensor(X_scaled), return_aux=False).numpy()
        
        # LightGBM predictions
        lgbm_pred = np.column_stack([m.predict(X_scaled) for m in self.lgbm_models])
        
        # Weighted ensemble
        pred = (self.weights[0] * gp_pred + 
                self.weights[1] * mlp_pred + 
                self.weights[2] * lgbm_pred)
        
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
print("Running Single Solvent CV with Multi-Task Ensemble...")
print("="*60)

single_errors = {}
all_preds = []
all_true = []

for fold_idx, (train_idx, test_idx, solvent) in enumerate(generate_leave_one_solvent_out_splits(single_merged)):
    X_train = X_single[train_idx]
    Y_train = Y_single[train_idx]
    Y_spange_train = Y_spange_single[train_idx]
    X_test = X_single[test_idx]
    Y_test = Y_single[test_idx]
    
    model = MultiTaskEnsemble(input_dim=len(feature_cols))
    model.fit(X_train, Y_train, Y_spange_train)
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

print(f"\nMulti-Task Ensemble Single Solvent CV MSE: {single_mse:.6f} +/- {single_std:.6f}")

# Run CV for mixtures
print("\n" + "="*60)
print("Running Mixture CV with Multi-Task Ensemble...")
print("="*60)

mix_errors = {}
mix_preds_list = []
mix_true_list = []

for fold_idx, (train_idx, test_idx, mixture) in enumerate(generate_leave_one_ramp_out_splits(mix_merged)):
    X_train = X_mix[train_idx]
    Y_train = Y_mix[train_idx]
    Y_spange_train = Y_spange_mix[train_idx]
    X_test = X_mix[test_idx]
    Y_test = Y_mix[test_idx]
    
    model = MultiTaskEnsemble(input_dim=len(feature_cols))
    model.fit(X_train, Y_train, Y_spange_train)
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

print(f"\nMulti-Task Ensemble Mixture CV MSE: {mix_mse:.6f} +/- {mix_std:.6f}")

# Calculate overall CV score
print("\n" + "="*60)
print("Multi-Task Ensemble Overall Results")
print("="*60)

n_single = len(all_true)
n_mix = len(mix_true)
n_total = n_single + n_mix

overall_mse = (n_single * single_mse + n_mix * mix_mse) / n_total

print(f"\nSingle Solvent CV MSE: {single_mse:.6f} +/- {single_std:.6f} (n={n_single})")
print(f"Mixture CV MSE: {mix_mse:.6f} +/- {mix_std:.6f} (n={n_mix})")
print(f"Overall CV MSE: {overall_mse:.6f}")

print(f"\nBaseline (exp_030): CV = 0.008298")
improvement = (overall_mse - 0.008298) / 0.008298 * 100
print(f"Improvement vs baseline: {improvement:+.1f}%")

if overall_mse < 0.008298:
    print("\n✓ BETTER than baseline!")
else:
    print("\n✗ WORSE than baseline.")

# Final Summary
print("\n" + "="*60)
print("EXPERIMENT 059 SUMMARY")
print("="*60)

print(f"\nMulti-Task Ensemble (GP + MultiTask MLP + LGBM):")
print(f"  Features: Spange + DRFP + Arrhenius ({len(feature_cols)} features)")
print(f"  Weights: GP=0.15, MLP=0.55, LGBM=0.30")
print(f"  Auxiliary tasks: Reconstruction + Solvent property prediction")
print(f"\n  Single Solvent CV: {single_mse:.6f}")
print(f"  Mixture CV: {mix_mse:.6f}")
print(f"  Overall CV: {overall_mse:.6f}")
print(f"  vs Baseline (exp_030): {improvement:+.1f}%")

print(f"\nKey insights:")
print(f"1. Auxiliary tasks force model to learn generalizable representations")
print(f"2. Reconstruction task acts as regularization")
print(f"3. Solvent property prediction captures chemistry knowledge")

print(f"\nRemaining submissions: 5")
print(f"Best model: exp_030 (GP 0.15 + MLP 0.55 + LGBM 0.3) with CV 0.008298, LB 0.0877")
