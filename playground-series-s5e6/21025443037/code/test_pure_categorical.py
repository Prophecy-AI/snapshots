#!/usr/bin/env python3
"""
Quick test of pure categorical treatment without binning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import label_ranking_average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/home/data/train.csv')
test = pd.read_csv('/home/data/test.csv')

print("=== Pure Categorical Treatment Test ===")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Identify columns
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
cat_features = ['Soil Type', 'Crop Type']
target_col = 'Fertilizer Name'
id_col = 'Id'

# Check cardinality
print("\nCardinality of numerical features:")
for col in numerical_cols:
    unique_vals = train[col].nunique()
    print(f"  {col}: {unique_vals} unique values")

# Convert to categorical WITHOUT binning
train_proc = train.copy()
test_proc = test.copy()

# Convert numerical features to categorical using original values as strings
for col in numerical_cols:
    train_proc[col] = train_proc[col].astype(str).astype('category')
    test_proc[col] = test_proc[col].astype(str).astype('category')
    
    # Ensure test categories are subset of train
    missing_cats = set(test_proc[col].cat.categories) - set(train_proc[col].cat.categories)
    if missing_cats:
        train_proc[col] = train_proc[col].cat.add_categories(list(missing_cats))

# Label encode categorical features
for col in cat_features:
    le = LabelEncoder()
    train_proc[col] = le.fit_transform(train_proc[col]).astype('category')
    test_proc[col] = le.transform(test_proc[col]).astype('category')
    
    # Handle unseen categories
    unseen_mask = ~test_proc[col].isin(train_proc[col].cat.categories)
    if unseen_mask.any():
        train_proc[col] = train_proc[col].cat.add_categories([-1])
        test_proc.loc[unseen_mask, col] = -1

# Convert categories to integer codes for XGBoost
X = train_proc.drop([target_col, id_col], axis=1)
y = train_proc[target_col]
X_test = test_proc.drop([id_col], axis=1)

for col in X.columns:
    if X[col].dtype.name == 'category':
        X[col] = X[col].cat.codes
        X_test[col] = X_test[col].cat.codes

print(f"\nProcessed data shapes:")
print(f"X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")

# Test different hyperparameters
param_grid = {
    'max_depth': [6, 7, 8],
    'learning_rate': [0.03, 0.05, 0.07]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score = 0
best_params = None

print("\n=== Hyperparameter Tuning ===")
for depth in param_grid['max_depth']:
    for lr in param_grid['learning_rate']:
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                tree_method='hist',
                max_depth=depth,
                learning_rate=lr,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                device='cuda'
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            val_pred = model.predict_proba(X_val)
            
            y_val_bin = pd.get_dummies(y_val).values
            fold_map3 = label_ranking_average_precision_score(y_val_bin, val_pred)
            fold_scores.append(fold_map3)
        
        cv_score = np.mean(fold_scores)
        print(f"Depth {depth}, LR {lr}: CV = {cv_score:.4f} ± {np.std(fold_scores):.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            best_params = {'max_depth': depth, 'learning_rate': lr}

print(f"\n=== Results ===")
print(f"Best CV MAP@3: {best_score:.4f}")
print(f"Best params: {best_params}")
print(f"Baseline CV: 0.3311")
print(f"Improvement: {best_score - 0.3311:.4f}")

if best_score > 0.3311:
    print("\n✅ Pure categorical treatment IMPROVES performance!")
    print("Recommendation: Implement this approach in exp_003")
else:
    print("\n❌ Pure categorical treatment does NOT improve performance")
    print("Need to investigate further or try different approach")