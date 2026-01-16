"""
Fix the duplicate ID issue in submission
Root cause: Merging on weight_capacity_r7 with duplicates in both DataFrames
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('/home/data/train.csv')
test = pd.read_csv('/home/data/test.csv')
original = pd.read_csv('/home/code/original_dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')

print("Extracting weight capacities...")
# Extract weight capacity from original dataset
original['weight_capacity'] = original['Weight Capacity (kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Extract from train and test
train['weight_capacity'] = train['Weight Capacity (kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
test['weight_capacity'] = test['Weight Capacity (kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

print("Computing original dataset features...")
# Compute orig_price: mean Price by Weight Capacity
orig_price = original.groupby('weight_capacity')['Price'].mean().reset_index()
orig_price.columns = ['weight_capacity', 'orig_price']

# Compute rounded versions - FIX: Use drop_duplicates to avoid merge expansion
for decimals in [7, 8, 9]:
    col_name = f'weight_capacity_r{decimals}'
    orig_price[col_name] = orig_price['weight_capacity'].round(decimals)
    
    # Compute mean by rounded weight capacity
    temp = original.copy()
    temp['weight_rounded'] = temp['weight_capacity'].round(decimals)
    rounded_price = temp.groupby('weight_rounded')['Price'].mean().reset_index()
    rounded_price.columns = [col_name, f'orig_price_r{decimals}']
    
    # Merge back - drop duplicates in orig_price to prevent expansion
    orig_price = orig_price.drop_duplicates(subset=[col_name]).merge(rounded_price, on=col_name, how='left')

print("Merging original features...")
# Merge original dataset features
def merge_orig_features(df, orig_features):
    df = df.copy()
    
    # Merge by exact weight capacity
    df = df.merge(orig_features[['weight_capacity', 'orig_price']], on='weight_capacity', how='left')
    
    # Merge by rounded weight capacity - drop duplicates first
    for decimals in [7, 8, 9]:
        # Drop duplicates in the mapping to prevent expansion
        mapping = orig_features[['weight_capacity_r' + str(decimals), 'orig_price_r' + str(decimals)]].drop_duplicates()
        df = df.merge(
            mapping,
            on='weight_capacity_r' + str(decimals),
            how='left'
        )
    
    return df

train = merge_orig_features(train, orig_price)
test = merge_orig_features(test, orig_price)

print("Creating COMBO features...")
# Identify categorical columns
cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Step 4a: NaN encoding
for col in cat_cols:
    train[col + '_isnan'] = train[col].isna().astype(int)
    test[col + '_isnan'] = test[col].isna().astype(int)

# Count total NaNs per row
train['NaNs'] = train[cat_cols].isna().sum(axis=1)
test['NaNs'] = test[cat_cols].isna().sum(axis=1)

# Step 4b: NaN × Weight Capacity
for col in cat_cols:
    train[f'{col}_nan_wc'] = train[col].isna().astype(int) * train['weight_capacity']
    test[f'{col}_nan_wc'] = test[col].isna().astype(int) * test['weight_capacity']

# Step 4c: {col}_wc features (factorized categorical × Weight Capacity)
for col in cat_cols:
    # Combine train and test for consistent factorization
    combined = pd.concat([train[col], test[col]], axis=0)
    
    # Factorize (NaN becomes -1, we add 1 to make it 0)
    codes, categories = pd.factorize(combined, sort=True)
    codes = codes + 1  # Shift so NaN becomes 0 instead of -1
    
    # Split back
    train[f'{col}_factorized'] = codes[:len(train)]
    test[f'{col}_factorized'] = codes[len(train):]
    
    # Create interaction with weight capacity
    train[f'{col}_wc'] = train[f'{col}_factorized'] * train['weight_capacity']
    test[f'{col}_wc'] = test[f'{col}_factorized'] * test['weight_capacity']

# Clean up factorized columns
for col in cat_cols:
    del train[f'{col}_factorized']
    del test[f'{col}_factorized']

print("Computing groupby statistics...")
# Define group keys
group_keys = ['weight_capacity'] + cat_cols

# Statistics to compute
stats_to_compute = ['mean', 'count', 'median']

def compute_groupby_stats(train_df, test_df, target_col='Price'):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for group_key in group_keys:
        # Initialize columns for each statistic
        for stat in stats_to_compute:
            train_df[f'{group_key}_{stat}_price'] = np.nan
            test_df[f'{group_key}_{stat}_price'] = np.nan
        
        # Compute statistics for each group
        if group_key == 'weight_capacity':
            # For numeric weight_capacity, use simple groupby
            grouped = train_df.groupby(group_key)[target_col].agg(['mean', 'count', 'median'])
            
            # Map back to train and test
            for stat in stats_to_compute:
                train_df[f'{group_key}_{stat}_price'] = train_df[group_key].map(grouped[stat])
                test_df[f'{group_key}_{stat}_price'] = test_df[group_key].map(grouped[stat])
        else:
            # For categorical columns
            # Factorize first to handle NaNs and get numeric codes
            combined = pd.concat([train_df[group_key], test_df[group_key]], axis=0)
            codes, _ = pd.factorize(combined, sort=True)
            codes = codes + 1  # Make NaN = 0
            
            train_df[f'{group_key}_code'] = codes[:len(train_df)]
            test_df[f'{group_key}_code'] = codes[len(train_df):]
            
            # Group by the code and compute statistics
            grouped = train_df.groupby(f'{group_key}_code')[target_col].agg(['mean', 'count', 'median'])
            
            # Map back
            for stat in stats_to_compute:
                train_df[f'{group_key}_{stat}_price'] = train_df[f'{group_key}_code'].map(grouped[stat])
                test_df[f'{group_key}_{stat}_price'] = test_df[f'{group_key}_code'].map(grouped[stat])
            
            # Clean up temporary code column
            train_df.drop(columns=[f'{group_key}_code'], inplace=True)
            test_df.drop(columns=[f'{group_key}_code'], inplace=True)
    
    return train_df, test_df

train, test = compute_groupby_stats(train, test)

print("Preparing features for training...")
# Get all feature columns
exclude_cols = ['Price', 'id', 'Weight Capacity', 'weight_capacity_r7', 'weight_capacity_r8', 'weight_capacity_r9']
exclude_cols += cat_cols

feature_cols = [col for col in train.columns if col not in exclude_cols]

print(f"Total features: {len(feature_cols)}")

# Prepare data
X_train = train[feature_cols]
y_train = train['Price']
X_test = test[feature_cols]

print("Training model...")
# XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': 4
}

# Train on full data
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train')],
    verbose_eval=100
)

print("Generating predictions...")
# Generate predictions
dtest = xgb.DMatrix(X_test)
predictions = model.predict(dtest)

print(f"Predictions shape: {predictions.shape}")
print(f"Test shape: {test.shape}")

# Create submission - ensure we use the correct index
submission_df = pd.DataFrame({
    'id': test['id'],
    'Price': predictions
})

print(f"Submission shape: {submission_df.shape}")
print(f"ID uniqueness: {submission_df['id'].nunique() == len(submission_df)}")

# Save submission
submission_df.to_csv('/home/code/submission_candidates/candidate_005_fixed.csv', index=False)
print("Fixed submission saved to: /home/code/submission_candidates/candidate_005_fixed.csv")
