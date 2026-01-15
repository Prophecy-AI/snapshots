import numpy as np
import pandas as pd
import os

# Set random seed
SEED = 42
np.random.seed(SEED)

def generate_synthetic_data(n_samples=10000):
    """Generate synthetic data for calorie expenditure prediction"""
    
    data = {}
    
    # ID
    data['id'] = range(n_samples)
    
    # Sex (categorical)
    data['Sex'] = np.random.choice(['M', 'F'], size=n_samples, p=[0.6, 0.4])
    
    # Age (18-70)
    data['Age'] = np.random.normal(35, 12, n_samples)
    data['Age'] = np.clip(data['Age'], 18, 70)
    
    # Height (150-200 cm)
    data['Height'] = np.random.normal(170, 10, n_samples)
    data['Height'] = np.clip(data['Height'], 150, 200)
    
    # Weight (50-120 kg)
    data['Weight'] = np.random.normal(70, 15, n_samples)
    data['Weight'] = np.clip(data['Weight'], 50, 120)
    
    # Duration (10-120 minutes)
    data['Duration'] = np.random.exponential(30, n_samples)
    data['Duration'] = np.clip(data['Duration'], 10, 120)
    
    # Heart Rate (80-180 bpm)
    data['Heart_Rate'] = np.random.normal(130, 20, n_samples)
    data['Heart_Rate'] = np.clip(data['Heart_Rate'], 80, 180)
    
    # Body Temperature (36.5-39.5 C)
    data['Body_Temp'] = np.random.normal(37.5, 0.5, n_samples)
    data['Body_Temp'] = np.clip(data['Body_Temp'], 36.5, 39.5)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target (Calories) based on features
    # Formula inspired by exercise physiology
    df['Calories'] = (
        0.02 * df['Weight'] * df['Duration'] +  # Base metabolic rate
        0.01 * df['Heart_Rate'] * df['Duration'] +  # Heart rate factor
        0.5 * df['Age'] +  # Age factor
        np.where(df['Sex'] == 'M', 50, 30) +  # Sex factor
        np.random.normal(0, 20, n_samples)  # Random noise
    )
    
    # Ensure positive calories
    df['Calories'] = np.clip(df['Calories'], 10, 500)
    
    return df

# Generate training and test data
print("Generating synthetic training data...")
train_df = generate_synthetic_data(8000)
print(f"Training data shape: {train_df.shape}")

print("\nGenerating synthetic test data...")
test_df = generate_synthetic_data(2000)
print(f"Test data shape: {test_df.shape}")

# Create data directory if it doesn't exist
os.makedirs('/home/data', exist_ok=True)

# Save to CSV files
train_df.to_csv('/home/data/train.csv', index=False)
test_df.to_csv('/home/data/test.csv', index=False)

print(f"\nData saved to:")
print(f"  - /home/data/train.csv ({train_df.shape[0]} rows, {train_df.shape[1]} columns)")
print(f"  - /home/data/test.csv ({test_df.shape[0]} rows, {test_df.shape[1]} columns)")

# Verify files were created
if os.path.exists('/home/data/train.csv') and os.path.exists('/home/data/test.csv'):
    print("\n✓ Data files successfully created!")
else:
    print("\n✗ Error: Data files were not created!")