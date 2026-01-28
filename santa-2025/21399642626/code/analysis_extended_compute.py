"""
Analysis: What would extended compute time achieve?

Key insight from discussions:
- Top teams run bbox3/SA for 24-72 HOURS with 24+ CPUs
- Our longest run was 53 minutes on 1 CPU
- That's 1/648th to 1/1944th of top competitor compute

Let's analyze what we could achieve with more compute.
"""

import subprocess
import time
import os

# Check if bbox3 exists and is executable
bbox3_path = "/home/code/experiments/bbox3"
if os.path.exists(bbox3_path):
    print(f"bbox3 found at {bbox3_path}")
    # Check if it's executable
    result = subprocess.run([bbox3_path, "-h"], capture_output=True, text=True, timeout=5)
    print(f"bbox3 help output:\n{result.stdout[:500] if result.stdout else result.stderr[:500]}")
else:
    print("bbox3 not found!")

# Check current best submission
print("\n=== Current best submission ===")
import pandas as pd
df = pd.read_csv('/home/code/experiments/029_final_ensemble_v2/submission.csv')
print(f"Rows: {len(df)}")
print(f"Sample:\n{df.head()}")
