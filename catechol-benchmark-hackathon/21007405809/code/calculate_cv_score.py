import pandas as pd
import numpy as np

DATA_PATH = '/home/data'
INPUT_LABELS_NUMERIC = ["Residence Time", "Temperature"]
INPUT_LABELS_SINGLE_SOLVENT = ["Residence Time", "Temperature", "SOLVENT NAME"]
INPUT_LABELS_FULL_SOLVENT = ["Residence Time", "Temperature", "SOLVENT A NAME", "SOLVENT B NAME", "SolventB%"]

def load_data(name="full"):
    if name == "full":
        df = pd.read_csv(f'{DATA_PATH}/catechol_full_data_yields.csv')
        X = df[INPUT_LABELS_FULL_SOLVENT]
    else:
        df = pd.read_csv(f'{DATA_PATH}/catechol_single_solvent_yields.csv')
        X = df[INPUT_LABELS_SINGLE_SOLVENT]
    Y = df[["Product 2", "Product 3", "SM"]]
    return X, Y

def generate_leave_one_out_splits(X, Y):
    for solvent in sorted(X["SOLVENT NAME"].unique()):
        mask = X["SOLVENT NAME"] != solvent
        yield (X[mask], Y[mask]), (X[~mask], Y[~mask])

def generate_leave_one_ramp_out_splits(X, Y):
    ramps = X[["SOLVENT A NAME", "SOLVENT B NAME"]].drop_duplicates()
    for _, row in ramps.iterrows():
        mask = ~((X["SOLVENT A NAME"] == row["SOLVENT A NAME"]) & (X["SOLVENT B NAME"] == row["SOLVENT B NAME"]))
        yield (X[mask], Y[mask]), (X[~mask], Y[~mask])

try:
    submission = pd.read_csv('/home/submission/submission.csv')
except FileNotFoundError:
    print("Submission file not found!")
    exit(1)

total_sq_error = 0
total_count = 0

# Single Solvent
X_single, Y_single = load_data("single_solvent")
split_gen_single = generate_leave_one_out_splits(X_single, Y_single)

print("Checking Single Solvent...")
for fold_idx, (_, (test_X, test_Y)) in enumerate(split_gen_single):
    sub_fold = submission[(submission['task'] == 0) & (submission['fold'] == fold_idx)].sort_values('row')
    if len(sub_fold) != len(test_Y):
        print(f"Mismatch in fold {fold_idx}: expected {len(test_Y)}, got {len(sub_fold)}")
    preds = sub_fold[['target_1', 'target_2', 'target_3']].values
    actuals = test_Y.values
    total_sq_error += np.sum((preds - actuals)**2)
    total_count += preds.size

# Full Data
X_full, Y_full = load_data("full")
split_gen_full = generate_leave_one_ramp_out_splits(X_full, Y_full)

print("Checking Full Data...")
for fold_idx, (_, (test_X, test_Y)) in enumerate(split_gen_full):
    sub_fold = submission[(submission['task'] == 1) & (submission['fold'] == fold_idx)].sort_values('row')
    if len(sub_fold) != len(test_Y):
        print(f"Mismatch in fold {fold_idx}: expected {len(test_Y)}, got {len(sub_fold)}")
    preds = sub_fold[['target_1', 'target_2', 'target_3']].values
    actuals = test_Y.values
    total_sq_error += np.sum((preds - actuals)**2)
    total_count += preds.size

overall_mse = total_sq_error / total_count
print(f"Overall MSE: {overall_mse}")
