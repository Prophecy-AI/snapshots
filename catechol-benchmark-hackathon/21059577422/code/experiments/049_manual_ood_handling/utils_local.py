from typing import Any, Generator
import pandas as pd

DATA_PATH = '/home/data'

def load_data(name = "full"):
    assert name in ["full", "single_solvent"]
    if name == "full":
        df = pd.read_csv(f'{DATA_PATH}/catechol_full_data_yields.csv')
        X = df[INPUT_LABELS_FULL_SOLVENT]
    else:
        df = pd.read_csv(f'{DATA_PATH}/catechol_single_solvent_yields.csv')
        X = df[INPUT_LABELS_SINGLE_SOLVENT]

    Y = df[TARGET_LABELS]
    return X, Y

def load_features(name = "spange_descriptors"):
    assert name in ["spange_descriptors", "acs_pca_descriptors", "drfps_catechol", "fragprints", "smiles"]
    features = pd.read_csv(f'{DATA_PATH}/{name}_lookup.csv', index_col=0)
    return features

def generate_leave_one_out_splits(X, Y):
    """Generate all leave-one-out splits across the solvents."""
    all_solvents = X["SOLVENT NAME"].unique()
    for solvent_name in sorted(all_solvents):
        train_idcs_mask = X["SOLVENT NAME"] != solvent_name
        train_idx = X[train_idcs_mask].index.tolist()
        test_idx = X[~train_idcs_mask].index.tolist()
        yield (train_idx, test_idx)

def generate_leave_one_ramp_out_splits(X, Y):
    """Generate all leave-one-out splits across the solvent ramps."""
    all_solvent_ramps = X[["SOLVENT A NAME", "SOLVENT B NAME"]].drop_duplicates()
    all_solvent_ramps = all_solvent_ramps.sort_values(by=["SOLVENT A NAME", "SOLVENT B NAME"])
    for _, solvent_pair in all_solvent_ramps.iterrows():
        train_idcs_mask = (X[["SOLVENT A NAME", "SOLVENT B NAME"]] != solvent_pair).all(axis=1)
        train_idx = X[train_idcs_mask].index.tolist()
        test_idx = X[~train_idcs_mask].index.tolist()
        yield (train_idx, test_idx)

INPUT_LABELS_FULL_SOLVENT = [
    "Residence Time",
    "Temperature",
    "SOLVENT A NAME",
    "SOLVENT B NAME",
    "SolventB%",
]

INPUT_LABELS_SINGLE_SOLVENT = [
    "Residence Time",
    "Temperature",
    "SOLVENT NAME",
]

TARGET_LABELS = [
    "Product 2",
    "Product 3",
    "SM",
]
