from typing import Any, Generator

import pandas as pd


def load_data(name = "full"):
    assert name in ["full", "single_solvent"]
    if name == "full":
        df = pd.read_csv('/kaggle/input/catechol-benchmark-hackathon/catechol_full_data_yields.csv')
        X = df[INPUT_LABELS_FULL_SOLVENT]
    
    else:
        df = pd.read_csv('/kaggle/input/catechol-benchmark-hackathon/catechol_single_solvent_yields.csv')
        X = df[INPUT_LABELS_SINGLE_SOLVENT]

    Y = df[TARGET_LABELS]
    
    return X, Y

def load_features(name = "spange_descriptors"):
    assert name in ["spange_descriptors", "acs_pca_descriptors", "drfps_catechol", "fragprints", "smiles"]
    features = pd.read_csv(f'/kaggle/input/catechol-benchmark-hackathon/{name}_lookup.csv', index_col=0)
    return features

def generate_leave_one_out_splits(
    X: pd.DataFrame, Y: pd.DataFrame
) -> Generator[
    tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    Any,
    None,
]:
    """Generate all leave-one-out splits across the solvents.

    For each split, one of the solvents will be removed from the training set to
    make a test set.
    """

    all_solvents = X["SOLVENT NAME"].unique()
    for solvent_name in sorted(all_solvents):
        train_idcs_mask = X["SOLVENT NAME"] != solvent_name
        yield (
            (X[train_idcs_mask], Y[train_idcs_mask]),
            (X[~train_idcs_mask], Y[~train_idcs_mask]),
        )

def generate_leave_one_ramp_out_splits(
    X: pd.DataFrame, Y: pd.DataFrame
) -> Generator[
    tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    Any,
    None,
]:
    """Generate all leave-one-out splits across the solvent ramps.

    For each split, one of the solvent ramps will be removed from the training
    set to make a test set.
    """

    all_solvent_ramps = X[["SOLVENT A NAME", "SOLVENT B NAME"]].drop_duplicates()
    all_solvent_ramps.sort_values(by=["SOLVENT A NAME", "SOLVENT B NAME"])
    for _, solvent_pair in all_solvent_ramps.iterrows():
        train_idcs_mask = (X[["SOLVENT A NAME", "SOLVENT B NAME"]] != solvent_pair).all(
            axis=1
        )
        yield (
            (X[train_idcs_mask], Y[train_idcs_mask]),
            (X[~train_idcs_mask], Y[~train_idcs_mask]),
        )

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

INPUT_LABELS_NUMERIC = [
    "Residence Time",
    "Temperature",
]

INPUT_LABELS_SINGLE_FEATURES = [
    "SOLVENT NAME",
]

INPUT_LABELS_FULL_FEATURES = [
    "SOLVENT A NAME",
    "SOLVENT B NAME",
    "SolventB%"
]

TARGET_LABELS = [
    "Product 2",
    "Product 3",
    "SM",
]