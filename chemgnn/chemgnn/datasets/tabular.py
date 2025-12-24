# chemgnn/datasets/tabular.py

import pandas as pd
from chemgnn.features.rdkit_descriptors import (
    compute_rdkit_descriptors,
    rf_feature_columns,
)

def build_rf_dataset(
    df: pd.DataFrame,
    smiles_col="SMILES",
    target_col="HoC",
):
    """
    Build RF-ready dataset from raw SMILES DataFrame.
    """
    df = compute_rdkit_descriptors(df, smiles_col=smiles_col)

    feature_cols = rf_feature_columns()

    X = df[feature_cols]
    y = df[target_col]

    return X, y
