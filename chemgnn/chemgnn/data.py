# chemgnn/data.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path):
    """Load CSV file."""
    return pd.read_csv(path, keep_default_na=False)


def split_train_test(
    X,
    y,
    test_size=0.2,
    random_state=41
):
    """
    Train / test split (RF-style).
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


def split_train_valid_test(
    df,
    train_frac=0.8,
    valid_frac=0.1,
    random_state=42
):
    """
    Train / validation / test split (GNN-style).

    Returns
    -------
    train_df, valid_df, test_df
    """
    if train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac must be < 1.0")

    # train split
    train_df = df.sample(frac=train_frac, random_state=random_state)

    remain_df = df.drop(train_df.index)

    # valid split from remaining
    valid_size = valid_frac / (1.0 - train_frac)
    valid_df = remain_df.sample(frac=valid_size, random_state=random_state)

    test_df = remain_df.drop(valid_df.index)

    return train_df, valid_df, test_df

