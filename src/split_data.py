"""
Split the feature table into a training set and a held-out test set.

Run from the project root:
    python -m src.split_data

Output:
    data/processed/train.csv -> used for training only (not committed)
    test_data.csv            -> held-out test set, uploaded to the Streamlit app (committed)
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.features import FEATURES, TARGET

INPUT_FILE = Path("data/processed/shots_features.csv")
TRAIN_FILE = Path("data/processed/train.csv")
TEST_FILE = Path("test_data.csv")

RANDOM_STATE = 42
N_SPLITS = 5  # keep one fold of five as the test set -> roughly 20% of the data


def split_by_match(df):
    """Hold out whole matches, keeping the goal rate similar in both sets."""
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(df, df[TARGET], groups=df["match_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def main():
    df = pd.read_csv(INPUT_FILE)
    train, test = split_by_match(df)

    # Safety check: no match may appear in both sets
    overlap = set(train["match_id"]) & set(test["match_id"])
    assert not overlap, f"{len(overlap)} matches appear in both train and test"

    train.to_csv(TRAIN_FILE, index=False)
    test[FEATURES + [TARGET]].to_csv(TEST_FILE, index=False)

    print("--- Train/test split ---")
    for name, part in [("Train", train), ("Test", test)]:
        print(
            f"{name:<6}: {len(part):>5} shots | "
            f"{part['match_id'].nunique():>3} matches | "
            f"goal % {part[TARGET].mean():.1%}"
        )
    print("Match overlap : none")
    print(f"\nSaved {TRAIN_FILE} and {TEST_FILE}")


if __name__ == "__main__":
    main()