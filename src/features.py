"""
Turn raw StatsBomb shot events into a clean, model-ready feature table.

Run from the project root:
    python -m src.features

Output:
    data/processed/shots_features.csv -> one row per shot: features + target + metadata
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pitch geometry (StatsBomb coordinates: pitch is 120 x 80, attacking left -> right)
# ---------------------------------------------------------------------------
GOAL_CENTRE = (120.0, 40.0)
LEFT_POST = (120.0, 36.0)
RIGHT_POST = (120.0, 44.0)

# ---------------------------------------------------------------------------
# Feature contract: the single source of truth for which columns the models use.
# Training, the test CSV and the Streamlit app all rely on these lists.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "x",
    "y",
    "distance_to_goal",
    "shot_angle",
    "minute",
    "defenders_in_cone",
    "opponents_within_5yd",
    "keeper_distance_to_goal",
]
BOOLEAN_FEATURES = [
    "under_pressure",
    "first_time",
    "one_on_one",
    "aerial_won",
    "open_goal",
    "follows_dribble",
]
CATEGORICAL_FEATURES = [
    "body_part",
    "technique",
    "play_pattern",
    "shot_type",
]
FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_goal"
META_COLUMNS = ["match_id", "competition"]  # kept for splitting, never used as features

# Raw StatsBomb column -> our clean feature name
BOOLEAN_SOURCES = {
    "under_pressure": "under_pressure",
    "shot_first_time": "first_time",
    "shot_one_on_one": "one_on_one",
    "shot_aerial_won": "aerial_won",
    "shot_open_goal": "open_goal",
    "shot_follows_dribble": "follows_dribble",
}
CATEGORICAL_SOURCES = {
    "shot_body_part": "body_part",
    "shot_technique": "technique",
    "play_pattern": "play_pattern",
    "shot_type": "shot_type",
}

RAW_FILE = Path("data/raw/shots_raw.pkl")
OUTPUT_FILE = Path("data/processed/shots_features.csv")


def clean_shots(raw):
    """Keep only in-game, non-penalty shots."""
    df = raw[raw["period"] != 5]  # period 5 = penalty shootout
    df = df[df["shot_type"].isin(["Open Play", "Free Kick"])]  # drop penalties and the few direct corners
    return df.copy()


def _in_triangle(p, a, b, c):
    """True if point p lies inside triangle abc (edges included)."""
    def cross(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1, d2, d3 = cross(p, a, b), cross(p, b, c), cross(p, c, a)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def _freeze_frame_features(freeze_frame, x, y):
    """Summarise where the opponents stood at the moment of the shot."""
    if not isinstance(freeze_frame, list):
        return {"defenders_in_cone": np.nan, "opponents_within_5yd": np.nan,
                "keeper_distance_to_goal": np.nan}

    shooter = (x, y)
    in_cone, nearby, keeper_dist = 0, 0, np.nan

    for player in freeze_frame:
        if player.get("teammate"):
            continue
        px, py = player["location"][:2]
        is_keeper = player.get("position", {}).get("name") == "Goalkeeper"

        if is_keeper:
            keeper_dist = float(np.hypot(GOAL_CENTRE[0] - px, GOAL_CENTRE[1] - py))
        elif _in_triangle((px, py), shooter, LEFT_POST, RIGHT_POST):
            in_cone += 1

        if np.hypot(px - x, py - y) <= 5:
            nearby += 1

    return {"defenders_in_cone": in_cone, "opponents_within_5yd": nearby,
            "keeper_distance_to_goal": keeper_dist}


def build_features(df):
    """Create the feature table from cleaned shot events."""
    out = pd.DataFrame(index=df.index)

    # Location and geometry
    out["x"] = df["location"].str[0].astype(float)
    out["y"] = df["location"].str[1].astype(float)
    out["distance_to_goal"] = np.hypot(GOAL_CENTRE[0] - out["x"], GOAL_CENTRE[1] - out["y"])
    angle_left = np.arctan2(LEFT_POST[1] - out["y"], LEFT_POST[0] - out["x"])
    angle_right = np.arctan2(RIGHT_POST[1] - out["y"], RIGHT_POST[0] - out["x"])
    out["shot_angle"] = np.degrees(np.abs(angle_right - angle_left))

    out["minute"] = df["minute"].astype(int)

    # Defensive pressure at the moment of the shot
    ff = [
        _freeze_frame_features(frame, x, y)
        for frame, x, y in zip(df["shot_freeze_frame"], out["x"], out["y"])
    ]
    out = out.join(pd.DataFrame(ff, index=df.index))

    # StatsBomb only records these flags when True; empty means False
    for raw_col, name in BOOLEAN_SOURCES.items():
        out[name] = df[raw_col].eq(True).astype(int) if raw_col in df.columns else 0

    for raw_col, name in CATEGORICAL_SOURCES.items():
        out[name] = df[raw_col].fillna("Unknown").astype(str)

    out[TARGET] = df["shot_outcome"].eq("Goal").astype(int)
    for col in META_COLUMNS:
        out[col] = df[col]

    return out[FEATURES + [TARGET] + META_COLUMNS].reset_index(drop=True)


def main():
    raw = pd.read_pickle(RAW_FILE)
    shots = clean_shots(raw)
    data = build_features(shots)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)

    print("--- Feature table ---")
    print(f"Raw shots       : {len(raw)}")
    print(f"After cleaning  : {len(data)}")
    print(f"Features        : {len(FEATURES)}")
    print(f"Goal %          : {data[TARGET].mean():.1%}")
    print("\nMissing values per feature:")
    missing = data[FEATURES].isna().sum()
    print(missing[missing > 0] if missing.any() else "  none")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()