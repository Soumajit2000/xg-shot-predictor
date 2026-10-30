import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.simplefilter("ignore")

# Tournaments to pull: (competition_name, season_name) exactly as StatsBomb names them.
COMPETITIONS = [
    ("FIFA World Cup", "2018"),
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2020"),
    ("UEFA Euro", "2024"),
]

RAW_DIR = Path("data/raw")
CACHE_DIR = RAW_DIR / "cache"
OUTPUT_FILE = RAW_DIR / "shots_raw.pkl"


def find_competitions(targets):
    """Translate human-readable names into StatsBomb competition/season IDs."""
    comps = sb.competitions()
    found = []
    for comp_name, season_name in targets:
        match = comps[
            (comps["competition_name"] == comp_name)
            & (comps["season_name"] == season_name)
        ]
        if match.empty:
            print(f"WARNING: '{comp_name} {season_name}' not found in open data, skipping.")
            continue
        row = match.iloc[0]
        found.append(
            {
                "competition_id": int(row["competition_id"]),
                "season_id": int(row["season_id"]),
                "label": f"{comp_name} {season_name}",
            }
        )
    return found


def load_match_shots(match_id):
    """Return all shots from one match, using the local cache when available."""
    cache_file = CACHE_DIR / f"shots_{match_id}.pkl"
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    events = sb.events(match_id=match_id)
    shots = events[events["type"] == "Shot"].copy()
    shots["match_id"] = match_id
    shots.to_pickle(cache_file)
    return shots


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_shots = []
    for comp in find_competitions(COMPETITIONS):
        matches = sb.matches(
            competition_id=comp["competition_id"], season_id=comp["season_id"]
        )
        print(f"{comp['label']}: {len(matches)} matches")

        for i, match_id in enumerate(matches["match_id"], start=1):
            shots = load_match_shots(match_id)
            shots["competition"] = comp["label"]
            all_shots.append(shots)
            if i % 10 == 0 or i == len(matches):
                print(f"  downloaded {i}/{len(matches)} matches")

    df = pd.concat(all_shots, ignore_index=True)
    df.to_pickle(OUTPUT_FILE)

    goal_rate = (df["shot_outcome"] == "Goal").mean()
    print("\n--- Summary ---")
    print(f"Matches : {df['match_id'].nunique()}")
    print(f"Shots   : {len(df)}")
    print(f"Columns : {df.shape[1]}")
    print(f"Goal %  : {goal_rate:.1%}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()