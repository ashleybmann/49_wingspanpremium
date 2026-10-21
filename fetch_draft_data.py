"""Fetch NBA draft combine anthropometrics and draft pick slots (2015-2024).

Outputs nba_draft_combine_wingspan.csv with barefoot height, wingspan,
wingspan_diff (wingspan - barefoot height, inches), and official draft slot.
"""

import re
import time
import unicodedata

import pandas as pd
from nba_api.stats.endpoints import draftcombineplayeranthro, drafthistory

YEARS = range(2015, 2025)
OUTPUT_CSV = "nba_draft_combine_wingspan.csv"
TIMEOUT = 60
PAUSE = 1.0
RETRIES = 3


def fetch(endpoint_cls, **kwargs):
    for attempt in range(1, RETRIES + 1):
        try:
            return endpoint_cls(timeout=TIMEOUT, **kwargs).get_data_frames()[0]
        except Exception as exc:
            if attempt == RETRIES:
                raise
            print(f"  retry {attempt} after error: {exc}")
            time.sleep(PAUSE * attempt * 2)


def clean_name(name):
    """Display-friendly name: strip accents and collapse whitespace."""
    if pd.isna(name):
        return name
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", name).strip()


def name_key(name):
    """Matching key: lowercase, no punctuation or generational suffixes."""
    key = re.sub(r"[^a-z ]", "", clean_name(name).lower())
    key = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", key)
    return re.sub(r"\s+", " ", key).strip()


def main():
    combine_frames, draft_frames = [], []
    for year in YEARS:
        print(f"Fetching {year}...")
        combine = fetch(draftcombineplayeranthro.DraftCombinePlayerAnthro, season_year=str(year))
        combine["DRAFT_YEAR"] = year
        combine_frames.append(combine)
        time.sleep(PAUSE)

        draft = fetch(drafthistory.DraftHistory, season_year_nullable=str(year))
        draft["DRAFT_YEAR"] = year
        draft_frames.append(draft)
        time.sleep(PAUSE)

    combine = pd.concat(combine_frames, ignore_index=True)[
        ["DRAFT_YEAR", "PLAYER_ID", "PLAYER_NAME", "POSITION", "HEIGHT_WO_SHOES", "WINGSPAN"]
    ]
    draft = pd.concat(draft_frames, ignore_index=True)[
        ["DRAFT_YEAR", "PERSON_ID", "PLAYER_NAME", "ROUND_NUMBER", "ROUND_PICK",
         "OVERALL_PICK", "TEAM_ABBREVIATION", "ORGANIZATION"]
    ].rename(columns={"PERSON_ID": "PLAYER_ID", "PLAYER_NAME": "DRAFT_NAME"})

    combine["NAME_KEY"] = combine["PLAYER_NAME"].map(name_key)
    draft["NAME_KEY"] = draft["DRAFT_NAME"].map(name_key)

    # Match on player ID first, then fall back to cleaned name within the same draft year.
    by_id = combine.merge(draft.drop(columns="NAME_KEY"), on=["DRAFT_YEAR", "PLAYER_ID"], how="inner")
    unmatched = combine[~combine.set_index(["DRAFT_YEAR", "PLAYER_ID"]).index.isin(
        by_id.set_index(["DRAFT_YEAR", "PLAYER_ID"]).index)]
    remaining_draft = draft[~draft.set_index(["DRAFT_YEAR", "PLAYER_ID"]).index.isin(
        by_id.set_index(["DRAFT_YEAR", "PLAYER_ID"]).index)]
    by_name = unmatched.merge(
        remaining_draft.drop(columns="PLAYER_ID"), on=["DRAFT_YEAR", "NAME_KEY"], how="inner"
    )
    merged = pd.concat([by_id, by_name], ignore_index=True)

    merged["PLAYER_NAME"] = merged["DRAFT_NAME"].map(clean_name)
    for col in ["HEIGHT_WO_SHOES", "WINGSPAN"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["HEIGHT_WO_SHOES", "WINGSPAN", "OVERALL_PICK"])
    merged["WINGSPAN_DIFF"] = (merged["WINGSPAN"] - merged["HEIGHT_WO_SHOES"]).round(2)

    out = (
        merged[["DRAFT_YEAR", "OVERALL_PICK", "ROUND_NUMBER", "ROUND_PICK", "PLAYER_ID",
                "PLAYER_NAME", "POSITION", "TEAM_ABBREVIATION", "ORGANIZATION",
                "HEIGHT_WO_SHOES", "WINGSPAN", "WINGSPAN_DIFF"]]
        .astype({"OVERALL_PICK": int, "ROUND_NUMBER": int, "ROUND_PICK": int})
        .sort_values(["DRAFT_YEAR", "OVERALL_PICK"])
        .reset_index(drop=True)
    )
    out.columns = out.columns.str.lower()
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nCombine rows: {len(combine)} | Draft picks: {len(draft)} | "
          f"Matched by ID: {len(by_id)} | by name: {len(by_name)}")
    print(f"Wrote {len(out)} rows to {OUTPUT_CSV}")
    print(out.groupby("draft_year").size().to_string())


if __name__ == "__main__":
    main()
