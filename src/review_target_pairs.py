"""
review_target_pairs.py  -  Makes contact sheets for the PP2020 (target-target)
pairs at larger pHash distances that you have not looked at yet.

WHAT IT PICKS
-------------
From manifests/phash_pairs_up_to_max.csv (made by phash_sweep.py) it keeps
pairs where BOTH images are from PP2020, the distance is in DISTANCES below
(default 8 and 10) and the pair is not reviewed yet. With your sweep result
this is 2 pairs.

WHAT IT WRITES
--------------
  manifests/review_target/target_0201_0202.png   contact sheet(s)
  manifests/target_review.csv                    empty "decision" column

It does not touch sweep_review.csv or near_duplicate_review.csv, so your
earlier decisions are safe. Pair numbers start at 201.

DECISION RULE (same as before): duplicate / different / unsure.

HOW TO RUN (project root, venv active):
    python src/review_target_pairs.py
    python src/review_target_pairs.py --distances 8 10 12
"""

import argparse
import sys

import pandas as pd

from config import MANIFEST_DIR
from review_pairs import PAIRS_PER_SHEET, draw_sheet, open_image, pixel_difference

PAIRS_FILE = MANIFEST_DIR / "phash_pairs_up_to_max.csv"
OUT_CSV = MANIFEST_DIR / "target_review.csv"
SHEET_DIR = MANIFEST_DIR / "review_target"
FIRST_PAIR_ID = 201


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=int, nargs="+", default=[8, 10])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not PAIRS_FILE.exists():
        sys.exit(f"{PAIRS_FILE} not found. Run phash_sweep.py first.")
    if OUT_CSV.exists() and not args.overwrite:
        sys.exit(f"{OUT_CSV} already exists, so your decisions are kept. "
                 "Use --overwrite to start over.")

    pairs = pd.read_csv(PAIRS_FILE, keep_default_na=False)
    pairs["already_reviewed"] = pairs["already_reviewed"].astype(str) == "True"
    pairs["same_label"] = pairs["same_label"].astype(str) == "True"
    keep = ((pairs["scope"] == "target-target")
            & pairs["phash_distance"].isin(args.distances)
            & ~pairs["already_reviewed"])
    pairs = pairs[keep].reset_index(drop=True)
    if pairs.empty:
        print("No target-target pairs found at those distances.")
        return

    pairs["dataset_a"] = "target"
    pairs["dataset_b"] = "target"
    pairs["pixel_diff"] = [
        pixel_difference(open_image(a), open_image(b))
        for a, b in zip(pairs["path_a"], pairs["path_b"])
    ]
    pairs = pairs.sort_values(["phash_distance", "pixel_diff"]).reset_index(drop=True)
    pairs.insert(0, "pair_id", range(FIRST_PAIR_ID, FIRST_PAIR_ID + len(pairs)))
    pairs["decision"] = ""

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pairs), PAIRS_PER_SHEET):
        chunk = pairs.iloc[start:start + PAIRS_PER_SHEET]
        name = f"target_{chunk['pair_id'].iloc[0]:04d}_{chunk['pair_id'].iloc[-1]:04d}.png"
        draw_sheet(chunk, SHEET_DIR / name)

    pairs.to_csv(OUT_CSV, index=False, lineterminator="\n")
    print(f"{len(pairs)} target-target pairs to review.")
    print(f"Contact sheets : {SHEET_DIR}")
    print(f"Decisions file : {OUT_CSV}")


if __name__ == "__main__":
    main()