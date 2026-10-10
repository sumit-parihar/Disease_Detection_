"""
review_sweep_pairs.py  -  Makes contact sheets for the extra pairs found by
phash_sweep.py, so you can check them by eye like the first 19.

WHICH PAIRS IT PICKS
--------------------
From manifests/phash_pairs_up_to_max.csv it keeps only pairs that
  - you have NOT reviewed yet, AND
  - are at pHash distance 6 (just above the limit of 5), OR
    are "cross" pairs (one source image and one target image).
(With your sweep result this is 42 + 2 = 44 pairs.)
Change EXTRA_DISTANCES below if you want other distances.

WHAT IT DOES
------------
Same as review_pairs.py: draws contact sheets (image A | image B | details)
and writes a csv with an empty "decision" column. Pair numbers start at 101
so they cannot be confused with the first 19 pairs.

  Output sheets : manifests/review_sweep/sweep_0101_0106.png, ...
  Output csv    : manifests/sweep_review.csv

DECISION RULE (same as before)
------------------------------
  duplicate : same leaf, same shot (maybe resized / re-saved / brighter)
  different : a different leaf that only looks similar
  unsure    : cannot tell (treated as duplicate)

ROUGH GUIDE FROM THE FIRST 19 PAIRS (not a rule, just a hint)
-------------------------------------------------------------
  pixel_diff about 5 to 15 : the real duplicates
  pixel_diff about 20 or more : different leaves
Always trust your eyes over this number.

HOW TO RUN (project root, venv active):
    python src/review_sweep_pairs.py             make sheets + csv
    python src/review_sweep_pairs.py --summarize count your decisions
"""

import argparse
import sys

import pandas as pd

from config import MANIFEST_DIR
from review_pairs import draw_sheet, open_image, pixel_difference, PAIRS_PER_SHEET

PAIRS_FILE = MANIFEST_DIR / "phash_pairs_up_to_max.csv"
SWEEP_REVIEW_FILE = MANIFEST_DIR / "sweep_review.csv"
SHEET_DIR = MANIFEST_DIR / "review_sweep"

EXTRA_DISTANCES = [6]        # distances to review besides the cross pairs
FIRST_PAIR_ID = 101
VALID_DECISIONS = {"duplicate", "different", "unsure"}


def dataset_of(path):
    """The manifest path tells us which dataset an image belongs to."""
    return "source" if "plantvillage" in path else "target"


def make_sheets(overwrite):
    if not PAIRS_FILE.exists():
        sys.exit(f"{PAIRS_FILE} not found. Run phash_sweep.py first.")
    if SWEEP_REVIEW_FILE.exists() and not overwrite:
        sys.exit(f"{SWEEP_REVIEW_FILE} already exists, so your decisions are kept. "
                 "Use --overwrite to start over.")

    pairs = pd.read_csv(PAIRS_FILE, keep_default_na=False)
    pairs["already_reviewed"] = pairs["already_reviewed"].astype(str) == "True"
    pairs["same_label"] = pairs["same_label"].astype(str) == "True"
    wanted = (pairs["phash_distance"].isin(EXTRA_DISTANCES)) | (pairs["scope"] == "cross")
    pairs = pairs[wanted & ~pairs["already_reviewed"]].reset_index(drop=True)
    if pairs.empty:
        print("No pairs to review.")
        return

    pairs["dataset_a"] = pairs["path_a"].map(dataset_of)
    pairs["dataset_b"] = pairs["path_b"].map(dataset_of)
    pairs["pixel_diff"] = [
        pixel_difference(open_image(a), open_image(b))
        for a, b in zip(pairs["path_a"], pairs["path_b"])
    ]
    # Cross pairs first (most important), then by distance and pixel_diff.
    pairs["is_cross"] = pairs["scope"] == "cross"
    pairs = pairs.sort_values(["is_cross", "phash_distance", "pixel_diff"],
                              ascending=[False, True, True]).reset_index(drop=True)
    pairs.insert(0, "pair_id", range(FIRST_PAIR_ID, FIRST_PAIR_ID + len(pairs)))
    pairs["decision"] = ""

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pairs), PAIRS_PER_SHEET):
        chunk = pairs.iloc[start:start + PAIRS_PER_SHEET]
        name = f"sweep_{chunk['pair_id'].iloc[0]:04d}_{chunk['pair_id'].iloc[-1]:04d}.png"
        draw_sheet(chunk, SHEET_DIR / name)

    pairs.drop(columns=["is_cross"]).to_csv(SWEEP_REVIEW_FILE, index=False, lineterminator="\n")
    print(f"{len(pairs)} pairs to review "
          f"({int((pairs['scope'] == 'cross').sum())} cross, "
          f"{int((pairs['scope'] != 'cross').sum())} at distance(s) {EXTRA_DISTANCES}).")
    print(f"Contact sheets : {SHEET_DIR}")
    print(f"Fill the 'decision' column in: {SWEEP_REVIEW_FILE}")
    print("Allowed values: duplicate / different / unsure")


def summarize():
    if not SWEEP_REVIEW_FILE.exists():
        sys.exit(f"{SWEEP_REVIEW_FILE} not found. Run without --summarize first.")
    review = pd.read_csv(SWEEP_REVIEW_FILE, keep_default_na=False)
    review["decision"] = review["decision"].str.strip().str.lower()
    bad = review[(review["decision"] != "") & (~review["decision"].isin(VALID_DECISIONS))]
    if len(bad):
        print(f"WARNING: {len(bad)} invalid decisions, e.g. pair_id {bad['pair_id'].head(5).tolist()}")
    print(f"Pairs total: {len(review)}   not yet decided: {int((review['decision'] == '').sum())}\n")
    print("Decisions by group:")
    review["group"] = review["scope"].where(review["scope"] == "cross", "distance " + review["phash_distance"].astype(str))
    print(pd.crosstab(review["group"], review["decision"]).to_string())
    real = review[review["decision"].isin(["duplicate", "unsure"])]
    print(f"\nTreated as duplicates (duplicate + unsure): {len(real)} of {len(review)}")
    cross_real = real[real["scope"] == "cross"]
    if len(cross_real):
        print(f"ATTENTION: {len(cross_real)} source-target pair(s) are duplicates. "
              "The SOURCE image of each must go on the exclusion list.")
        print(cross_real[["pair_id", "path_a", "path_b"]].to_string(index=False))
    print("\nRule of thumb: if only a few of the distance-6 pairs are real, keep the "
          "limit at 5 and handle those pairs one by one. If many are real, raise "
          "the limit to 6 and rerun the audit before the splits.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        make_sheets(args.overwrite)


if __name__ == "__main__":
    main()