"""
review_pairs.py  -  Helps YOU decide which near-duplicate candidates are
real duplicates.

WHAT THIS FILE DOES
-------------------
audit_dataset.py lists pairs of images whose pHash is close
(near_duplicate_candidates.csv). A small pHash distance only means "worth a
look". This file makes the looking easy:
  1. It reads the candidate pairs (exact SHA-256 copies are left out, they are
     already certain duplicates).
  2. For each pair it computes a quick extra hint: the average pixel
     difference after shrinking both images to the same small size
     (pixel_diff, 0 = identical, 255 = completely different).
  3. It draws "contact sheets": PNG pictures with several pairs per sheet, image A
     on the left, image B on the right, and the pair number, pHash distance,
     labels and pixel_diff written next to them.
  4. It writes near_duplicate_review.csv with an EMPTY column called
     "decision". You fill it in by hand.

THE DECISION RULE (fixed BEFORE reviewing, so results cannot bias it)
---------------------------------------------------------------------
  duplicate : same leaf, same shot. May be resized, re-saved or slightly
              brighter, but spots, edges and background match.
  different : a different leaf that only looks similar.
  unsure    : cannot tell. RULE: "unsure" is treated as "duplicate", i.e. both
              images stay in the same split group. This is the safe choice.

HOW TO RUN (project root, venv active)
--------------------------------------
  python src/review_pairs.py             make sheets + review csv
  (open manifests/review/*.png, fill the "decision" column in
   manifests/near_duplicate_review.csv with duplicate / different / unsure)
  python src/review_pairs.py --summarize  count decisions per pHash distance

Nothing is deleted or changed in the data. Decisions are only recorded.
The review csv is not overwritten if it exists (your decisions are safe);
use --overwrite to start the review again.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from config import MANIFEST_DIR, PROJECT_ROOT

NEAR_FILE = MANIFEST_DIR / "near_duplicate_candidates.csv"
REVIEW_FILE = MANIFEST_DIR / "near_duplicate_review.csv"
SHEET_DIR = MANIFEST_DIR / "review"

PAIRS_PER_SHEET = 6
THUMB = 256                      # each image is shown at 256 x 256
ROW_HEIGHT = THUMB + 8
TEXT_WIDTH = 330
VALID_DECISIONS = {"duplicate", "different", "unsure"}


def open_image(relative_path):
    return Image.open(PROJECT_ROOT / relative_path).convert("RGB")


def pixel_difference(image_a, image_b):
    """Average absolute difference per pixel after resizing both to 64 x 64
    grey. Small value = the pictures really are the same."""
    a = np.asarray(image_a.convert("L").resize((64, 64)), dtype=np.float32)
    b = np.asarray(image_b.convert("L").resize((64, 64)), dtype=np.float32)
    return float(np.abs(a - b).mean())


def draw_sheet(rows, sheet_path):
    """Draw one contact sheet. Each row: image A | image B | text."""
    width = THUMB * 2 + TEXT_WIDTH + 30
    height = ROW_HEIGHT * len(rows) + 4
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (_, row) in enumerate(rows.iterrows()):
        top = index * ROW_HEIGHT + 4
        for column, key in enumerate(["path_a", "path_b"]):
            image = open_image(row[key]).resize((THUMB, THUMB))
            sheet.paste(image, (column * (THUMB + 10) + 4, top))
        left = THUMB * 2 + 24
        lines = [
            f"PAIR {row['pair_id']}",
            f"pHash distance : {row['phash_distance']}",
            f"pixel_diff     : {row['pixel_diff']:.1f}",
            f"A: {row['dataset_a']} / {row['label_a']}",
            f"B: {row['dataset_b']} / {row['label_b']}",
            "same label" if row["same_label"] else "DIFFERENT labels",
            "",
            "A file: " + row["path_a"].split("/")[-1],
            "B file: " + row["path_b"].split("/")[-1],
        ]
        for line_number, text in enumerate(lines):
            draw.text((left, top + 6 + line_number * 20), text, fill="black")
        draw.line([(0, top + THUMB + 3), (width, top + THUMB + 3)], fill="gray")
    sheet.save(sheet_path)


def make_review(overwrite):
    if not NEAR_FILE.exists():
        sys.exit(f"{NEAR_FILE} not found. Run audit_dataset.py first.")
    if REVIEW_FILE.exists() and not overwrite:
        sys.exit(f"{REVIEW_FILE} already exists, so your decisions are kept. "
                 "Use --overwrite to start over.")

    pairs = pd.read_csv(NEAR_FILE)
    pairs = pairs[~pairs["same_sha256"]]     # exact copies need no review
    if pairs.empty:
        print("No near-duplicate candidates to review (besides exact copies).")
        return

    pairs = pairs.reset_index(drop=True)
    pixel_diffs = []
    for _, row in pairs.iterrows():
        pixel_diffs.append(pixel_difference(open_image(row["path_a"]),
                                            open_image(row["path_b"])))
    pairs["pixel_diff"] = pixel_diffs
    # Closest pairs first, so the clearest cases come first.
    pairs = pairs.sort_values(["phash_distance", "pixel_diff"]).reset_index(drop=True)
    pairs.insert(0, "pair_id", range(1, len(pairs) + 1))
    pairs["decision"] = ""

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pairs), PAIRS_PER_SHEET):
        chunk = pairs.iloc[start:start + PAIRS_PER_SHEET]
        name = f"pairs_{chunk['pair_id'].iloc[0]:04d}_{chunk['pair_id'].iloc[-1]:04d}.png"
        draw_sheet(chunk, SHEET_DIR / name)

    pairs.to_csv(REVIEW_FILE, index=False, lineterminator="\n")
    print(f"{len(pairs)} pairs to review.")
    print(f"Contact sheets : {SHEET_DIR}")
    print(f"Fill the 'decision' column in: {REVIEW_FILE}")
    print("Allowed values: duplicate / different / unsure")


def summarize():
    if not REVIEW_FILE.exists():
        sys.exit(f"{REVIEW_FILE} not found. Run without --summarize first.")
    review = pd.read_csv(REVIEW_FILE, keep_default_na=False)
    review["decision"] = review["decision"].str.strip().str.lower()
    bad = review[(review["decision"] != "") & (~review["decision"].isin(VALID_DECISIONS))]
    if len(bad):
        print(f"WARNING: {len(bad)} rows have an invalid decision, e.g. pair_id "
              f"{bad['pair_id'].head(5).tolist()}")
    empty = int((review["decision"] == "").sum())
    print(f"Pairs total: {len(review)}   not yet decided: {empty}\n")
    table = pd.crosstab(review["phash_distance"], review["decision"])
    print("Decisions per pHash distance:")
    print(table.to_string())
    decided = review[review["decision"].isin(VALID_DECISIONS)]
    if len(decided):
        as_duplicate = int(decided["decision"].isin(["duplicate", "unsure"]).sum())
        print(f"\nTreated as duplicates (duplicate + unsure): {as_duplicate} of {len(decided)} decided")
        edge = decided[decided["phash_distance"] >= decided["phash_distance"].max() - 1]
        if len(edge):
            share = edge["decision"].isin(["duplicate", "unsure"]).mean() * 100
            print(f"At the highest distances ({edge['phash_distance'].min()}-"
                  f"{edge['phash_distance'].max()}): {share:.0f}% are duplicates.")
            print("If this share is high, real copies may also exist just above "
                  "the limit; consider checking a larger limit.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        make_review(args.overwrite)


if __name__ == "__main__":
    main()