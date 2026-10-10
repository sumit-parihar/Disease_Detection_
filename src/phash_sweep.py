"""
phash_sweep.py  -  Shows how many near-duplicate pairs exist at EVERY pHash
distance from 0 up to a chosen maximum, so you can choose the limit with the
full picture in front of you.

WHAT THIS FILE DOES
-------------------
audit_dataset.py only saved pairs with distance <= 5 (PHASH_MAX_DISTANCE).
This file looks at the SAME manifest again and counts pairs at distances
0, 1, 2, ... up to --max (default 10), so you can see what lies beyond 5.

  1. Reads manifests/manifest.csv (the pHash of every image is already in it,
     so the images are NOT opened again and the run takes seconds).
  2. Compares every image with every other image (about 9.5 million pairs).
  3. Leaves out exact copies (same SHA-256), they are already handled.
  4. Prints a table: pairs per distance, split into source-source,
     target-target and source-target (cross) pairs, plus how many pairs
     at that distance you have NOT reviewed yet.
  5. Saves two files in manifests/:
       phash_sweep_table.csv       the table
       phash_pairs_up_to_max.csv   every pair up to --max, with a column
                                   "already_reviewed"

WHAT IT DOES NOT DO
-------------------
It does not change the manifest, PHASH_MAX_DISTANCE or any image. Changing the
limit is a separate decision, made by editing config.py and (if needed)
re-running the audit with --overwrite BEFORE the splits.

HOW TO READ THE TABLE
---------------------
Real copies sit at small distances. Unrelated images sit around 28-36 bits.
If the number of pairs keeps growing quickly as the distance grows, the new
pairs are mostly look-alikes (false alarms), not real copies. Only a look at
the pictures can tell, so review a few of the NEW pairs at the next distance.

HOW TO RUN (project root, venv active):
    python src/phash_sweep.py
    python src/phash_sweep.py --max 12
"""

import argparse
import sys

import numpy as np
import pandas as pd

from config import MANIFEST_DIR, PHASH_MAX_DISTANCE

MANIFEST_FILE = MANIFEST_DIR / "manifest.csv"
REVIEW_FILE = MANIFEST_DIR / "near_duplicate_review.csv"
TABLE_FILE = MANIFEST_DIR / "phash_sweep_table.csv"
PAIRS_FILE = MANIFEST_DIR / "phash_pairs_up_to_max.csv"


def popcount_rows(values):
    """Number of 1-bits in each 64-bit integer = pHash distance after XOR."""
    as_bytes = values.view(np.uint8).reshape(-1, 8)
    return np.unpackbits(as_bytes, axis=1).sum(axis=1)


def reviewed_pairs():
    """Pairs you already looked at in near_duplicate_review.csv."""
    if not REVIEW_FILE.exists():
        return set()
    review = pd.read_csv(REVIEW_FILE, keep_default_na=False)
    return {frozenset((a, b)) for a, b in zip(review["path_a"], review["path_b"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=10,
                        help="largest pHash distance to count (default 10)")
    args = parser.parse_args()

    if not MANIFEST_FILE.exists():
        sys.exit(f"{MANIFEST_FILE} not found. Run audit_dataset.py first.")

    manifest = pd.read_csv(MANIFEST_FILE, keep_default_na=False)
    usable = manifest[manifest["decode_ok"].astype(str) == "True"].reset_index(drop=True)
    hashes = np.array([int(h, 16) for h in usable["phash"]], dtype=np.uint64)
    print(f"Comparing {len(usable)} images "
          f"({len(usable) * (len(usable) - 1) // 2:,} pairs)...")

    found = []
    for i in range(len(usable) - 1):
        distances = popcount_rows(hashes[i] ^ hashes[i + 1:])
        for offset in np.nonzero(distances <= args.max)[0]:
            j = i + 1 + int(offset)
            found.append((i, j, int(distances[offset])))

    seen = reviewed_pairs()
    rows = []
    for i, j, distance in found:
        a, b = usable.iloc[i], usable.iloc[j]
        if a["sha256"] == b["sha256"]:
            continue    # exact copy, handled by the exact-duplicate list
        pair_scope = "-".join(sorted([a["dataset"], b["dataset"]]))
        rows.append({
            "phash_distance": distance,
            "scope": {"source-source": "source-source",
                      "target-target": "target-target",
                      "source-target": "cross"}[pair_scope],
            "path_a": a["path"], "path_b": b["path"],
            "label_a": a["label"], "label_b": b["label"],
            "same_label": a["label"] == b["label"],
            "already_reviewed": frozenset((a["path"], b["path"])) in seen,
        })
    pairs = pd.DataFrame(rows, columns=[
        "phash_distance", "scope", "path_a", "path_b", "label_a", "label_b",
        "same_label", "already_reviewed"])
    pairs = pairs.sort_values(["phash_distance", "path_a"]).reset_index(drop=True)
    pairs.to_csv(PAIRS_FILE, index=False, lineterminator="\n")

    table = []
    for distance in range(args.max + 1):
        at = pairs[pairs["phash_distance"] == distance]
        table.append({
            "distance": distance,
            "pairs": len(at),
            "source-source": int((at["scope"] == "source-source").sum()),
            "target-target": int((at["scope"] == "target-target").sum()),
            "cross": int((at["scope"] == "cross").sum()),
            "different_label": int((~at["same_label"]).sum()),
            "not_reviewed_yet": int((~at["already_reviewed"]).sum()),
        })
    table = pd.DataFrame(table)
    table.to_csv(TABLE_FILE, index=False, lineterminator="\n")

    print(f"\nPairs per pHash distance (exact SHA-256 copies left out). "
          f"Current limit in config.py: {PHASH_MAX_DISTANCE}\n")
    print(table.to_string(index=False))
    cumulative = int(table["pairs"].sum())
    print(f"\nTotal pairs up to distance {args.max}: {cumulative}")
    print(f"Saved: {TABLE_FILE.name}, {PAIRS_FILE.name} (in {MANIFEST_DIR})")
    print("\nNothing was changed. To check a distance, look at the 'not_reviewed_yet' "
          "pairs in phash_pairs_up_to_max.csv.")


if __name__ == "__main__":
    main()