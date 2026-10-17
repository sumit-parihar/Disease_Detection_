"""
make_target_manifest.py  -  Creates the LABEL-FREE list of target images for the
frozen inference. This closes the open item "label-free target inference manifest".

WHAT IT PRODUCES
----------------
manifests/target_inference_manifest.csv with exactly four columns and NO labels:
    path        image path (relative to the project root)
    sha256      hash of the image file
    cluster_id  id of the group of repeated photos the image belongs to
    cluster_size  number of images in that group
Only the 1,730 eligible images are listed: PP2020 images of the three shared classes
(healthy, scab, rust). The 91 multiple_diseases images are left out BEFORE inference,
which also removes Train_1173 (the copy of Train_379 that carries the multiple_diseases
label). Train_379 stays.

WHERE THE LABELS ARE USED, AND WHERE THEY ARE NOT
-------------------------------------------------
The label column of manifest.csv is read ONLY to decide which images are eligible (the
decision of the paper: multiple_diseases images are excluded before inference). The labels
are not written anywhere in the output, and the inference script reads only this file.
The clusters come from the duplicate reviews (what the photos look like), not from labels.

HOW THE CLUSTERS ARE BUILT (decision: repeated photos are resampled together)
----------------------------------------------------------------------------
Two eligible target images are joined into one cluster when
  - they have the same SHA-256 (exact copies), or
  - a duplicate review marked the pair "duplicate" or "unsure"
    (near_duplicate_review.csv, sweep_review.csv, target_review.csv).
Joining is transitive (A~B and B~C put A, B, C together). Images with no partner form a
cluster of their own.

CHECKS IT MAKES
---------------
  * the eligible counts equal the published ones (516 / 592 / 622 = 1,730)
  * every eligible image file still has the SHA-256 recorded in manifest.csv
  * every pair used from the review files has a decision (no empty decision)
  * no label information appears in the output file

THE FILE IS MEANT TO STAY FIXED
-------------------------------
It refuses to overwrite an existing manifest, and it refuses to run if target inference
results already exist.

HOW TO RUN (project root, venv active):
    python src/make_target_manifest.py
Before running, make sure the 'decision' column of manifests/target_review.csv is filled in.
"""

import argparse
import sys
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    EXPECTED_TARGET_COUNTS,
    MANIFEST_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    make_folders,
)
from frozen_pipeline import sha256_file

OUT_FILE = MANIFEST_DIR / "target_inference_manifest.csv"
OUT_SIDECAR = MANIFEST_DIR / "target_inference_manifest.csv.sha256"
OUT_SUMMARY = MANIFEST_DIR / "target_inference_manifest_summary.txt"
REVIEW_FILES = ["near_duplicate_review.csv", "sweep_review.csv", "target_review.csv"]
VALID = {"duplicate", "different", "unsure"}
OUTPUT_COLUMNS = ["path", "sha256", "cluster_id", "cluster_size"]


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def read_decisions(name):
    path = MANIFEST_DIR / name
    if not path.is_file():
        sys.exit(f"{path} not found.")
    table = pd.read_csv(path, keep_default_na=False)
    table["decision"] = table["decision"].astype(str).str.strip().str.lower()
    bad = table[~table["decision"].isin(VALID)]
    if len(bad):
        sys.exit(f"{name}: {len(bad)} pair(s) have no valid decision (for example pair_id {bad['pair_id'].head(3).tolist()}). "
                 "Fill the decision column (duplicate / different / unsure) and run again.")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace the manifest (only before inference)")
    args = parser.parse_args()
    make_folders()
    if list(RESULTS_DIR.glob("target_predictions_*")):
        sys.exit("Target inference results already exist. The manifest can no longer be changed.")
    if OUT_FILE.exists() and not args.overwrite:
        sys.exit(f"{OUT_FILE.name} already exists. It is meant to stay fixed. Use --overwrite only before inference.")

    manifest = pd.read_csv(MANIFEST_DIR / "manifest.csv", keep_default_na=False)
    target = manifest[manifest["dataset"] == "target"]
    eligible = target[target["label"].isin(CLASS_NAMES)].sort_values("path").reset_index(drop=True)
    counts = Counter(eligible["label"])
    if any(counts[name] != EXPECTED_TARGET_COUNTS[name] for name in CLASS_NAMES):
        sys.exit(f"Eligible counts {dict(counts)} differ from the published {EXPECTED_TARGET_COUNTS}.")
    paths = eligible["path"].tolist()
    path_set = set(paths)

    print(f"Checking the SHA-256 of {len(paths)} eligible image files ...")
    for path, recorded in tqdm(list(zip(eligible["path"], eligible["sha256"])), unit="img"):
        if sha256_file(PROJECT_ROOT / path) != recorded:
            sys.exit(f"File changed since the audit: {path}")

    finder = UnionFind(paths)
    reasons = Counter()
    by_hash = defaultdict(list)
    for path, sha in zip(eligible["path"], eligible["sha256"]):
        by_hash[sha].append(path)
    for members in by_hash.values():
        for other in members[1:]:
            finder.union(members[0], other)
            reasons["exact copy"] += 1
    for name in REVIEW_FILES:
        for _, row in read_decisions(name).iterrows():
            if row["decision"] in ("duplicate", "unsure") and row["path_a"] in path_set and row["path_b"] in path_set:
                finder.union(row["path_a"], row["path_b"])
                reasons[name] += 1

    members = defaultdict(list)
    for path in paths:
        members[finder.find(path)].append(path)
    ordered = sorted(members.values(), key=lambda m: m[0])
    cluster_of, size_of = {}, {}
    for number, group in enumerate(ordered, start=1):
        for path in group:
            cluster_of[path], size_of[path] = f"c{number:04d}", len(group)

    output = pd.DataFrame({"path": eligible["path"], "sha256": eligible["sha256"],
                           "cluster_id": [cluster_of[p] for p in paths], "cluster_size": [size_of[p] for p in paths]})
    assert list(output.columns) == OUTPUT_COLUMNS
    output.to_csv(OUT_FILE, index=False, lineterminator="\n")
    OUT_SIDECAR.write_text(f"{sha256_file(OUT_FILE)}  {OUT_FILE.name}\n", encoding="utf-8")

    multi = [g for g in ordered if len(g) > 1]
    lines = ["TARGET INFERENCE MANIFEST (label-free)", "=" * 50, "",
             f"Images listed      : {len(output)}  (the 91 multiple_diseases images are not listed)",
             f"Columns written    : {', '.join(OUTPUT_COLUMNS)}  (no labels)",
             f"Clusters           : {len(ordered)}  ({len(multi)} with more than one image, "
             f"{sum(len(g) for g in multi)} images in them)",
             f"Links used         : {dict(reasons)}",
             f"Manifest SHA-256   : {sha256_file(OUT_FILE)}"]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\nNext: commit this manifest, then run src/run_target_inference.py once.")


if __name__ == "__main__":
    main()