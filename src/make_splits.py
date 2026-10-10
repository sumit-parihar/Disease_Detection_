"""
make_splits.py  -  Step 3: divides the PlantVillage (source) images into the
five subsets of the paper, keeping duplicate images together.

WHAT THIS FILE DOES (paper, Section 3.4)
----------------------------------------
Input : the 2,550 source images listed in manifests/manifest.csv.
Output: every image gets exactly one subset:
    s_train        50%   fit the disease classifier
    s_val          10%   model selection and early stopping
    s_reliability  15%   train and validate the failure-prediction head
    s_calibration  10%   temperature scaling and source-only thresholds
    s_test         15%   untouched in-domain reference

The split is
  * seeded       : the same seed (config.SEED) always gives the same result.
  * stratified   : each subset has about the same class mix
                   (healthy / scab / rust) as the whole source set.
  * group-aware  : images that are duplicates of each other form a GROUP, and
                   a whole group always goes into ONE subset. A group is never
                   cut between, for example, s_train and s_test.

WHERE THE GROUPS COME FROM (these 3 inputs, nothing else)
---------------------------------------------------------
  1. Exact copies    : source images with the same SHA-256 in manifest.csv
                       (the same information as duplicates_exact.csv).
  2. near_duplicate_review.csv : pairs YOU marked "duplicate" or "unsure".
  3. sweep_review.csv          : pairs YOU marked "duplicate" or "unsure".
Groups can chain: if A~B and B~C, then A, B and C are one group.
A pair with a different label is not special: it is grouped like any other pair
and the script reports it.

SOURCE IMAGES THAT MATCH A TARGET IMAGE
---------------------------------------
If a source image is a duplicate of a PP2020 image (exact copy, or a pair you
marked duplicate/unsure), the SOURCE image is left out of the split and written
to splits/source_exclusions.csv (paper rule: target stays untouched). The
image file itself is never deleted. In your data this list should be empty.

HOW THE IMAGES ARE ASSIGNED
---------------------------
For each class: (1) work out how many images each subset should get
(proportion x class size, rounded so the numbers add up); (2) sort the groups,
biggest first, with a seeded shuffle among equal sizes; (3) put each group into
the subset that is currently furthest below its target. Since almost all groups
are single images, the final counts end up almost exactly on target.

BEFORE IT RUNS
--------------
Every row of near_duplicate_review.csv and sweep_review.csv must have a decision
(duplicate / different / unsure). The script stops and tells you if not.

OUTPUT FILES (in splits/)
-------------------------
  split_assignments.csv         path, label, group_id, subset  (the result)
  split_assignments.csv.sha256  hash of that file
  groups_multi.csv              every group with more than one image, and why
  source_exclusions.csv         source images left out (normally empty)
  split_summary.txt             counts per subset and class, and the checks

Like the manifest, the split is meant to stay fixed. The script refuses to
overwrite it unless you add --overwrite.

HOW TO RUN (project root, venv active):
    python src/make_splits.py
"""

import argparse
import hashlib
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

from config import (
    CLASS_NAMES,
    MANIFEST_DIR,
    SEED,
    SPLIT_PROPORTIONS,
    SPLITS_DIR,
    make_folders,
)

MANIFEST_FILE = MANIFEST_DIR / "manifest.csv"
MANIFEST_SIDECAR = MANIFEST_DIR / "manifest.csv.sha256"
NEAR_REVIEW_FILE = MANIFEST_DIR / "near_duplicate_review.csv"
SWEEP_REVIEW_FILE = MANIFEST_DIR / "sweep_review.csv"
EXACT_FILE = MANIFEST_DIR / "duplicates_exact.csv"

ASSIGN_FILE = SPLITS_DIR / "split_assignments.csv"
ASSIGN_SIDECAR = SPLITS_DIR / "split_assignments.csv.sha256"
GROUPS_FILE = SPLITS_DIR / "groups_multi.csv"
EXCLUSION_FILE = SPLITS_DIR / "source_exclusions.csv"
SUMMARY_FILE = SPLITS_DIR / "split_summary.txt"

SUBSETS = list(SPLIT_PROPORTIONS.keys())
VALID_DECISIONS = {"duplicate", "different", "unsure"}
DUPLICATE_DECISIONS = {"duplicate", "unsure"}   # "unsure" counts as duplicate


# ---------------------------------------------------------------------------
# PART 1: read the inputs
# ---------------------------------------------------------------------------
def load_manifest():
    if not MANIFEST_FILE.exists():
        sys.exit(f"{MANIFEST_FILE} not found. Run audit_dataset.py first.")
    manifest = pd.read_csv(MANIFEST_FILE, keep_default_na=False)
    manifest["decode_ok"] = manifest["decode_ok"].astype(str) == "True"
    return manifest


def load_review(path, name):
    """Read a review file and make sure every pair has a valid decision."""
    if not path.exists():
        sys.exit(f"{path} not found. Run the review step for {name} first.")
    review = pd.read_csv(path, keep_default_na=False)
    review["decision"] = review["decision"].astype(str).str.strip().str.lower()
    empty = review[review["decision"] == ""]
    invalid = review[(review["decision"] != "") & ~review["decision"].isin(VALID_DECISIONS)]
    if len(empty) or len(invalid):
        message = [f"{name}: the decision column is not complete."]
        if len(empty):
            message.append(f"  {len(empty)} pairs have no decision, e.g. pair_id {empty['pair_id'].head(5).tolist()}")
        if len(invalid):
            message.append(f"  {len(invalid)} pairs have an invalid decision, e.g. pair_id {invalid['pair_id'].head(5).tolist()}")
        message.append("  Allowed values: duplicate / different / unsure. Fix the csv and run again.")
        sys.exit("\n".join(message))
    return review


# ---------------------------------------------------------------------------
# PART 2: find exclusions and build groups
# ---------------------------------------------------------------------------
class UnionFind:
    """Tiny helper that merges images into groups (A~B and B~C -> one group)."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # keep the alphabetically smaller path as root so results are stable
            if root_a < root_b:
                self.parent[root_b] = root_a
            else:
                self.parent[root_a] = root_b


def collect_links(manifest, reviews):
    """Return (links, exclusions).
    links      : list of (path_a, path_b, reason), both images are SOURCE images
    exclusions : dict source_path -> reason (source image matches a target image)"""
    dataset_of = dict(zip(manifest["path"], manifest["dataset"]))
    links, exclusions = [], {}

    # 1. exact copies: images that share a SHA-256
    usable = manifest[manifest["decode_ok"]]
    for sha, group in usable.groupby("sha256"):
        if len(group) < 2:
            continue
        source_paths = sorted(group[group["dataset"] == "source"]["path"])
        target_paths = sorted(group[group["dataset"] == "target"]["path"])
        if target_paths:
            for path in source_paths:
                exclusions[path] = f"exact copy of target image {target_paths[0]}"
        else:
            for other in source_paths[1:]:
                links.append((source_paths[0], other, "exact SHA-256 copy"))

    # 2 and 3. pairs you marked duplicate / unsure
    for review_name, review in reviews:
        for _, row in review.iterrows():
            if row["decision"] not in DUPLICATE_DECISIONS:
                continue
            a, b = row["path_a"], row["path_b"]
            kinds = {dataset_of[a], dataset_of[b]}
            reason = f"{review_name} pair {row['pair_id']} ({row['decision']})"
            if kinds == {"source"}:
                links.append((a, b, reason))
            elif kinds == {"source", "target"}:
                source_path, target_path = (a, b) if dataset_of[a] == "source" else (b, a)
                exclusions[source_path] = f"{reason}, matches target image {target_path}"
            # both target: not part of the source split, ignored here
    return links, exclusions


def build_groups(pool, links):
    """pool: DataFrame of source images that stay in the split.
    Returns dict group_id -> list of paths, and dict path -> group_id."""
    finder = UnionFind(pool["path"].tolist())
    in_pool = set(pool["path"])
    used_links = [(a, b, why) for a, b, why in links if a in in_pool and b in in_pool]
    for a, b, _ in used_links:
        finder.union(a, b)

    members = defaultdict(list)
    for path in sorted(pool["path"]):
        members[finder.find(path)].append(path)
    ordered = sorted(members.values(), key=lambda paths: paths[0])
    groups, group_of = {}, {}
    for number, paths in enumerate(ordered, start=1):
        group_id = f"g{number:04d}"
        groups[group_id] = paths
        for path in paths:
            group_of[path] = group_id
    return groups, group_of, used_links


# ---------------------------------------------------------------------------
# PART 3: the split itself
# ---------------------------------------------------------------------------
def target_counts(size):
    """How many of `size` images each subset should get (largest remainder)."""
    raw = {name: size * SPLIT_PROPORTIONS[name] for name in SUBSETS}
    counts = {name: int(value) for name, value in raw.items()}
    leftover = size - sum(counts.values())
    by_fraction = sorted(SUBSETS, key=lambda name: (-(raw[name] - counts[name]), SUBSETS.index(name)))
    for name in by_fraction[:leftover]:
        counts[name] += 1
    return counts


def assign_groups(groups, group_label, rng):
    """Put every group into one subset, class by class."""
    subset_of_group = {}
    wanted = {}
    for label in CLASS_NAMES:
        label_groups = sorted(g for g in groups if group_label[g] == label)
        total = sum(len(groups[g]) for g in label_groups)
        wanted[label] = target_counts(total)
        rng.shuffle(label_groups)                                   # seeded shuffle
        label_groups.sort(key=lambda g: -len(groups[g]))            # biggest first (stable)
        current = {name: 0 for name in SUBSETS}
        for group_id in label_groups:
            deficit = {name: wanted[label][name] - current[name] for name in SUBSETS}
            best = max(deficit.values())
            choices = [name for name in SUBSETS if deficit[name] == best]
            chosen = choices[0] if len(choices) == 1 else rng.choice(choices)
            subset_of_group[group_id] = chosen
            current[chosen] += len(groups[group_id])
    return subset_of_group, wanted


# ---------------------------------------------------------------------------
# PART 4: checks and summary
# ---------------------------------------------------------------------------
def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checks(assignments, groups, used_links, pool_size):
    results = []
    results.append(("every source image in the pool has exactly one subset",
                    len(assignments) == pool_size and assignments["path"].is_unique
                    and assignments["subset"].isin(SUBSETS).all()))
    spans = assignments.groupby("group_id")["subset"].nunique()
    results.append(("no group is split across subsets", bool((spans == 1).all())))
    subset_of = dict(zip(assignments["path"], assignments["subset"]))
    broken = [(a, b) for a, b, _ in used_links if subset_of[a] != subset_of[b]]
    results.append(("every duplicate pair (exact + reviewed) is in one subset", not broken))
    return results


def build_summary(assignments, wanted, groups, group_label, used_links, exclusions,
                  checks, manifest_hash, problems):
    lines = ["SPLIT SUMMARY", "=" * 60, ""]
    lines.append(f"Seed                    : {SEED}")
    lines.append(f"Manifest SHA-256 used   : {manifest_hash}")
    lines.append(f"Source images split     : {len(assignments)}")
    lines.append(f"Source images excluded  : {len(exclusions)} (see source_exclusions.csv)")
    multi = {g: p for g, p in groups.items() if len(p) > 1}
    lines.append(f"Groups                  : {len(groups)} "
                 f"({len(multi)} with more than one image, "
                 f"{sum(len(p) for p in multi.values())} images in them)")
    lines.append(f"Links used for grouping : {len(used_links)}")
    by_reason = Counter(why.split(" pair ")[0] for _, _, why in used_links)
    for reason, n in sorted(by_reason.items()):
        lines.append(f"    {reason:<28} {n}")

    lines += ["", "Images per subset and class (actual / target):"]
    header = f"  {'subset':<15}" + "".join(f"{label:>16}" for label in CLASS_NAMES) + f"{'total':>10}"
    lines.append(header)
    for name in SUBSETS:
        row = f"  {name:<15}"
        total = 0
        for label in CLASS_NAMES:
            actual = int(((assignments["subset"] == name) & (assignments["label"] == label)).sum())
            total += actual
            row += f"{f'{actual} / {wanted[label][name]}':>16}"
        row += f"{total:>10}"
        lines.append(row)
    lines.append("")
    share = assignments["subset"].value_counts(normalize=True)
    for name in SUBSETS:
        lines.append(f"  {name:<15} {100 * share.get(name, 0):5.1f}%  (paper: {100 * SPLIT_PROPORTIONS[name]:.0f}%)")

    lines += ["", "Checks:"]
    for text, ok in checks:
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {text}")
    lines += ["", "Notes:"]
    lines += [f"  {p}" for p in problems] if problems else ["  none"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing split (use only on purpose)")
    args = parser.parse_args()

    make_folders()
    if ASSIGN_FILE.exists() and not args.overwrite:
        sys.exit(f"{ASSIGN_FILE} already exists. The split is meant to stay fixed. "
                 "Use --overwrite only if you really want to redo it.")

    manifest = load_manifest()
    reviews = [("near_duplicate_review", load_review(NEAR_REVIEW_FILE, "near_duplicate_review.csv")),
               ("sweep_review", load_review(SWEEP_REVIEW_FILE, "sweep_review.csv"))]
    problems = []

    source = manifest[manifest["dataset"] == "source"]
    undecodable = source[~source["decode_ok"]]
    for path in undecodable["path"]:
        problems.append(f"left out, could not be decoded: {path}")

    links, exclusions = collect_links(manifest, reviews)
    for path in undecodable["path"]:
        exclusions.setdefault(path, "could not be decoded")

    pool = source[~source["path"].isin(exclusions)].reset_index(drop=True)
    groups, group_of, used_links = build_groups(pool, links)

    label_of = dict(zip(pool["path"], pool["label"]))
    group_label = {}
    for group_id, paths in groups.items():
        labels = Counter(label_of[p] for p in paths)
        if len(labels) > 1:
            problems.append(f"{group_id} contains different labels {dict(labels)}: "
                            f"grouped under the most common one")
        group_label[group_id] = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    rng = random.Random(SEED)
    subset_of_group, wanted = assign_groups(groups, group_label, rng)

    assignments = pd.DataFrame({
        "path": pool["path"],
        "label": pool["label"],
        "group_id": pool["path"].map(group_of),
    })
    assignments["subset"] = assignments["group_id"].map(subset_of_group)
    assignments = assignments.sort_values("path").reset_index(drop=True)

    checks = run_checks(assignments, groups, used_links, len(pool))

    # --- write the files ---------------------------------------------------
    assignments.to_csv(ASSIGN_FILE, index=False, lineterminator="\n")
    ASSIGN_SIDECAR.write_text(f"{sha256_of_file(ASSIGN_FILE)}  split_assignments.csv\n", encoding="utf-8")

    link_text = defaultdict(list)
    for a, b, why in used_links:
        link_text[group_of[a]].append(why)
    rows = [{
        "group_id": gid, "size": len(paths), "label": group_label[gid],
        "subset": subset_of_group[gid], "reasons": " | ".join(sorted(set(link_text[gid]))),
        "paths": ";".join(paths),
    } for gid, paths in groups.items() if len(paths) > 1]
    pd.DataFrame(rows, columns=["group_id", "size", "label", "subset", "reasons", "paths"]).to_csv(
        GROUPS_FILE, index=False, lineterminator="\n")

    pd.DataFrame(sorted(exclusions.items()), columns=["path", "reason"]).to_csv(
        EXCLUSION_FILE, index=False, lineterminator="\n")

    manifest_hash = MANIFEST_SIDECAR.read_text().split()[0] if MANIFEST_SIDECAR.exists() else "sidecar not found"
    summary = build_summary(assignments, wanted, groups, group_label, used_links,
                            exclusions, checks, manifest_hash, problems)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Files written to: {SPLITS_DIR}")
    if not all(ok for _, ok in checks):
        sys.exit("A check FAILED. Do not use this split. Send the summary above for help.")


if __name__ == "__main__":
    main()