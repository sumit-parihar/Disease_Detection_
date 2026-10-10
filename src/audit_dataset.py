"""
audit_dataset.py  -  Step 2: checks both datasets BEFORE any training and
writes the "manifest", the official list of exactly which files we use.

WHAT THIS FILE DOES (paper, Section 3.3 and Appendix A)
-------------------------------------------------------
For every image in the SOURCE (PlantVillage, 3 classes) and the TARGET
(Plant Pathology 2020, labelled train images) it:
  1. Opens and fully decodes the image  -> finds corrupted files.
  2. Records file size, width, height, colour mode and channels.
  3. Computes a SHA-256 hash of the file -> two files with the same hash are
     byte-identical (EXACT duplicates).
  4. Computes a perceptual hash (pHash) -> two images that look the same, even
     if resized or re-saved, get hashes that differ by only a few bits
     (NEAR-duplicate CANDIDATES).
Then it:
  5. Counts images per class and compares with the numbers in the paper.
  6. Checks the target label file against the image folder (label-file
     consistency).
  7. Writes everything into the manifests/ folder.

IT DOES NOT DELETE OR CHANGE ANY IMAGE. It only reports. Near-duplicate
candidates must be reviewed by you (the paper says "manually reviewed").

OUTPUT FILES (in manifests/)
----------------------------
  manifest.csv                  one row per image (the official data list)
  manifest.csv.sha256           SHA-256 of manifest.csv (the "sidecar")
  duplicates_exact.csv          groups of byte-identical files
  near_duplicate_candidates.csv pairs of look-alike images to review
  audit_summary.txt             counts, problems found, pass/fail checks

WHY THE MANIFEST IS "IMMUTABLE"
-------------------------------
The paper says the manifest and its SHA-256 sidecar define the exact data
version of the experiment. So this script refuses to overwrite an existing
manifest unless you add --overwrite on purpose. File paths are stored with
forward slashes and the rows are sorted, so the same data gives the SAME
manifest on any computer. You can compare the two sidecar files from your two
PCs: if they match, the data is identical.

HOW TO RUN (project root, venv active):
    python src/audit_dataset.py
    python src/audit_dataset.py --overwrite     (only to redo the audit)
"""

import argparse
import hashlib
import sys

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import (
    EXPECTED_SOURCE_COUNTS,
    EXPECTED_TARGET_COUNTS,
    EXPECTED_TARGET_MULTIPLE,
    MANIFEST_DIR,
    PHASH_MAX_DISTANCE,
    PLANTVILLAGE_DIR,
    PP2020_CSV,
    PP2020_IMAGES_DIR,
    PROJECT_ROOT,
    SOURCE_FOLDERS,
    TARGET_LABEL_COLUMNS,
    TARGET_MULTIPLE_COLUMN,
    make_folders,
)

MANIFEST_FILE = MANIFEST_DIR / "manifest.csv"
SIDECAR_FILE = MANIFEST_DIR / "manifest.csv.sha256"
EXACT_FILE = MANIFEST_DIR / "duplicates_exact.csv"
NEAR_FILE = MANIFEST_DIR / "near_duplicate_candidates.csv"
SUMMARY_FILE = MANIFEST_DIR / "audit_summary.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CHANNELS_BY_MODE = {"L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}


# ---------------------------------------------------------------------------
# PART 1: make the list of files to audit
# ---------------------------------------------------------------------------
def list_source_files():
    """PlantVillage: the class is the name of the folder the image is in."""
    rows = []
    for class_name, folder in SOURCE_FOLDERS.items():
        class_dir = PLANTVILLAGE_DIR / folder
        if not class_dir.exists():
            sys.exit(f"Source folder not found: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"dataset": "source", "label": class_name, "path": path})
    return rows


def list_target_files(problems):
    """Plant Pathology 2020: the class comes from train.csv (4 yes/no
    columns). Also checks that csv and image folder agree."""
    if not PP2020_CSV.exists():
        sys.exit(f"Target label file not found: {PP2020_CSV}. "
                 "Run prepare_pp2020.py first.")
    labels = pd.read_csv(PP2020_CSV)
    label_columns = TARGET_LABEL_COLUMNS + [TARGET_MULTIPLE_COLUMN]

    # Each row must have exactly one label equal to 1.
    row_sum = labels[label_columns].sum(axis=1)
    bad = labels.loc[row_sum != 1, "image_id"].tolist()
    if bad:
        problems.append(f"{len(bad)} target rows without exactly one label, e.g. {bad[:5]}")

    rows, csv_ids = [], set()
    for _, row in labels.iterrows():
        image_id = row["image_id"]
        csv_ids.add(image_id)
        path = PP2020_IMAGES_DIR / f"{image_id}.jpg"
        if not path.exists():
            problems.append(f"Listed in train.csv but file missing: {image_id}")
            continue
        label = next((c for c in label_columns if row[c] == 1), "unlabelled")
        rows.append({"dataset": "target", "label": label, "path": path})

    # Image files that no csv row mentions.
    on_disk = {p.stem for p in PP2020_IMAGES_DIR.glob("*.jpg")}
    extra = sorted(on_disk - csv_ids)
    if extra:
        problems.append(f"{len(extra)} image files not listed in train.csv, e.g. {extra[:5]}")
    if any(name.startswith("Test_") for name in on_disk):
        problems.append("Test_* images are present in the target folder (they must not be)")
    rows.sort(key=lambda r: str(r["path"]))
    return rows


# ---------------------------------------------------------------------------
# PART 2: inspect one image
# ---------------------------------------------------------------------------
def sha256_of_file(path):
    """SHA-256 of the raw bytes. Same bytes -> same hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path):
    """Fully decode the image and collect its properties and hashes."""
    info = {
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of_file(path),
        "width": None, "height": None, "mode": None, "channels": None,
        "phash": None, "decode_ok": False, "error": "",
    }
    try:
        with Image.open(path) as image:
            image.load()   # forces full decoding; truncated files fail here
            info["width"], info["height"] = image.size
            info["mode"] = image.mode
            info["channels"] = CHANNELS_BY_MODE.get(image.mode)
            info["phash"] = str(imagehash.phash(image))   # 64 bits, as 16 hex characters
            info["decode_ok"] = True
    except Exception as error:   # any failure means the file is not usable
        # Only the error TYPE is stored: the full message contains the
        # computer-specific folder path, which would make the manifest
        # differ between your two PCs.
        info["error"] = type(error).__name__
    return info


# ---------------------------------------------------------------------------
# PART 3: find duplicates
# ---------------------------------------------------------------------------
def find_exact_duplicates(manifest):
    """Group files that share the same SHA-256."""
    rows = []
    for sha, group in manifest.groupby("sha256"):
        if len(group) < 2:
            continue
        datasets = sorted(group["dataset"].unique())
        scope = "cross_dataset" if len(datasets) > 1 else f"within_{datasets[0]}"
        rows.append({
            "sha256": sha,
            "count": len(group),
            "scope": scope,
            "labels": ";".join(sorted(group["label"].unique())),
            "paths": ";".join(group["path"]),
        })
    columns = ["sha256", "count", "scope", "labels", "paths"]
    return pd.DataFrame(rows, columns=columns)


def popcount_rows(values):
    """Number of 1-bits in each 64-bit integer (used for pHash distance)."""
    as_bytes = values.view(np.uint8).reshape(-1, 8)
    return np.unpackbits(as_bytes, axis=1).sum(axis=1)


def find_near_duplicates(manifest):
    """List pairs whose pHash differs by at most PHASH_MAX_DISTANCE bits.
    The Hamming distance between two hashes = number of differing bits."""
    usable = manifest[manifest["decode_ok"]].reset_index(drop=True)
    hashes = np.array([int(h, 16) for h in usable["phash"]], dtype=np.uint64)
    rows = []
    for i in range(len(usable) - 1):
        distances = popcount_rows(hashes[i] ^ hashes[i + 1:])
        for offset in np.nonzero(distances <= PHASH_MAX_DISTANCE)[0]:
            j = i + 1 + int(offset)
            a, b = usable.iloc[i], usable.iloc[j]
            rows.append({
                "path_a": a["path"], "path_b": b["path"],
                "dataset_a": a["dataset"], "dataset_b": b["dataset"],
                "label_a": a["label"], "label_b": b["label"],
                "phash_distance": int(distances[offset]),
                "same_sha256": a["sha256"] == b["sha256"],
                "same_label": a["label"] == b["label"],
            })
    columns = ["path_a", "path_b", "dataset_a", "dataset_b", "label_a", "label_b",
               "phash_distance", "same_sha256", "same_label"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# PART 4: summary text
# ---------------------------------------------------------------------------
def count_check(title, found, expected):
    lines = [title]
    for name, want in expected.items():
        have = int(found.get(name, 0))
        status = "OK" if have == want else "CHECK"
        lines.append(f"  {name:<20} found {have:>5}   paper {want:>5}   {status}")
    return lines


def build_summary(manifest, exact, near, problems):
    lines = ["DATASET AUDIT SUMMARY", "=" * 60, ""]
    lines.append(f"Files audited          : {len(manifest)}")
    lines.append(f"  source (PlantVillage): {(manifest['dataset'] == 'source').sum()}")
    lines.append(f"  target (PP2020)      : {(manifest['dataset'] == 'target').sum()}")

    failed = manifest[~manifest["decode_ok"]]
    lines += ["", f"Corrupted / undecodable: {len(failed)}"]
    for _, row in failed.head(20).iterrows():
        lines.append(f"  {row['path']}  ->  {row['error']}")

    ok = manifest[manifest["decode_ok"]]
    lines += ["", "Colour modes (channels):"]
    for (dataset, mode), n in ok.groupby(["dataset", "mode"]).size().items():
        lines.append(f"  {dataset:<7} {mode:<5} {n}")
    lines += ["", "Image size (width x height) range:"]
    for dataset, group in ok.groupby("dataset"):
        lines.append(f"  {dataset:<7} width {group['width'].min()}-{group['width'].max()}, "
                     f"height {group['height'].min()}-{group['height'].max()}")

    source_counts = manifest[manifest["dataset"] == "source"]["label"].value_counts().to_dict()
    target_counts = manifest[manifest["dataset"] == "target"]["label"].value_counts().to_dict()
    lines += [""] + count_check("Source class counts vs paper:", source_counts, EXPECTED_SOURCE_COUNTS)
    expected_target = dict(EXPECTED_TARGET_COUNTS)
    expected_target[TARGET_MULTIPLE_COLUMN] = EXPECTED_TARGET_MULTIPLE
    lines += [""] + count_check("Target class counts vs paper:", target_counts, expected_target)

    lines += ["", "Exact duplicates (same SHA-256):"]
    if exact.empty:
        lines.append("  none found")
    else:
        for scope, group in exact.groupby("scope"):
            lines.append(f"  {scope}: {len(group)} groups, {int(group['count'].sum())} files")
    lines += ["", f"Near-duplicate candidates (pHash distance <= {PHASH_MAX_DISTANCE}):"]
    if near.empty:
        lines.append("  none found")
    else:
        pairs = near[~near["same_sha256"]]
        lines.append(f"  {len(pairs)} pairs (exact-duplicate pairs excluded from this count)")
        cross = pairs[pairs["dataset_a"] != pairs["dataset_b"]]
        lines.append(f"  of which across source and target: {len(cross)}")
        different_label = pairs[~pairs["same_label"]]
        lines.append(f"  of which with different labels: {len(different_label)}")

    lines += ["", "Other problems:"]
    lines += [f"  {p}" for p in problems] if problems else ["  none"]
    lines += ["", "NEXT: review near_duplicate_candidates.csv by eye and decide which",
              "pairs are real duplicates. Nothing has been deleted."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing manifest (use only to redo the audit)")
    args = parser.parse_args()

    make_folders()
    if MANIFEST_FILE.exists() and not args.overwrite:
        sys.exit(f"{MANIFEST_FILE} already exists. The manifest is meant to stay "
                 "fixed. Use --overwrite only if you really want to redo the audit.")

    problems = []
    files = list_source_files() + list_target_files(problems)
    print(f"Auditing {len(files)} images (this takes a few minutes)...")

    records = []
    for item in tqdm(files, unit="img"):
        path = item["path"]
        record = {
            "dataset": item["dataset"],
            "label": item["label"],
            # relative path with forward slashes -> identical on every computer
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
        }
        record.update(inspect_image(path))
        records.append(record)

    manifest = pd.DataFrame(records).sort_values(["dataset", "path"]).reset_index(drop=True)
    # Keep whole numbers as whole numbers (a failed file would turn them into 64.0)
    for column in ["width", "height", "channels"]:
        manifest[column] = manifest[column].astype("Int64")

    exact = find_exact_duplicates(manifest)
    near = find_near_duplicates(manifest)

    manifest.to_csv(MANIFEST_FILE, index=False, lineterminator="\n")
    sidecar_hash = sha256_of_file(MANIFEST_FILE)
    SIDECAR_FILE.write_text(f"{sidecar_hash}  manifest.csv\n", encoding="utf-8")
    exact.to_csv(EXACT_FILE, index=False, lineterminator="\n")
    near.to_csv(NEAR_FILE, index=False, lineterminator="\n")

    summary = build_summary(manifest, exact, near, problems)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Manifest SHA-256: {sidecar_hash}")
    print(f"Files written to: {MANIFEST_DIR}")


if __name__ == "__main__":
    main()