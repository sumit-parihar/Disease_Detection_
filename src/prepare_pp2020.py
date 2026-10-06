"""
prepare_pp2020.py  -  Separates the labelled Plant Pathology 2020 TRAIN
images from the unlabelled TEST images in the Kaggle download.

WHAT THE KAGGLE DOWNLOAD LOOKS LIKE (input)
-------------------------------------------
    data/pp2020_raw/
        train.csv              labels for 1,821 images
        test.csv               list of 1,821 test images (NO labels)
        sample_submission.csv  Kaggle upload format (not needed)
        images/
            Train_0.jpg ... Train_1820.jpg    labelled   -> we use these
            Test_0.jpg  ... Test_1820.jpg     unlabelled -> NOT used

WHAT THIS FILE DOES
-------------------
1. Reads data/pp2020_raw/train.csv.
2. COPIES only the images named in train.csv into data/pp2020/images/.
   It also copies train.csv to data/pp2020/train.csv.
3. Does NOT touch the raw folder (the paper says raw data stays unchanged).
4. Writes manifests/pp2020_excluded_test_files.txt, a list of the test
   images that were left out, so the audit can show they existed and were
   excluded on purpose.
5. Prints checks: row count, missing files, and class counts compared with
   the numbers published in the paper.

WHY WE NEED IT
--------------
The study needs correct/wrong labels to measure AURC. Test images have no
public labels, so they cannot be scored and must not enter the experiment.
Filtering by the names listed in train.csv (not by guessing from file names)
is the safest way to be sure only labelled images are used.

HOW TO RUN (from the project root, venv active):
    python src/prepare_pp2020.py
"""

import shutil
import sys

import pandas as pd

from config import (
    EXPECTED_TARGET_COUNTS,
    EXPECTED_TARGET_MULTIPLE,
    MANIFEST_DIR,
    PP2020_DIR,
    PP2020_IMAGES_DIR,
    PP2020_RAW_DIR,
    TARGET_LABEL_COLUMNS,
    TARGET_MULTIPLE_COLUMN,
    make_folders,
)

EXPECTED_ROWS = 1821  # labelled training images in the competition


def find_raw_images_folder():
    """The images may be in data/pp2020_raw/images/ or directly in
    data/pp2020_raw/ depending on how you unzipped. Find where they are."""
    for candidate in (PP2020_RAW_DIR / "images", PP2020_RAW_DIR):
        if any(candidate.glob("Train_*.jpg")):
            return candidate
    sys.exit(f"No Train_*.jpg files found inside {PP2020_RAW_DIR}. "
             "Check that you unzipped the Kaggle download there.")


def main():
    make_folders()

    raw_csv = PP2020_RAW_DIR / "train.csv"
    if not raw_csv.exists():
        sys.exit(f"{raw_csv} not found. Unzip the Kaggle download into "
                 f"{PP2020_RAW_DIR} first.")

    raw_images = find_raw_images_folder()
    print(f"Reading labels from : {raw_csv}")
    print(f"Reading images from : {raw_images}")

    # ---- Step 1: read the label file and check its shape ------------------
    labels = pd.read_csv(raw_csv)
    needed_columns = ["image_id"] + TARGET_LABEL_COLUMNS + [TARGET_MULTIPLE_COLUMN]
    missing_columns = [c for c in needed_columns if c not in labels.columns]
    if missing_columns:
        sys.exit(f"train.csv is missing columns: {missing_columns}")
    print(f"\nRows in train.csv   : {len(labels)} (paper/competition: {EXPECTED_ROWS})")

    # ---- Step 2: copy only the labelled images ---------------------------
    PP2020_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    missing_files = []
    copied = 0
    for image_id in labels["image_id"]:
        source_file = raw_images / f"{image_id}.jpg"
        target_file = PP2020_IMAGES_DIR / f"{image_id}.jpg"
        if not source_file.exists():
            missing_files.append(image_id)
            continue
        # Skip files already copied with the same size (safe to re-run).
        if target_file.exists() and target_file.stat().st_size == source_file.stat().st_size:
            continue
        shutil.copy2(source_file, target_file)
        copied += 1

    shutil.copy2(raw_csv, PP2020_DIR / "train.csv")
    print(f"Newly copied images : {copied}")
    print(f"Images now in {PP2020_IMAGES_DIR}: "
          f"{len(list(PP2020_IMAGES_DIR.glob('Train_*.jpg')))}")
    if missing_files:
        print(f"WARNING: {len(missing_files)} images listed in train.csv were "
              f"not found, for example {missing_files[:5]}")

    # ---- Step 3: record the test images that were left out ----------------
    test_files = sorted(p.name for p in raw_images.glob("Test_*.jpg"))
    excluded_list = MANIFEST_DIR / "pp2020_excluded_test_files.txt"
    excluded_list.write_text("\n".join(test_files) + "\n", encoding="utf-8")
    print(f"\nTest images excluded: {len(test_files)} "
          f"(names saved in {excluded_list.name})")

    # ---- Step 4: check class counts against the paper ---------------------
    print("\nClass counts (from train.csv) vs paper:")
    for column in TARGET_LABEL_COLUMNS:
        found = int(labels[column].sum())
        expected = EXPECTED_TARGET_COUNTS[column]
        status = "OK" if found == expected else "CHECK"
        print(f"  {column:<18} {found:>5}   expected {expected:>5}   {status}")
    found = int(labels[TARGET_MULTIPLE_COLUMN].sum())
    status = "OK" if found == EXPECTED_TARGET_MULTIPLE else "CHECK"
    print(f"  {TARGET_MULTIPLE_COLUMN:<18} {found:>5}   expected "
          f"{EXPECTED_TARGET_MULTIPLE:>5}   {status}")

    # Each image should have exactly one label of the four.
    label_sum = labels[TARGET_LABEL_COLUMNS + [TARGET_MULTIPLE_COLUMN]].sum(axis=1)
    bad_rows = int((label_sum != 1).sum())
    print(f"\nRows without exactly one label: {bad_rows} (should be 0)")


if __name__ == "__main__":
    main()