"""
extract_calibration_outputs.py  -  Runs the FROZEN system on the S-calibration
images and saves what it produces. Nothing is trained or changed.

WHAT IT DOES
------------
For each of the three classifier seeds (42, 123, 2026):
  1. Loads the frozen disease classifier (and checks its SHA-256 against the hash
     recorded earlier).
  2. Loads the three frozen failure heads (proposed, logits-only, features-only).
  3. Runs every S-calibration image once (normal preparation, no augmentation).
  4. Saves, for every image: the true class, the predicted class, whether the
     prediction was right, the 3 scores, the simple "how unsure" scores (MSP,
     entropy, margin) and the 3 failure-head scores.

WHY
---
The next step (temperature scaling and the 80% / 90% cut-offs) works only from
these saved numbers, so it can be repeated exactly and without touching a model.

WHAT IT NEVER TOUCHES
---------------------
No target (PP2020) image or label, and no S-test image. S-test is used only once,
at the very end, by a separate deliberate step. No model file is changed.

OUTPUT FILES (in results/, one set per classifier seed)
-------------------------------------------------------
  s_calibration_outputs_seed{seed}.csv   one row per image (all scores)
  s_calibration_features_seed{seed}.npz  the 576 image numbers and the 3 scores
  s_calibration_outputs_summary.txt      counts and accuracy
  s_calibration_outputs_hashes.txt       SHA-256 of inputs and outputs

HOW TO RUN (project root, venv active):
    python src/extract_calibration_outputs.py
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from config import CLASS_NAMES, EVAL_BATCH_SIZE, PROJECT_ROOT, RESULTS_DIR, SPLITS_DIR, make_folders
from dataset import SourceDataset, make_loader
from frozen_pipeline import (
    HEAD_VARIANTS,
    classifier_path,
    head_path,
    load_frozen_classifier,
    load_frozen_head,
    run_images,
    sha256_file,
)
from train_failure_head import CLASSIFIER_SEEDS

SUBSET = "s_calibration"
SUMMARY_FILE = RESULTS_DIR / "s_calibration_outputs_summary.txt"
HASHES_FILE = RESULTS_DIR / "s_calibration_outputs_hashes.txt"
SPLIT_FILE = SPLITS_DIR / "split_assignments.csv"


def csv_path(seed):
    return RESULTS_DIR / f"s_calibration_outputs_seed{seed}.csv"


def npz_path(seed):
    return RESULTS_DIR / f"s_calibration_features_seed{seed}.npz"


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def process_seed(seed, device):
    classifier = load_frozen_classifier(seed, device)
    heads = {variant: load_frozen_head(variant, seed, device) for variant in HEAD_VARIANTS}
    dataset = SourceDataset(SUBSET)
    if len(dataset) == 0:
        raise ValueError(f"No images are assigned to {SUBSET} in {SPLIT_FILE.name}.")
    loader = make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False)
    out = run_images(loader, classifier, heads, device)

    # checks against the split file
    assigned = pd.read_csv(SPLIT_FILE)
    assigned = assigned[assigned["subset"] == SUBSET].sort_values("path")
    if list(out["paths"]) != assigned["path"].tolist():
        raise ValueError("Processed images do not match the S-calibration list in the split file.")
    if not np.isfinite(out["logits"]).all() or any(not np.isfinite(s).all() for s in out["head_scores"].values()):
        raise ValueError("Non-finite numbers found in the outputs.")

    probs = softmax(out["logits"].astype(np.float64))
    ordered = np.sort(probs, axis=1)
    predicted = probs.argmax(axis=1)
    table = pd.DataFrame({
        "path": out["paths"],
        "label": [CLASS_NAMES[i] for i in out["label_index"]],
        "label_index": out["label_index"],
        "predicted_index": predicted,
        "correct": (predicted == out["label_index"]).astype(int),
        "logit_healthy": out["logits"][:, 0], "logit_scab": out["logits"][:, 1], "logit_rust": out["logits"][:, 2],
        "msp_score": 1.0 - ordered[:, -1],
        "entropy_score": -(probs * np.log(probs + 1e-12)).sum(axis=1),
        "margin_score": 1.0 - (ordered[:, -1] - ordered[:, -2]),
        "head_proposed": out["head_scores"]["proposed"],
        "head_logits_only": out["head_scores"]["logits_only"],
        "head_features_only": out["head_scores"]["features_only"],
    })
    table.to_csv(csv_path(seed), index=False, lineterminator="\n")
    np.savez(npz_path(seed), features=out["features"], logits=out["logits"],
             label_index=out["label_index"], paths=out["paths"])
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace this script's own output files")
    args = parser.parse_args()
    make_folders()
    outputs = [csv_path(s) for s in CLASSIFIER_SEEDS] + [npz_path(s) for s in CLASSIFIER_SEEDS] + [SUMMARY_FILE, HASHES_FILE]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        sys.exit("Outputs of this script already exist (for example " + existing[0].name +
                 "). Use --overwrite only if you want to run again.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Only {SUBSET} images are read; target and S-test are not touched.\n")

    summary = ["S-CALIBRATION OUTPUTS OF THE FROZEN SYSTEM", "=" * 60, ""]
    for seed in CLASSIFIER_SEEDS:
        print(f"Classifier seed {seed} ...")
        table = process_seed(seed, device)
        errors = int((table["correct"] == 0).sum())
        counts = table["label"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict()
        summary += [f"Classifier seed {seed}: {len(table)} images {counts}",
                    f"  correct {len(table) - errors}, wrong {errors}, "
                    f"accuracy {100 * (1 - errors / len(table)):.2f}%"]
    summary += ["", "Note: temperature scaling uses the wrong predictions, so when there are only a few of them it",
                "is less reliable. The 80% / 90% cut-offs depend only on the ranking of the scores."]
    SUMMARY_FILE.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary))

    inputs = [SPLIT_FILE] + [classifier_path(s) for s in CLASSIFIER_SEEDS] + \
             [head_path(v, s) for s in CLASSIFIER_SEEDS for v in HEAD_VARIANTS]
    lines = ["SHA-256 checksums for the S-calibration outputs", "", "INPUT FILES (read-only):"]
    lines += [f"{sha256_file(p)}  {p.relative_to(PROJECT_ROOT).as_posix()}" for p in inputs]
    lines += ["", "OUTPUT FILES:"] + [f"{sha256_file(p)}  {p.relative_to(PROJECT_ROOT).as_posix()}"
                                       for p in outputs if p != HASHES_FILE]
    HASHES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved new files (names start with s_calibration_) in {RESULTS_DIR}")


if __name__ == "__main__":
    main()