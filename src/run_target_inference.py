"""
run_target_inference.py  -  The ONE-TIME frozen inference on the PP2020 target images.
It produces predictions and reliability scores only. It never reads any label.

WHAT THE PROTOCOL SAYS
----------------------
Target labels may be used only after the predictions and reliability scores have been
produced, saved and hashed. Nothing may be changed in response to the target.

WHAT THIS SCRIPT DOES
---------------------
  1. Checks that the frozen system is unchanged (freeze_system.verify_frozen).
  2. Checks that the label-free manifest is the one recorded by make_target_manifest.py
     (its SHA-256 sidecar) and that every image file still matches its recorded hash.
  3. Refuses to run if target prediction files already exist (the "once" rule).
  4. For each of the three classifier seeds, runs every one of the 1,730 eligible images
     through the frozen classifier and the three frozen failure heads, using the same
     preparation as S-test (resize, normalisation, no augmentation).
  5. Saves, per seed, the logits, the predicted class and the scores of all 7 methods
     (B4 uses the frozen temperature). Then it saves a hash file for everything.

WHAT IT NEVER TOUCHES
---------------------
  * It does not open train.csv or manifest.csv. It reads only the label-free manifest.
  * It does not fit, calibrate or change anything. Temperatures and cut-offs are the frozen ones.

AFTER IT RUNS
-------------
Commit and push the prediction files and hash file FIRST. Only then join the labels
(src/evaluate_target.py, written as the next step).

OUTPUT FILES (results/)
-----------------------
  target_predictions_seed{seed}.csv   one row per image: predictions and all scores
  target_features_seed{seed}.npz      the 576 image numbers and the 3 scores (not tracked by Git)
  target_inference_hashes.txt         SHA-256 of the manifest, the freeze record and all outputs

HOW TO RUN (project root, venv active), once:
    python src/run_target_inference.py
There is no --overwrite option on purpose.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import CLASS_NAMES, EVAL_BATCH_SIZE, MANIFEST_DIR, PROJECT_ROOT, RESULTS_DIR, make_folders
from dataset import build_eval_transform, load_image, make_loader
from freeze_system import FREEZE_FILE, verify_frozen
from frozen_pipeline import HEAD_VARIANTS, load_frozen_classifier, load_frozen_head, run_images, sha256_file
from metrics import msp_score, softmax
from train_failure_head import CLASSIFIER_SEEDS

MANIFEST_FILE = MANIFEST_DIR / "target_inference_manifest.csv"
MANIFEST_SIDECAR = MANIFEST_DIR / "target_inference_manifest.csv.sha256"
SETTINGS_FILE = RESULTS_DIR / "frozen_calibration_settings.json"
EXPECTED_COLUMNS = ["path", "sha256", "cluster_id", "cluster_size"]
EXPECTED_IMAGES = 1730

OUT_PREDICTIONS = {s: RESULTS_DIR / f"target_predictions_seed{s}.csv" for s in CLASSIFIER_SEEDS}
OUT_FEATURES = {s: RESULTS_DIR / f"target_features_seed{s}.npz" for s in CLASSIFIER_SEEDS}
OUT_HASHES = RESULTS_DIR / "target_inference_hashes.txt"


class LabelFreeImages(Dataset):
    """Images listed in the label-free manifest. Each item is (image tensor, path)."""

    def __init__(self, manifest):
        self.paths = manifest["path"].tolist()
        self.transform = build_eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_image(self.paths[index])), self.paths[index]


def build_table(out, manifest, temperature):
    logits = out["logits"].astype(np.float64)
    probs = softmax(logits)
    ordered = np.sort(probs, axis=1)
    predicted = probs.argmax(axis=1)
    assert list(out["paths"]) == manifest["path"].tolist()
    return pd.DataFrame({
        "path": out["paths"],
        "cluster_id": manifest["cluster_id"].to_numpy(),
        "predicted_index": predicted,
        "predicted_class": [CLASS_NAMES[i] for i in predicted],
        "logit_healthy": logits[:, 0], "logit_scab": logits[:, 1], "logit_rust": logits[:, 2],
        "msp_score": 1.0 - ordered[:, -1],
        "entropy_score": -(probs * np.log(probs + 1e-12)).sum(axis=1),
        "margin_score": 1.0 - (ordered[:, -1] - ordered[:, -2]),
        "temp_scaled_msp_score": msp_score(logits, temperature),
        "head_proposed": out["head_scores"]["proposed"],
        "head_logits_only": out["head_scores"]["logits_only"],
        "head_features_only": out["head_scores"]["features_only"],
    })


def main():
    make_folders()
    outputs = list(OUT_PREDICTIONS.values()) + list(OUT_FEATURES.values()) + [OUT_HASHES]
    existing = [p for p in outputs if p.exists()]
    if existing:
        sys.exit(f"Target outputs already exist (for example {existing[0].name}). Inference is run once. "
                 "Deleting them and running again must be a deliberate, documented decision.")
    verify_frozen()
    print("Freeze record verified: every frozen file is unchanged.")

    if not MANIFEST_FILE.is_file() or not MANIFEST_SIDECAR.is_file():
        sys.exit("The label-free manifest is missing. Run make_target_manifest.py first.")
    if sha256_file(MANIFEST_FILE) != MANIFEST_SIDECAR.read_text(encoding="utf-8").split()[0]:
        sys.exit("The label-free manifest has changed since it was created.")
    manifest = pd.read_csv(MANIFEST_FILE, keep_default_na=False)
    if list(manifest.columns) != EXPECTED_COLUMNS:
        sys.exit(f"Unexpected columns in the manifest: {list(manifest.columns)}")
    if len(manifest) != EXPECTED_IMAGES or manifest["path"].duplicated().any():
        sys.exit(f"The manifest must list {EXPECTED_IMAGES} distinct images, found {len(manifest)}.")
    print(f"Checking the SHA-256 of {len(manifest)} image files ...")
    for path, recorded in tqdm(list(zip(manifest["path"], manifest["sha256"])), unit="img"):
        if sha256_file(PROJECT_ROOT / path) != recorded:
            sys.exit(f"Image file differs from the manifest: {path}")

    settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. No label is read.\n")
    loader = make_loader(LabelFreeImages(manifest), EVAL_BATCH_SIZE, shuffle=False)

    tables, features = {}, {}
    for seed in CLASSIFIER_SEEDS:
        print(f"Running classifier seed {seed} ...")
        classifier = load_frozen_classifier(seed, device)
        heads = {variant: load_frozen_head(variant, seed, device) for variant in HEAD_VARIANTS}
        out = run_images(loader, classifier, heads, device)
        for name, array in [("logits", out["logits"])] + [(v, s) for v, s in out["head_scores"].items()]:
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite numbers in {name} for seed {seed}.")
        tables[seed] = build_table(out, manifest, float(settings[str(seed)]["temperature"]))
        features[seed] = out

    for seed in CLASSIFIER_SEEDS:
        tables[seed].to_csv(OUT_PREDICTIONS[seed], index=False, lineterminator="\n")
        np.savez(OUT_FEATURES[seed], features=features[seed]["features"], logits=features[seed]["logits"],
                 paths=features[seed]["paths"])

    lines = ["SHA-256 checksums for the frozen target inference (no labels were read)", "",
             "INPUTS:"] + [f"{sha256_file(p)}  {p.relative_to(PROJECT_ROOT).as_posix()}" for p in [MANIFEST_FILE, FREEZE_FILE]]
    lines += ["", "OUTPUT FILES:"] + [f"{sha256_file(p)}  {p.relative_to(PROJECT_ROOT).as_posix()}"
                                       for p in outputs if p != OUT_HASHES]
    OUT_HASHES.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nPREDICTED CLASS COUNTS (no labels involved)")
    for seed in CLASSIFIER_SEEDS:
        counts = tables[seed]["predicted_class"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict()
        print(f"  seed {seed}: {counts}")
    print(f"\nSaved target_predictions_seed*.csv, target_features_seed*.npz and {OUT_HASHES.name} in {RESULTS_DIR}")
    print("NEXT: commit and push these files BEFORE any target label is opened.")


if __name__ == "__main__":
    main()