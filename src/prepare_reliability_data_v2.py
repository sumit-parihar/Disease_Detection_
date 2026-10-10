"""Prepare an expanded source-only S-reliability dataset (Version 2).

What this script does
---------------------
1. Verifies the original source split and reuses the existing, previously
   created 80/20 S-reliability train/validation split. It does NOT remake or
   overwrite either split.
2. Creates 13 views per original S-reliability image:
      - 1 original image
      - 3 brightness views
      - 3 contrast views
      - 3 Gaussian-blur views
      - 3 Gaussian-noise views
3. Samples a value independently for each perturbation view within the ranges
   already specified in the research methodology. Per-view seeds and sampled
   values are written to a manifest for reproducibility.
4. Runs the same views through the three frozen Phase B disease classifiers
   (seeds 42, 123, and 2026) and saves features, logits, predictions, and
   view-level correct/incorrect labels.
5. Reports positive-failure counts. It does NOT train a failure head and does
   NOT automatically change perturbation severity if the minimum is unmet.

This is a separate v2 run. Its outputs use v2 filenames, so the original
five-view experiment and its results are preserved. The existing source split,
secondary reliability split, and disease-classifier checkpoints are read-only.

Run from the project root with the virtual environment active:
    python src/prepare_reliability_data_v2.py

If you intentionally want to replace outputs created by THIS v2 script:
    python src/prepare_reliability_data_v2.py --overwrite
This never overwrites the original split, secondary split, or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    IMAGE_SIZE,
    MANIFEST_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    SPLITS_DIR,
    make_folders,
)
from dataset import IMAGENET_MEAN, IMAGENET_STD, load_image
from model import DiseaseClassifier

# ---------------------------------------------------------------------------
# Agreed protocol and already-specified perturbation ranges
# ---------------------------------------------------------------------------
RELIABILITY_FRACTION = 0.80
VALIDATION_FRACTION = 0.20
PERTURBATION_REPLICATES = 3  # approved v2 proposal: 3 samples per type
VIEW_TYPES = ("original", "brightness", "contrast", "gaussian_blur", "gaussian_noise")
VIEWS_PER_ORIGINAL = 1 + 4 * PERTURBATION_REPLICATES  # 13

BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)
BLUR_SIGMA_RANGE = (0.5, 1.5)
NOISE_SIGMA_RANGE = (0.01, 0.05)  # applied after ImageNet normalization
MIN_POSITIVE_TRAIN_FAILURES = 50  # report only; never automatically alter settings

PHASE_B_CHECKPOINTS = {
    42: MODELS_DIR / "disease_seed42_phaseB.pt",
    123: MODELS_DIR / "disease_seed123_phaseB.pt",
    2026: MODELS_DIR / "disease_seed2026_phaseB.pt",
}

SPLIT_ASSIGNMENTS = SPLITS_DIR / "split_assignments.csv"
SPLIT_ASSIGNMENTS_SHA = SPLITS_DIR / "split_assignments.csv.sha256"

# Read-only, already-created secondary split from the five-view run.
SECONDARY_SPLIT_FILE = SPLITS_DIR / "reliability_train_val.csv"
SECONDARY_SPLIT_SHA = SPLITS_DIR / "reliability_train_val.csv.sha256"

# Separate v2 output paths: do not overwrite the original five-view results.
VIEWS_MANIFEST_FILE = MANIFEST_DIR / "reliability_views_manifest_v2.csv"
SUMMARY_FILE = RESULTS_DIR / "reliability_generation_summary_v2.txt"
HASHES_FILE = RESULTS_DIR / "reliability_generation_hashes_v2.txt"
FEATURE_FILES = {
    seed: RESULTS_DIR / f"reliability_v2_features_seed{seed}.npz"
    for seed in PHASE_B_CHECKPOINTS
}
PREDICTION_FILES = {
    seed: RESULTS_DIR / f"reliability_v2_predictions_seed{seed}.csv"
    for seed in PHASE_B_CHECKPOINTS
}

RESIZE = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
TO_TENSOR = T.ToTensor()
NORMALIZE = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def read_and_verify_sha_sidecar(file_path: Path, sidecar_path: Path) -> str:
    if not file_path.is_file():
        raise FileNotFoundError(f"Required file not found: {file_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Required SHA-256 sidecar not found: {sidecar_path}")
    fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    if not fields:
        raise ValueError(f"SHA-256 sidecar is empty: {sidecar_path}")
    expected = fields[0].lower()
    actual = sha256_file(file_path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {file_path.name}.\n"
            f"Expected: {expected}\nActual:   {actual}\n"
            "Stop here; the existing split must not be changed by this script."
        )
    return actual


def verify_original_split() -> tuple[pd.DataFrame, str]:
    """Verify the fixed source split against its existing hash sidecar."""
    split_hash = read_and_verify_sha_sidecar(SPLIT_ASSIGNMENTS, SPLIT_ASSIGNMENTS_SHA)
    table = pd.read_csv(SPLIT_ASSIGNMENTS, keep_default_na=False)
    required = {"path", "label", "group_id", "subset"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Source split is missing columns: {sorted(missing)}")
    if table["path"].duplicated().any():
        raise ValueError("Duplicate image paths found in original source split.")
    if not table["label"].isin(CLASS_NAMES).all():
        raise ValueError("Unexpected labels found in original source split.")
    return table, split_hash


def load_existing_secondary_split(assignments: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Read and validate the prior 80/20 split; do not regenerate or overwrite it."""
    split_hash = read_and_verify_sha_sidecar(SECONDARY_SPLIT_FILE, SECONDARY_SPLIT_SHA)
    reliability = assignments.loc[assignments["subset"] == "s_reliability"].copy()
    expected_paths = set(reliability["path"])
    secondary = pd.read_csv(SECONDARY_SPLIT_FILE, keep_default_na=False)
    required = {"path", "label", "group_id", "source_subset", "reliability_split"}
    missing = required - set(secondary.columns)
    if missing:
        raise ValueError(f"Existing reliability split is missing columns: {sorted(missing)}")
    if secondary["path"].duplicated().any():
        raise ValueError("An original image appears more than once in the existing reliability split.")
    if set(secondary["path"]) != expected_paths:
        raise ValueError("Existing reliability split does not cover exactly the S-reliability source images.")

    # Ensure labels and group assignments still match the source split exactly.
    expected = reliability.set_index("path")[["label", "group_id"]].sort_index()
    observed = secondary.set_index("path")[["label", "group_id"]].sort_index()
    if not expected.equals(observed):
        raise ValueError("Existing reliability split labels/group IDs do not match the original source split.")
    if not (secondary["source_subset"] == "s_reliability").all():
        raise ValueError("Existing secondary split contains rows outside S-reliability.")
    allowed_parts = {"reliability_train", "reliability_val"}
    if not set(secondary["reliability_split"]).issubset(allowed_parts):
        raise ValueError("Unexpected reliability_split values found.")
    if set(secondary["reliability_split"]) != allowed_parts:
        raise ValueError("Both reliability_train and reliability_val must be present.")

    group_spans = secondary.groupby("group_id")["reliability_split"].nunique()
    if not bool((group_spans == 1).all()):
        raise ValueError("A duplicate group crosses reliability train/validation portions.")

    frac_train = float((secondary["reliability_split"] == "reliability_train").mean())
    frac_val = float((secondary["reliability_split"] == "reliability_val").mean())
    if abs(frac_train - RELIABILITY_FRACTION) > 0.03 or abs(frac_val - VALIDATION_FRACTION) > 0.03:
        raise ValueError(
            "The saved secondary split is not approximately 80/20: "
            f"train={frac_train:.3f}, validation={frac_val:.3f}."
        )

    secondary = secondary.sort_values("path").reset_index(drop=True)
    print("Reusing and verifying the existing secondary split (not modifying it):")
    print(f"  SHA-256: {split_hash}")
    print(f"  Original images: {len(secondary)}")
    print("  Class counts (train / validation):")
    for label in CLASS_NAMES:
        n_train = int(((secondary["label"] == label) &
                       (secondary["reliability_split"] == "reliability_train")).sum())
        n_val = int(((secondary["label"] == label) &
                     (secondary["reliability_split"] == "reliability_val")).sum())
        print(f"    {label:<8} {n_train:>4} / {n_val:<4}")
    print(f"  Multi-image groups in S-reliability: {int((secondary.groupby('group_id').size() > 1).sum())}")
    return secondary, split_hash


def stable_seed(relative_path: str, view_type: str, replicate_index: int, root_seed: int) -> int:
    """Derive a stable seed for each original-image/view/replicate combination."""
    token = f"{root_seed}\0{relative_path}\0{view_type}\0{replicate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**63 - 1)


def normalize_pil(image) -> torch.Tensor:
    """Convert a resized PIL image to an ImageNet-normalized tensor."""
    return NORMALIZE(TO_TENSOR(image))


def create_views(relative_path: str, label: str, group_id: str,
                 reliability_split: str, root_seed: int):
    """Generate the original plus 3 sampled versions of each approved perturbation."""
    source_image = load_image(relative_path)
    resized = RESIZE(source_image)
    original_tensor = normalize_pil(resized)
    path_key = relative_path.replace("\\", "/")

    views: list[torch.Tensor] = [original_tensor]
    parameter_rows: list[dict] = [{
        "view_type": "original", "replicate_index": 0, "rng_seed": "",
        "brightness_factor": "", "contrast_factor": "", "blur_sigma": "", "noise_sigma": "",
    }]

    # Use a separate, reproducible seed per original image, type, and replicate.
    for replicate in range(1, PERTURBATION_REPLICATES + 1):
        brightness_seed = stable_seed(path_key, "brightness", replicate, root_seed)
        rng = np.random.default_rng(brightness_seed)
        brightness_factor = float(rng.uniform(*BRIGHTNESS_RANGE))
        views.append(normalize_pil(TF.adjust_brightness(resized, brightness_factor)))
        parameter_rows.append({
            "view_type": "brightness", "replicate_index": replicate,
            "rng_seed": brightness_seed, "brightness_factor": brightness_factor,
            "contrast_factor": "", "blur_sigma": "", "noise_sigma": "",
        })

    for replicate in range(1, PERTURBATION_REPLICATES + 1):
        contrast_seed = stable_seed(path_key, "contrast", replicate, root_seed)
        rng = np.random.default_rng(contrast_seed)
        contrast_factor = float(rng.uniform(*CONTRAST_RANGE))
        views.append(normalize_pil(TF.adjust_contrast(resized, contrast_factor)))
        parameter_rows.append({
            "view_type": "contrast", "replicate_index": replicate,
            "rng_seed": contrast_seed, "brightness_factor": "",
            "contrast_factor": contrast_factor, "blur_sigma": "", "noise_sigma": "",
        })

    for replicate in range(1, PERTURBATION_REPLICATES + 1):
        blur_seed = stable_seed(path_key, "gaussian_blur", replicate, root_seed)
        rng = np.random.default_rng(blur_seed)
        blur_sigma = float(rng.uniform(*BLUR_SIGMA_RANGE))
        blur_image = resized.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        views.append(normalize_pil(blur_image))
        parameter_rows.append({
            "view_type": "gaussian_blur", "replicate_index": replicate,
            "rng_seed": blur_seed, "brightness_factor": "", "contrast_factor": "",
            "blur_sigma": blur_sigma, "noise_sigma": "",
        })

    for replicate in range(1, PERTURBATION_REPLICATES + 1):
        noise_seed = stable_seed(path_key, "gaussian_noise", replicate, root_seed)
        rng = np.random.default_rng(noise_seed)
        noise_sigma = float(rng.uniform(*NOISE_SIGMA_RANGE))
        noise = rng.normal(loc=0.0, scale=noise_sigma,
                           size=tuple(original_tensor.shape)).astype(np.float32)
        views.append(original_tensor + torch.from_numpy(noise))
        parameter_rows.append({
            "view_type": "gaussian_noise", "replicate_index": replicate,
            "rng_seed": noise_seed, "brightness_factor": "", "contrast_factor": "",
            "blur_sigma": "", "noise_sigma": noise_sigma,
        })

    if len(views) != VIEWS_PER_ORIGINAL or len(parameter_rows) != VIEWS_PER_ORIGINAL:
        raise RuntimeError(f"Expected {VIEWS_PER_ORIGINAL} views, got {len(views)} for {relative_path}")

    metadata = []
    for view_tensor, parameters in zip(views, parameter_rows):
        view_type = parameters["view_type"]
        replicate = int(parameters["replicate_index"])
        view_id = hashlib.sha256(
            f"{path_key}\0{view_type}\0{replicate}".encode("utf-8")
        ).hexdigest()[:20]
        if tuple(view_tensor.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Wrong tensor shape for {relative_path} ({view_type}-{replicate}): {view_tensor.shape}")
        if not bool(torch.isfinite(view_tensor).all()):
            raise RuntimeError(f"Non-finite values in view {view_type}-{replicate} for {relative_path}")
        metadata.append({
            "view_id": view_id,
            "path": relative_path,
            "label": label,
            "label_index": CLASS_TO_INDEX[label],
            "group_id": group_id,
            "source_subset": "s_reliability",
            "reliability_split": reliability_split,
            "view_type": view_type,
            "replicate_index": replicate,
            "rng_seed": parameters["rng_seed"],
            "brightness_factor": parameters["brightness_factor"],
            "contrast_factor": parameters["contrast_factor"],
            "blur_sigma": parameters["blur_sigma"],
            "noise_sigma": parameters["noise_sigma"],
            "view_generation_seed": root_seed,
        })
    return views, metadata


def load_frozen_models(device: torch.device) -> dict[int, DiseaseClassifier]:
    """Load and freeze the existing Phase B classifiers."""
    models: dict[int, DiseaseClassifier] = {}
    for seed, checkpoint_path in PHASE_B_CHECKPOINTS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Required Phase B checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("seed") != seed or checkpoint.get("phase") != "B":
            raise ValueError(
                f"Checkpoint metadata mismatch in {checkpoint_path.name}: "
                f"seed={checkpoint.get('seed')}, phase={checkpoint.get('phase')}"
            )
        if checkpoint.get("class_names") != CLASS_NAMES:
            raise ValueError(
                f"Class order mismatch in {checkpoint_path.name}: "
                f"{checkpoint.get('class_names')} != {CLASS_NAMES}"
            )
        model = DiseaseClassifier(pretrained=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        model.requires_grad_(False)
        model.to(device)
        models[seed] = model
        print(f"Loaded and froze seed {seed} Phase B checkpoint: {checkpoint_path.name}")
    return models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 13-view source-only reliability data (v2), without training models."
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="replace only outputs created by this v2 script; never changes split files/checkpoints")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"seed for reproducible perturbation recipes (default: {SEED})")
    args = parser.parse_args()

    make_folders()
    output_paths = [VIEWS_MANIFEST_FILE, SUMMARY_FILE, HASHES_FILE,
                    *FEATURE_FILES.values(), *PREDICTION_FILES.values()]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "V2 output from a previous run already exists. Nothing was changed.\n"
            "Use --overwrite only if you intentionally want to replace these v2 outputs:\n  " +
            "\n  ".join(existing)
        )

    assignments, source_split_hash = verify_original_split()
    reliability_split, secondary_split_hash = load_existing_secondary_split(assignments)

    ordered = reliability_split.sort_values("path").reset_index(drop=True)
    metadata_rows: list[dict] = []
    true_indices: list[int] = []
    features_by_seed: dict[int, list[np.ndarray]] = {seed: [] for seed in PHASE_B_CHECKPOINTS}
    logits_by_seed: dict[int, list[np.ndarray]] = {seed: [] for seed in PHASE_B_CHECKPOINTS}
    predictions_by_seed: dict[int, list[np.ndarray]] = {seed: [] for seed in PHASE_B_CHECKPOINTS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nExpanded source-only reliability generation (v2):")
    print(f"  Views per original: {VIEWS_PER_ORIGINAL}")
    print(f"  Perturbation replicates per type: {PERTURBATION_REPLICATES}")
    print(f"  Inference device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    models = load_frozen_models(device)

    with torch.inference_mode():
        for index, row in enumerate(ordered.itertuples(index=False), start=1):
            views, view_metadata = create_views(
                relative_path=row.path,
                label=row.label,
                group_id=row.group_id,
                reliability_split=row.reliability_split,
                root_seed=args.seed,
            )
            batch = torch.stack(views, dim=0).to(device)
            metadata_rows.extend(view_metadata)
            true_indices.extend([CLASS_TO_INDEX[row.label]] * len(views))

            for seed, model in models.items():
                feature_tensor = model.extract_features(batch)
                # Dropout is inactive in eval mode; these are the usual frozen logits.
                logits_tensor = model.fc(model.dropout(feature_tensor))
                if not bool(torch.isfinite(feature_tensor).all()):
                    raise RuntimeError(f"Non-finite features for image {row.path}, seed {seed}")
                if not bool(torch.isfinite(logits_tensor).all()):
                    raise RuntimeError(f"Non-finite logits for image {row.path}, seed {seed}")
                features_by_seed[seed].append(feature_tensor.cpu().numpy().astype(np.float32))
                logits_by_seed[seed].append(logits_tensor.cpu().numpy().astype(np.float32))
                predictions_by_seed[seed].append(logits_tensor.argmax(dim=1).cpu().numpy().astype(np.int64))

            if index % 25 == 0 or index == len(ordered):
                print(f"  Processed {index}/{len(ordered)} originals; "
                      f"{index * VIEWS_PER_ORIGINAL} views per classifier so far.")

    views_manifest = pd.DataFrame(metadata_rows)
    expected_views = len(ordered) * VIEWS_PER_ORIGINAL
    if len(views_manifest) != expected_views:
        raise RuntimeError(f"Expected {expected_views} view records, got {len(views_manifest)}")
    if views_manifest["view_id"].duplicated().any():
        raise RuntimeError("Generated duplicate view IDs; refusing to save outputs.")
    if views_manifest.groupby("path").size().nunique() != 1 or \
       not (views_manifest.groupby("path").size() == VIEWS_PER_ORIGINAL).all():
        raise RuntimeError("At least one original image does not have exactly the expected number of views.")

    expected_specs = [("original", 0)]
    for view_type in VIEW_TYPES[1:]:
        expected_specs.extend((view_type, replicate)
                              for replicate in range(1, PERTURBATION_REPLICATES + 1))
    for path, group in views_manifest.groupby("path", sort=False):
        observed_specs = list(zip(group["view_type"], group["replicate_index"].astype(int)))
        if observed_specs != expected_specs:
            raise RuntimeError(f"View types/order/count mismatch for {path}: {observed_specs}")
        if group["reliability_split"].nunique() != 1:
            raise RuntimeError(f"Views of original image {path} crossed train/validation portions.")

    # Assert the original/group never crosses the secondary split.
    original_assignments = views_manifest[["path", "group_id", "reliability_split"]].drop_duplicates()
    if original_assignments.groupby("group_id")["reliability_split"].nunique().max() != 1:
        raise RuntimeError("A duplicate group crosses reliability train/validation portions.")

    true_array = np.asarray(true_indices, dtype=np.int64)
    view_ids = views_manifest["view_id"].to_numpy(dtype="U20")
    paths_array = views_manifest["path"].to_numpy(dtype=str)
    group_ids_array = views_manifest["group_id"].to_numpy(dtype=str)
    split_array = views_manifest["reliability_split"].to_numpy(dtype=str)
    view_types_array = views_manifest["view_type"].to_numpy(dtype=str)
    replicate_array = views_manifest["replicate_index"].to_numpy(dtype=np.int64)

    prediction_frames: dict[int, pd.DataFrame] = {}
    npz_arrays: dict[int, dict[str, np.ndarray]] = {}
    report_lines = [
        "SOURCE-ONLY RELIABILITY DATA GENERATION SUMMARY (V2)",
        "=" * 62,
        "",
        f"Seed for per-view random recipes: {args.seed}",
        f"Original S-reliability images: {len(ordered)}",
        f"Views per original: {VIEWS_PER_ORIGINAL}",
        f"Generated view records: {len(views_manifest)}",
        "View recipe: 1 original + 3 brightness + 3 contrast + 3 gaussian_blur + 3 gaussian_noise",
        f"Brightness factor range: {BRIGHTNESS_RANGE}",
        f"Contrast factor range: {CONTRAST_RANGE}",
        f"Gaussian blur sigma range: {BLUR_SIGMA_RANGE}",
        f"Gaussian noise sigma range (normalized space): {NOISE_SIGMA_RANGE}",
        f"Original source split SHA-256: {source_split_hash}",
        f"Existing secondary split SHA-256: {secondary_split_hash}",
        "Existing secondary split was reused unchanged: YES",
        "Disease classifier training: NOT performed",
        "Failure-head training: NOT performed",
        "PP2020 target data: NOT read or used",
        "Original image and all its views stay in the same reliability portion: CHECKED",
        "Positive counts include correlated views and are not independent image counts.",
        "",
        "Secondary split by original image:",
    ]
    for label in CLASS_NAMES:
        class_rows = reliability_split[reliability_split["label"] == label]
        n_train = int((class_rows["reliability_split"] == "reliability_train").sum())
        n_val = int((class_rows["reliability_split"] == "reliability_val").sum())
        report_lines.append(f"  {label}: train={n_train}, validation={n_val}, total={len(class_rows)}")
    report_lines.append("")

    for seed in PHASE_B_CHECKPOINTS:
        feature_array = np.concatenate(features_by_seed[seed], axis=0)
        logits_array = np.concatenate(logits_by_seed[seed], axis=0)
        predicted_array = np.concatenate(predictions_by_seed[seed], axis=0)
        failure_array = (predicted_array != true_array).astype(np.int8)
        if feature_array.shape != (expected_views, 576):
            raise RuntimeError(f"Unexpected feature matrix shape for seed {seed}: {feature_array.shape}")
        if logits_array.shape != (expected_views, 3):
            raise RuntimeError(f"Unexpected logit matrix shape for seed {seed}: {logits_array.shape}")

        predictions = views_manifest.copy()
        predictions["true_label_index"] = true_array
        predictions["predicted_label_index"] = predicted_array
        predictions["predicted_label"] = [CLASS_NAMES[i] for i in predicted_array]
        predictions["failure_label"] = failure_array
        predictions["logit_healthy"] = logits_array[:, 0]
        predictions["logit_scab"] = logits_array[:, 1]
        predictions["logit_rust"] = logits_array[:, 2]
        predictions["classifier_seed"] = seed
        prediction_frames[seed] = predictions
        npz_arrays[seed] = {
            "features": feature_array,
            "logits": logits_array,
            "true_label_index": true_array,
            "predicted_label_index": predicted_array,
            "failure_label": failure_array,
            "view_ids": view_ids,
            "paths": paths_array,
            "group_ids": group_ids_array,
            "reliability_split": split_array,
            "view_types": view_types_array,
            "replicate_index": replicate_array,
            "classifier_seed": np.asarray(seed, dtype=np.int64),
        }

        report_lines.append(f"Seed {seed}:")
        for part in ("reliability_train", "reliability_val"):
            mask = predictions["reliability_split"].to_numpy() == part
            part_failures = failure_array[mask]
            positives = int(part_failures.sum())
            negatives = int(len(part_failures) - positives)
            unique_positive_originals = int(
                predictions.loc[mask & (failure_array == 1), "path"].nunique()
            )
            report_lines.append(
                f"  {part}: views={int(mask.sum())}, positive failures={positives}, "
                f"non-failures={negatives}, originals with >=1 failure={unique_positive_originals}"
            )
            if part == "reliability_train" and positives < MIN_POSITIVE_TRAIN_FAILURES:
                report_lines.append(
                    f"  WARNING: fewer than {MIN_POSITIVE_TRAIN_FAILURES} positive training views; "
                    "do not silently alter perturbation ranges or select views by outcome. "
                    "Revisit the protocol before head training."
                )
        report_lines.append("  Failure counts by view type (train / validation):")
        for view_type in VIEW_TYPES:
            counts = []
            for part in ("reliability_train", "reliability_val"):
                mask = ((predictions["view_type"].to_numpy() == view_type) &
                        (predictions["reliability_split"].to_numpy() == part))
                counts.append(f"{int(failure_array[mask].sum())}/{int(mask.sum())}")
            report_lines.append(f"    {view_type}: {counts[0]} / {counts[1]}")
        if seed != list(PHASE_B_CHECKPOINTS)[-1]:
            report_lines.append("")

    # Write only v2 outputs, and only after generation/inference/all checks succeed.
    views_manifest.to_csv(VIEWS_MANIFEST_FILE, index=False, lineterminator="\n")
    for seed in PHASE_B_CHECKPOINTS:
        np.savez_compressed(FEATURE_FILES[seed], **npz_arrays[seed])
        prediction_frames[seed].to_csv(PREDICTION_FILES[seed], index=False, lineterminator="\n")
    write_text(SUMMARY_FILE, "\n".join(report_lines))

    # Include both input and output checksums for traceability.
    hash_lines = [
        "SHA-256 checksums for reliability data-generation v2",
        "",
        "INPUT FILES (read-only):",
        f"{source_split_hash}  {SPLIT_ASSIGNMENTS.relative_to(PROJECT_ROOT).as_posix()}",
        f"{secondary_split_hash}  {SECONDARY_SPLIT_FILE.relative_to(PROJECT_ROOT).as_posix()}",
    ]
    for seed, path in PHASE_B_CHECKPOINTS.items():
        hash_lines.append(f"{sha256_file(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}")
    hash_lines += ["", "V2 OUTPUT FILES:"]
    for path in [VIEWS_MANIFEST_FILE, SUMMARY_FILE, *FEATURE_FILES.values(), *PREDICTION_FILES.values()]:
        hash_lines.append(f"{sha256_file(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}")
    write_text(HASHES_FILE, "\n".join(hash_lines))

    print("\nV2 generation complete. No classifier or failure head was trained.")
    print(f"Views per original: {VIEWS_PER_ORIGINAL}")
    print(f"View records:       {len(views_manifest)}")
    print(f"View manifest:      {VIEWS_MANIFEST_FILE.relative_to(PROJECT_ROOT)}")
    for seed in PHASE_B_CHECKPOINTS:
        print(f"Seed {seed}: features={FEATURE_FILES[seed].relative_to(PROJECT_ROOT)}; "
              f"predictions={PREDICTION_FILES[seed].relative_to(PROJECT_ROOT)}")
    print(f"Summary:            {SUMMARY_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Checksums:          {HASHES_FILE.relative_to(PROJECT_ROOT)}")
    print("\nRead reliability_generation_summary_v2.txt before any failure-head training.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
