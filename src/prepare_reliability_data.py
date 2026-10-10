"""Prepare source-only S-reliability data for failure-head development.

This script does NOT train a failure head and does NOT touch PP2020 data.
It:
  1. verifies the existing source split file against its SHA-256 sidecar;
  2. creates a reproducible, class-stratified 80/20 split of S-reliability,
     keeping each existing duplicate group intact;
  3. creates five views per original in memory: original, brightness, contrast,
     Gaussian blur, and Gaussian noise;
  4. runs the three frozen Phase B disease classifiers on the same views;
  5. saves view recipes, per-seed features/logits/predictions/error labels,
     counts, and SHA-256 records.

The generated view tensors are processed in memory rather than saved as image
files, to avoid writing hundreds of MB or more of redundant image data. The
CSV manifest records each sampled perturbation value and random seed, making
view generation reproducible with this script and the same software stack.

Run from the project root with the venv active:
    python src/prepare_reliability_data.py

If this script was run before and you intentionally want to replace its output:
    python src/prepare_reliability_data.py --overwrite
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
# Fixed, previously specified methodology settings
# ---------------------------------------------------------------------------
RELIABILITY_FRACTION = 0.80  # original-image groups assigned to head training
VALIDATION_FRACTION = 0.20   # original-image groups assigned to head validation
VIEW_TYPES = ("original", "brightness", "contrast", "gaussian_blur", "gaussian_noise")

BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)
BLUR_SIGMA_RANGE = (0.5, 1.5)
NOISE_SIGMA_RANGE = (0.01, 0.05)  # added after ImageNet normalization
MIN_POSITIVE_TRAIN_FAILURES = 50  # report only; does not silently change protocol

PHASE_B_CHECKPOINTS = {
    42: MODELS_DIR / "disease_seed42_phaseB.pt",
    123: MODELS_DIR / "disease_seed123_phaseB.pt",
    2026: MODELS_DIR / "disease_seed2026_phaseB.pt",
}

SPLIT_ASSIGNMENTS = SPLITS_DIR / "split_assignments.csv"
SPLIT_ASSIGNMENTS_SHA = SPLITS_DIR / "split_assignments.csv.sha256"

SECONDARY_SPLIT_FILE = SPLITS_DIR / "reliability_train_val.csv"
SECONDARY_SPLIT_SHA = SPLITS_DIR / "reliability_train_val.csv.sha256"
VIEWS_MANIFEST_FILE = MANIFEST_DIR / "reliability_views_manifest.csv"
SUMMARY_FILE = RESULTS_DIR / "reliability_generation_summary.txt"
HASHES_FILE = RESULTS_DIR / "reliability_generation_hashes.txt"

FEATURE_FILES = {
    seed: RESULTS_DIR / f"reliability_features_seed{seed}.npz"
    for seed in PHASE_B_CHECKPOINTS
}
PREDICTION_FILES = {
    seed: RESULTS_DIR / f"reliability_predictions_seed{seed}.csv"
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


def verify_existing_split() -> pd.DataFrame:
    """Validate the existing source split and its recorded hash."""
    if not SPLIT_ASSIGNMENTS.is_file():
        raise FileNotFoundError(
            f"{SPLIT_ASSIGNMENTS} not found. Do not regenerate the source split here."
        )
    if not SPLIT_ASSIGNMENTS_SHA.is_file():
        raise FileNotFoundError(
            f"{SPLIT_ASSIGNMENTS_SHA} not found; cannot verify the existing split."
        )

    expected_text = SPLIT_ASSIGNMENTS_SHA.read_text(encoding="utf-8").strip().split()
    if not expected_text:
        raise ValueError(f"The hash sidecar is empty: {SPLIT_ASSIGNMENTS_SHA}")
    expected_hash = expected_text[0].lower()
    actual_hash = sha256_file(SPLIT_ASSIGNMENTS)
    if actual_hash != expected_hash:
        raise ValueError(
            "The existing split-assignment CSV does not match its SHA-256 sidecar.\n"
            f"Expected: {expected_hash}\nActual:   {actual_hash}\n"
            "Stop here and investigate; do not regenerate or overwrite the split."
        )

    table = pd.read_csv(SPLIT_ASSIGNMENTS, keep_default_na=False)
    required = {"path", "label", "group_id", "subset"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Split assignment file is missing columns: {sorted(missing)}")
    if table["path"].duplicated().any():
        examples = table.loc[table["path"].duplicated(), "path"].head(5).tolist()
        raise ValueError(f"Duplicate paths found in split assignment file: {examples}")
    if not table["label"].isin(CLASS_NAMES).all():
        bad = sorted(set(table.loc[~table["label"].isin(CLASS_NAMES), "label"]))
        raise ValueError(f"Unexpected source labels in split assignments: {bad}")
    return table


def stable_seed(relative_path: str, view_type: str, root_seed: int) -> int:
    """Derive a stable per-image/per-view seed independent of processing order."""
    token = f"{root_seed}\0{relative_path}\0{view_type}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**63 - 1)


def largest_remainder_targets(class_counts: dict[str, int], fraction: float,
                              overall_n: int) -> dict[str, int]:
    """Class-wise targets whose sum matches the rounded overall target."""
    ideal = {name: class_counts[name] * fraction for name in CLASS_NAMES}
    targets = {name: math.floor(ideal[name]) for name in CLASS_NAMES}
    overall_target = int(round(overall_n * fraction))
    remaining = overall_target - sum(targets.values())
    order = sorted(
        CLASS_NAMES,
        key=lambda name: (-(ideal[name] - targets[name]), CLASS_NAMES.index(name)),
    )
    for name in order[:remaining]:
        targets[name] += 1
    return targets


def choose_groups_close_to_target(groups: list[tuple[str, int]], target: int) -> list[str]:
    """Choose whole groups with total image count closest to target.

    A subset-sum dynamic program keeps groups intact. In the current split all
    S-reliability groups are singletons, so the class targets can be met exactly.
    """
    # Group order is already seeded/shuffled by the caller. Retain the first
    # reproducible choice when more than one combination reaches a total.
    dp: dict[int, tuple[str, ...]] = {0: ()}
    for group_id, size in groups:
        previous = sorted(list(dp.items()), key=lambda item: item[0], reverse=True)
        for total, selected in previous:
            new_total = total + size
            if new_total not in dp:
                dp[new_total] = selected + (group_id,)
    best_total = min(dp, key=lambda total: (abs(total - target), total > target, total))
    return list(dp[best_total])


def make_secondary_split(assignments: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Make the 80/20 reliability split by original group, stratified by class."""
    reliability = assignments.loc[assignments["subset"] == "s_reliability"].copy()
    if reliability.empty:
        raise ValueError("No rows with subset='s_reliability' were found.")
    reliability = reliability.sort_values("path").reset_index(drop=True)

    # A group cannot be split, and this script does not silently resolve a group
    # whose members have different labels.
    label_counts_per_group = reliability.groupby("group_id")["label"].nunique()
    mixed_groups = label_counts_per_group[label_counts_per_group > 1].index.tolist()
    if mixed_groups:
        raise ValueError(
            "S-reliability contains duplicate groups with conflicting labels. "
            f"Review before splitting. Examples: {mixed_groups[:10]}"
        )

    class_counts = {
        label: int((reliability["label"] == label).sum()) for label in CLASS_NAMES
    }
    validation_targets = largest_remainder_targets(
        class_counts, VALIDATION_FRACTION, len(reliability)
    )

    group_table = (
        reliability.groupby("group_id", sort=True)
        .agg(label=("label", "first"), image_count=("path", "size"))
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    validation_groups: set[str] = set()
    class_split_actual: dict[str, int] = {}

    for label in CLASS_NAMES:
        class_groups = group_table.loc[group_table["label"] == label, ["group_id", "image_count"]]
        items = [(str(row.group_id), int(row.image_count))
                 for row in class_groups.itertuples(index=False)]
        if not items:
            class_split_actual[label] = 0
            continue
        order = rng.permutation(len(items)).tolist()
        shuffled_items = [items[i] for i in order]
        selected = choose_groups_close_to_target(shuffled_items, validation_targets[label])
        validation_groups.update(selected)
        class_split_actual[label] = sum(size for gid, size in items if gid in set(selected))

    output = reliability[["path", "label", "group_id", "subset"]].copy()
    output = output.rename(columns={"subset": "source_subset"})
    output["reliability_split"] = output["group_id"].map(
        lambda gid: "reliability_val" if gid in validation_groups else "reliability_train"
    )

    # Hard validation: no group is present in both secondary portions.
    group_spans = output.groupby("group_id")["reliability_split"].nunique()
    if not bool((group_spans == 1).all()):
        raise RuntimeError("A duplicate group crossed the reliability train/validation split.")
    if output["path"].nunique() != len(reliability):
        raise RuntimeError("An original image appears more than once in the reliability split.")
    if set(output["source_subset"]) != {"s_reliability"}:
        raise RuntimeError("Unexpected source subsets found in reliability split output.")

    print("Secondary S-reliability split:")
    print(f"  Original images: {len(output)}")
    print(f"  Validation target total: {int(round(len(output) * VALIDATION_FRACTION))}")
    print("  Actual counts by class (train / validation):")
    for label in CLASS_NAMES:
        train_n = int(((output["label"] == label) &
                       (output["reliability_split"] == "reliability_train")).sum())
        val_n = int(((output["label"] == label) &
                     (output["reliability_split"] == "reliability_val")).sum())
        print(f"    {label:<8} {train_n:>4} / {val_n:<4} (validation target {validation_targets[label]})")
    print(f"  Actual total: {sum(output['reliability_split'] == 'reliability_train')} train / "
          f"{sum(output['reliability_split'] == 'reliability_val')} validation")
    print(f"  Multi-image groups in S-reliability: "
          f"{int((group_table['image_count'] > 1).sum())}; kept intact.")
    return output


def normalize_pil(image) -> torch.Tensor:
    """Convert a resized PIL image to an ImageNet-normalized tensor."""
    return NORMALIZE(TO_TENSOR(image))


def create_views(relative_path: str, label: str, group_id: str,
                 reliability_split: str, root_seed: int):
    """Create five views and matching metadata for one original source image."""
    source_image = load_image(relative_path)
    resized = RESIZE(source_image)
    original_tensor = normalize_pil(resized)

    views: list[torch.Tensor] = [original_tensor]
    path_seed_key = relative_path.replace("\\", "/")

    brightness_seed = stable_seed(path_seed_key, "brightness", root_seed)
    brightness_rng = np.random.default_rng(brightness_seed)
    brightness_factor = float(brightness_rng.uniform(*BRIGHTNESS_RANGE))
    brightness_image = TF.adjust_brightness(resized, brightness_factor)
    views.append(normalize_pil(brightness_image))

    contrast_seed = stable_seed(path_seed_key, "contrast", root_seed)
    contrast_rng = np.random.default_rng(contrast_seed)
    contrast_factor = float(contrast_rng.uniform(*CONTRAST_RANGE))
    contrast_image = TF.adjust_contrast(resized, contrast_factor)
    views.append(normalize_pil(contrast_image))

    blur_seed = stable_seed(path_seed_key, "gaussian_blur", root_seed)
    blur_rng = np.random.default_rng(blur_seed)
    blur_sigma = float(blur_rng.uniform(*BLUR_SIGMA_RANGE))
    blur_image = resized.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    views.append(normalize_pil(blur_image))

    noise_seed = stable_seed(path_seed_key, "gaussian_noise", root_seed)
    noise_rng = np.random.default_rng(noise_seed)
    noise_sigma = float(noise_rng.uniform(*NOISE_SIGMA_RANGE))
    noise = noise_rng.normal(
        loc=0.0, scale=noise_sigma, size=tuple(original_tensor.shape)
    ).astype(np.float32)
    noisy_tensor = original_tensor + torch.from_numpy(noise)
    views.append(noisy_tensor)

    parameter_rows = [
        {"view_type": "original", "rng_seed": "", "brightness_factor": "",
         "contrast_factor": "", "blur_sigma": "", "noise_sigma": ""},
        {"view_type": "brightness", "rng_seed": brightness_seed,
         "brightness_factor": brightness_factor, "contrast_factor": "",
         "blur_sigma": "", "noise_sigma": ""},
        {"view_type": "contrast", "rng_seed": contrast_seed,
         "brightness_factor": "", "contrast_factor": contrast_factor,
         "blur_sigma": "", "noise_sigma": ""},
        {"view_type": "gaussian_blur", "rng_seed": blur_seed,
         "brightness_factor": "", "contrast_factor": "",
         "blur_sigma": blur_sigma, "noise_sigma": ""},
        {"view_type": "gaussian_noise", "rng_seed": noise_seed,
         "brightness_factor": "", "contrast_factor": "",
         "blur_sigma": "", "noise_sigma": noise_sigma},
    ]

    metadata = []
    for view_tensor, parameters in zip(views, parameter_rows):
        view_type = parameters["view_type"]
        view_id = hashlib.sha256(
            f"{path_seed_key}\0{view_type}".encode("utf-8")
        ).hexdigest()[:20]
        metadata.append({
            "view_id": view_id,
            "path": relative_path,
            "label": label,
            "label_index": CLASS_TO_INDEX[label],
            "group_id": group_id,
            "source_subset": "s_reliability",
            "reliability_split": reliability_split,
            "view_type": view_type,
            "rng_seed": parameters["rng_seed"],
            "brightness_factor": parameters["brightness_factor"],
            "contrast_factor": parameters["contrast_factor"],
            "blur_sigma": parameters["blur_sigma"],
            "noise_sigma": parameters["noise_sigma"],
            "view_generation_seed": root_seed,
        })
        if tuple(view_tensor.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Wrong tensor shape for {relative_path} ({view_type}): {view_tensor.shape}")
        if not bool(torch.isfinite(view_tensor).all()):
            raise RuntimeError(f"Non-finite values in view {view_type} for {relative_path}")

    return views, metadata


def load_frozen_models(device: torch.device) -> dict[int, DiseaseClassifier]:
    """Load and freeze the already-trained Phase B classifiers."""
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


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the source-only reliability split and generate view features/predictions."
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="replace outputs created by this script; never changes the original source split")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"fixed seed for the secondary split and view recipes (default: {SEED})")
    args = parser.parse_args()

    make_folders()
    output_paths = [SECONDARY_SPLIT_FILE, SECONDARY_SPLIT_SHA, VIEWS_MANIFEST_FILE,
                    SUMMARY_FILE, HASHES_FILE, *FEATURE_FILES.values(), *PREDICTION_FILES.values()]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output from a previous run already exists. Nothing was changed.\n" +
            "Use --overwrite only if you intentionally want to replace these outputs:\n  " +
            "\n  ".join(existing)
        )

    assignments = verify_existing_split()
    reliability_split = make_secondary_split(assignments, args.seed)

    # Build all output views once; these same tensors are sent to all three seeds.
    ordered = reliability_split.sort_values("path").reset_index(drop=True)
    metadata_rows: list[dict] = []
    true_indices: list[int] = []
    features_by_seed: dict[int, list[np.ndarray]] = {seed: [] for seed in PHASE_B_CHECKPOINTS}
    logits_by_seed: dict[int, list[np.ndarray]] = {seed: [] for seed in PHASE_B_CHECKPOINTS}
    predictions_by_seed: dict[int, list[np.ndarray]] = {seed: [] for seed in PHASE_B_CHECKPOINTS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInference device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
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
                # In eval mode dropout is disabled, so fc(features) is the same
                # disease-logit computation as model(batch), without a second backbone pass.
                logits_tensor = model.fc(model.dropout(feature_tensor))
                if not bool(torch.isfinite(feature_tensor).all()):
                    raise RuntimeError(f"Non-finite features for image {row.path}, seed {seed}")
                if not bool(torch.isfinite(logits_tensor).all()):
                    raise RuntimeError(f"Non-finite logits for image {row.path}, seed {seed}")
                features_by_seed[seed].append(feature_tensor.cpu().numpy().astype(np.float32))
                logits_by_seed[seed].append(logits_tensor.cpu().numpy().astype(np.float32))
                predictions_by_seed[seed].append(logits_tensor.argmax(dim=1).cpu().numpy().astype(np.int64))

            if index % 50 == 0 or index == len(ordered):
                print(f"Processed {index}/{len(ordered)} original S-reliability images "
                      f"({index * len(VIEW_TYPES)} views each processed so far).")

    views_manifest = pd.DataFrame(metadata_rows)
    expected_views = len(ordered) * len(VIEW_TYPES)
    if len(views_manifest) != expected_views:
        raise RuntimeError(f"Expected {expected_views} view records, got {len(views_manifest)}")
    views_per_image = views_manifest.groupby("path")["view_type"].agg(list)
    expected_types = list(VIEW_TYPES)
    for path, types in views_per_image.items():
        if types != expected_types:
            raise RuntimeError(f"View types/order mismatch for {path}: {types}")
    if views_manifest["view_id"].duplicated().any():
        raise RuntimeError("Generated duplicate view IDs; refusing to save outputs.")

    true_array = np.asarray(true_indices, dtype=np.int64)
    view_ids = views_manifest["view_id"].to_numpy(dtype="U20")
    predictions_frames: dict[int, pd.DataFrame] = {}
    npz_arrays: dict[int, dict[str, np.ndarray]] = {}
    report_lines = [
        "SOURCE-ONLY RELIABILITY DATA GENERATION SUMMARY",
        "=" * 58,
        "",
        f"Seed for secondary split and per-view random recipes: {args.seed}",
        f"Original S-reliability images: {len(ordered)}",
        f"Generated view records: {len(views_manifest)}",
        f"Views per original: {', '.join(VIEW_TYPES)}",
        "Disease classifier training: NOT performed",
        "Failure-head training: NOT performed",
        "PP2020 target data: NOT read or used",
        "",
        "Secondary split by class (original images):",
    ]
    for label in CLASS_NAMES:
        class_rows = reliability_split[reliability_split["label"] == label]
        n_train = int((class_rows["reliability_split"] == "reliability_train").sum())
        n_val = int((class_rows["reliability_split"] == "reliability_val").sum())
        report_lines.append(f"  {label}: train={n_train}, validation={n_val}, total={len(class_rows)}")
    report_lines += ["", "Failure labels are view-level: 1 = frozen classifier wrong; 0 = correct.",
                     "Positive counts include related views of the same original and are not independent images.", ""]

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
        predictions_frames[seed] = predictions
        npz_arrays[seed] = {
            "features": feature_array,
            "logits": logits_array,
            "true_label_index": true_array,
            "predicted_label_index": predicted_array,
            "failure_label": failure_array,
            "view_ids": view_ids,
            "classifier_seed": np.asarray(seed, dtype=np.int64),
        }

        report_lines.append(f"Seed {seed}:")
        for part in ("reliability_train", "reliability_val"):
            mask = (predictions["reliability_split"].to_numpy() == part)
            part_failures = failure_array[mask]
            positives = int(part_failures.sum())
            negatives = int(len(part_failures) - positives)
            unique_positive_originals = int(predictions.loc[mask & (failure_array == 1), "path"].nunique())
            report_lines.append(
                f"  {part}: views={int(mask.sum())}, positive failures={positives}, "
                f"non-failures={negatives}, originals with >=1 failure={unique_positive_originals}"
            )
            if part == "reliability_train" and positives < MIN_POSITIVE_TRAIN_FAILURES:
                report_lines.append(
                    f"  WARNING: fewer than {MIN_POSITIVE_TRAIN_FAILURES} positive training views; "
                    "do not silently change perturbation severity. Revisit the protocol before head training."
                )
        report_lines.append("  Positive failures by view type:")
        for view_type in VIEW_TYPES:
            view_mask = predictions["view_type"].to_numpy() == view_type
            positive_n = int(failure_array[view_mask].sum())
            report_lines.append(f"    {view_type}: {positive_n}/{int(view_mask.sum())}")
        report_lines.append("")

    # Save data only after all generation/inference/checks complete.
    reliability_split.to_csv(SECONDARY_SPLIT_FILE, index=False, lineterminator="\n")
    write_text(SECONDARY_SPLIT_SHA,
               f"{sha256_file(SECONDARY_SPLIT_FILE)}  {SECONDARY_SPLIT_FILE.name}")
    views_manifest.to_csv(VIEWS_MANIFEST_FILE, index=False, lineterminator="\n")

    for seed in PHASE_B_CHECKPOINTS:
        np.savez_compressed(FEATURE_FILES[seed], **npz_arrays[seed])
        predictions_frames[seed].to_csv(PREDICTION_FILES[seed], index=False, lineterminator="\n")

    write_text(SUMMARY_FILE, "\n".join(report_lines))

    files_to_hash = [SECONDARY_SPLIT_FILE, SECONDARY_SPLIT_SHA, VIEWS_MANIFEST_FILE,
                     SUMMARY_FILE, *FEATURE_FILES.values(), *PREDICTION_FILES.values()]
    hash_lines = ["SHA-256 checksums for reliability data-generation outputs", ""]
    for path in files_to_hash:
        hash_lines.append(f"{sha256_file(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}")
    write_text(HASHES_FILE, "\n".join(hash_lines))

    print("\nGeneration complete. No classifier or failure head was trained.")
    print(f"Secondary split: {SECONDARY_SPLIT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"View manifest:   {VIEWS_MANIFEST_FILE.relative_to(PROJECT_ROOT)}")
    for seed in PHASE_B_CHECKPOINTS:
        print(f"Seed {seed} features/predictions: {FEATURE_FILES[seed].relative_to(PROJECT_ROOT)}; "
              f"{PREDICTION_FILES[seed].relative_to(PROJECT_ROOT)}")
    print(f"Summary:         {SUMMARY_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Checksums:       {HASHES_FILE.relative_to(PROJECT_ROOT)}")
    print("\nRead reliability_generation_summary.txt before any failure-head training.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
