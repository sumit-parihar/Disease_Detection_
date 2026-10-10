"""Train the three source-only failure-prediction heads (Version 2).

This script uses only the saved Version 2 S-reliability features/logits and
view-level correctness labels. It does not load images, disease-classifier
checkpoints, S-test, S-calibration, or PP2020 data.

For each disease-classifier seed, a matching failure head is trained on that
seed's reliability_train views and selected by reliability_val average
precision (AUPRC-error). The disease classifier is not retrained or modified.

Run from the project root with the venv active:
    python src/train_failure_heads.py

Use --overwrite only to intentionally replace outputs created by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    FAILURE_DROPOUT,
    FAILURE_HIDDEN_UNITS,
    FAILURE_INPUT_DIM,
    FAILURE_L2,
    FAILURE_POS_WEIGHT_CAP,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    make_folders,
)

# Methodology-specified failure-head training settings.
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5
CLASSIFIER_SEEDS = (42, 123, 2026)
MIN_POSITIVE_TRAIN_FAILURES = 50  # already met in V2; checked again here

FEATURE_FILES = {
    seed: RESULTS_DIR / f"reliability_v2_features_seed{seed}.npz"
    for seed in CLASSIFIER_SEEDS
}
PREDICTION_FILES = {
    seed: RESULTS_DIR / f"reliability_v2_predictions_seed{seed}.csv"
    for seed in CLASSIFIER_SEEDS
}
CHECKPOINT_FILES = {
    seed: MODELS_DIR / f"failure_head_seed{seed}.pt"
    for seed in CLASSIFIER_SEEDS
}
HISTORY_FILES = {
    seed: RESULTS_DIR / f"failure_head_history_seed{seed}.csv"
    for seed in CLASSIFIER_SEEDS
}
VALIDATION_PREDICTION_FILES = {
    seed: RESULTS_DIR / f"failure_head_val_predictions_seed{seed}.csv"
    for seed in CLASSIFIER_SEEDS
}
SUMMARY_FILE = RESULTS_DIR / "failure_head_training_summary.txt"
HASHES_FILE = RESULTS_DIR / "failure_head_training_hashes.txt"

EXPECTED_ARRAY_KEYS = {
    "features", "logits", "true_label_index", "predicted_label_index",
    "failure_label", "view_ids", "paths", "group_ids",
    "reliability_split", "view_types", "replicate_index", "classifier_seed",
}
EXPECTED_VIEW_TYPES = {
    "original": 1,
    "brightness": 3,
    "contrast": 3,
    "gaussian_blur": 3,
    "gaussian_noise": 3,
}


class FailureHead(nn.Module):
    """579-D input -> Dense(64, ReLU) -> Dropout(0.30) -> one failure logit."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(FAILURE_INPUT_DIM, FAILURE_HIDDEN_UNITS)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(FAILURE_DROPOUT)
        self.output = nn.Linear(FAILURE_HIDDEN_UNITS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return logits for numerically stable BCEWithLogitsLoss. Applying
        # sigmoid to these logits gives the planned failure probability.
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.output(x).squeeze(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Improve repeatability for this small MLP. No new data or model selection
    # is performed by these settings.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required Version 2 input file(s) are missing:\n  " + "\n  ".join(missing)
        )


def load_and_validate(seed: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load an NPZ/CSV pair and verify that records align before training."""
    npz_path = FEATURE_FILES[seed]
    csv_path = PREDICTION_FILES[seed]
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing_keys = EXPECTED_ARRAY_KEYS - set(loaded.files)
        if missing_keys:
            raise ValueError(f"{npz_path.name} is missing arrays: {sorted(missing_keys)}")
        arrays = {key: loaded[key].copy() for key in EXPECTED_ARRAY_KEYS}

    table = pd.read_csv(csv_path, keep_default_na=False)
    required_columns = {
        "view_id", "path", "label", "label_index", "group_id",
        "source_subset", "reliability_split", "view_type", "replicate_index",
        "true_label_index", "predicted_label_index", "failure_label", "classifier_seed",
    }
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing_columns)}")

    n = len(table)
    for key, value in arrays.items():
        if value.ndim == 0:
            continue
        if len(value) != n:
            raise ValueError(
                f"Seed {seed}: NPZ array {key!r} has {len(value)} rows but CSV has {n}."
            )
    if arrays["features"].shape != (n, FAILURE_INPUT_DIM - 3):
        raise ValueError(f"Seed {seed}: unexpected feature shape {arrays['features'].shape}.")
    if arrays["logits"].shape != (n, 3):
        raise ValueError(f"Seed {seed}: unexpected logits shape {arrays['logits'].shape}.")
    if int(arrays["classifier_seed"].item()) != seed:
        raise ValueError(f"Seed mismatch inside {npz_path.name}.")

    # The CSV and NPZ must refer to the same ordered records; otherwise features
    # could accidentally be paired with another image's labels.
    aligned_pairs = (
        ("view_ids", "view_id"),
        ("paths", "path"),
        ("group_ids", "group_id"),
        ("reliability_split", "reliability_split"),
        ("view_types", "view_type"),
        ("replicate_index", "replicate_index"),
        ("true_label_index", "true_label_index"),
        ("predicted_label_index", "predicted_label_index"),
        ("failure_label", "failure_label"),
    )
    for array_name, column_name in aligned_pairs:
        left = arrays[array_name].astype(str)
        right = table[column_name].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(
                f"Seed {seed}: NPZ array {array_name!r} is not aligned with CSV column {column_name!r}."
            )

    if not table["classifier_seed"].astype(int).eq(seed).all():
        raise ValueError(f"Unexpected classifier_seed value in {csv_path.name}.")
    if not table["source_subset"].eq("s_reliability").all():
        raise ValueError(f"Non-S-reliability source row found in {csv_path.name}.")
    if set(table["reliability_split"].unique()) != {"reliability_train", "reliability_val"}:
        raise ValueError(f"Both reliability_train and reliability_val must be present for seed {seed}.")
    if table["view_id"].duplicated().any():
        raise ValueError(f"Duplicate view_id values found for seed {seed}.")
    if not set(table["failure_label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"Invalid failure labels found for seed {seed}.")
    if not np.array_equal(
        table["failure_label"].astype(int).to_numpy(),
        (table["predicted_label_index"].astype(int).to_numpy() !=
         table["true_label_index"].astype(int).to_numpy()).astype(int),
    ):
        raise ValueError(f"Failure labels do not match predicted vs true class for seed {seed}.")
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["logits"]).all():
        raise ValueError(f"Non-finite feature or logit values found for seed {seed}.")

    # Every original must have all 13 expected views, all in one portion.
    group_sizes = table.groupby("path", sort=False).size()
    if not group_sizes.eq(sum(EXPECTED_VIEW_TYPES.values())).all():
        raise ValueError(f"Not every original has exactly 13 views for seed {seed}.")
    per_path_split_counts = table.groupby("path")["reliability_split"].nunique()
    if not per_path_split_counts.eq(1).all():
        raise ValueError(f"Views from an original image cross reliability portions for seed {seed}.")
    per_path_group_counts = table.groupby("path")["group_id"].nunique()
    if not per_path_group_counts.eq(1).all():
        raise ValueError(f"An original image has inconsistent group IDs for seed {seed}.")
    per_path_types = table.groupby(["path", "view_type"]).size().unstack(fill_value=0)
    for view_type, expected_count in EXPECTED_VIEW_TYPES.items():
        if view_type not in per_path_types.columns or not per_path_types[view_type].eq(expected_count).all():
            raise ValueError(
                f"Original images do not each have {expected_count} {view_type!r} view(s) for seed {seed}."
            )
    # Duplicate groups (not just individual paths) must also remain in one part.
    if not table.groupby("group_id")["reliability_split"].nunique().eq(1).all():
        raise ValueError(f"A source group crosses reliability portions for seed {seed}.")

    return arrays, table


def make_data(arrays: dict[str, np.ndarray], table: pd.DataFrame):
    x_all = np.concatenate([arrays["features"], arrays["logits"]], axis=1).astype(np.float32)
    y_all = arrays["failure_label"].astype(np.float32)
    train_mask = table["reliability_split"].eq("reliability_train").to_numpy()
    val_mask = table["reliability_split"].eq("reliability_val").to_numpy()
    if np.any(train_mask & val_mask) or not np.all(train_mask | val_mask):
        raise ValueError("Rows do not belong exclusively to train or validation.")

    x_train = torch.from_numpy(x_all[train_mask])
    y_train = torch.from_numpy(y_all[train_mask])
    x_val = torch.from_numpy(x_all[val_mask])
    y_val = torch.from_numpy(y_all[val_mask])
    train_table = table.loc[train_mask].reset_index(drop=True)
    val_table = table.loc[val_mask].reset_index(drop=True)

    positives = int(y_train.sum().item())
    negatives = int(len(y_train) - positives)
    val_positives = int(y_val.sum().item())
    if positives < MIN_POSITIVE_TRAIN_FAILURES:
        raise ValueError(
            f"Only {positives} positive training views; the protocol minimum is "
            f"{MIN_POSITIVE_TRAIN_FAILURES}. Do not train until the protocol is reviewed."
        )
    if val_positives == 0:
        raise ValueError("Validation contains no positive failures; AUPRC-error cannot be selected reliably.")

    # Approved formula: compute from training examples only and cap at 5.
    positive_weight = min(float(FAILURE_POS_WEIGHT_CAP), negatives / positives)
    return (x_train, y_train, x_val, y_val, train_table, val_table,
            positives, negatives, val_positives, positive_weight)


def l2_hidden_kernel(model: FailureHead) -> torch.Tensor:
    """L2 penalty on the hidden Dense kernel only, as specified in the paper."""
    return FAILURE_L2 * model.hidden.weight.square().sum()


def validation_metrics(model: FailureHead, x_val: torch.Tensor,
                       y_val: torch.Tensor, criterion: nn.Module,
                       device: torch.device) -> tuple[float, float, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        x_device = x_val.to(device)
        y_device = y_val.to(device)
        logits = model(x_device)
        loss = float(criterion(logits, y_device).item())
        probabilities = torch.sigmoid(logits).cpu().numpy()
    ap = float(average_precision_score(y_val.numpy().astype(np.int64), probabilities))
    return loss, ap, probabilities


def train_one_seed(seed: int, device: torch.device, overwrite: bool) -> dict:
    arrays, table = load_and_validate(seed)
    (x_train, y_train, x_val, y_val, train_table, val_table,
     positives, negatives, val_positives, positive_weight) = make_data(arrays, table)

    # Reuse the corresponding, prespecified classifier seed for reproducible
    # head initialization and minibatch ordering.
    set_random_seeds(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
        generator=generator, num_workers=0, drop_last=False,
    )

    model = FailureHead().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_ap = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history_rows = []
    n_train_originals = int(train_table["path"].nunique())
    n_val_originals = int(val_table["path"].nunique())
    n_train_positive_originals = int(
        train_table.loc[train_table["failure_label"].astype(int).eq(1), "path"].nunique()
    )
    n_val_positive_originals = int(
        val_table.loc[val_table["failure_label"].astype(int).eq(1), "path"].nunique()
    )

    print(f"\nSeed {seed}: training failure head")
    print(f"  Train views: {len(x_train)} ({positives} failures, {negatives} non-failures)")
    print(f"  Train originals with >=1 failure: {n_train_positive_originals}/{n_train_originals}")
    print(f"  Validation views: {len(x_val)} ({val_positives} failures)")
    print(f"  Validation originals with >=1 failure: {n_val_positive_originals}/{n_val_originals}")
    print(f"  Positive-class weight: min({FAILURE_POS_WEIGHT_CAP}, {negatives}/{positives}) = {positive_weight:.6g}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y) + l2_hidden_kernel(model)
            loss.backward()
            optimizer.step()
            count = len(batch_x)
            total_loss += float(loss.detach().item()) * count
            seen += count

        train_loss = total_loss / max(seen, 1)
        val_loss, val_ap, val_probabilities = validation_metrics(model, x_val, y_val, criterion, device)
        improved = val_ap > best_ap  # strict improvement; ties keep the earliest checkpoint
        if improved:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_rows.append({
            "epoch": epoch,
            "train_weighted_bce_plus_hidden_l2": train_loss,
            "validation_weighted_bce": val_loss,
            "validation_auprc_error": val_ap,
            "validation_positive_views": val_positives,
            "validation_positive_originals": n_val_positive_originals,
            "selected_best_so_far": bool(improved),
            "epochs_without_auprc_improvement": epochs_without_improvement,
        })
        print(
            f"  Epoch {epoch:02d}/{MAX_EPOCHS}: train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, val_AUPRC_error={val_ap:.6f}"
            f"{'  [best]' if improved else ''}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping: no strict validation AUPRC improvement for {PATIENCE} epochs.")
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint was selected for seed {seed}.")

    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        best_val_logits = model(x_val.to(device))
        best_val_probabilities = torch.sigmoid(best_val_logits).cpu().numpy()
    best_val_ap = float(average_precision_score(y_val.numpy().astype(np.int64), best_val_probabilities))

    train_hash = sha256_file(FEATURE_FILES[seed])
    prediction_hash = sha256_file(PREDICTION_FILES[seed])
    checkpoint_payload = {
        "state_dict": best_state,
        "classifier_seed": seed,
        "failure_head_seed": seed,
        "best_epoch": best_epoch,
        "best_validation_auprc_error": best_val_ap,
        "positive_weight": positive_weight,
        "architecture": {
            "input_dim": FAILURE_INPUT_DIM,
            "hidden_units": FAILURE_HIDDEN_UNITS,
            "hidden_activation": "ReLU",
            "dropout": FAILURE_DROPOUT,
            "output": "single logit; sigmoid applied for failure probability",
            "hidden_kernel_l2": FAILURE_L2,
        },
        "training_config": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "selection_metric": "validation AUPRC-error",
            "positive_weight_formula": "min(FAILURE_POS_WEIGHT_CAP, train_negatives / train_positives)",
            "seed": seed,
        },
        "data": {
            "feature_file": FEATURE_FILES[seed].name,
            "prediction_file": PREDICTION_FILES[seed].name,
            "feature_file_sha256": train_hash,
            "prediction_file_sha256": prediction_hash,
            "train_views": len(x_train),
            "train_positive_views": positives,
            "train_negative_views": negatives,
            "train_originals": n_train_originals,
            "train_positive_originals": n_train_positive_originals,
            "validation_views": len(x_val),
            "validation_positive_views": val_positives,
            "validation_originals": n_val_originals,
            "validation_positive_originals": n_val_positive_originals,
        },
    }
    torch.save(checkpoint_payload, CHECKPOINT_FILES[seed])
    pd.DataFrame(history_rows).to_csv(HISTORY_FILES[seed], index=False, lineterminator="\n")

    validation_output = val_table[[
        "view_id", "path", "group_id", "label", "reliability_split",
        "view_type", "replicate_index", "true_label_index", "predicted_label_index", "failure_label",
    ]].copy()
    validation_output["predicted_failure_probability"] = best_val_probabilities
    validation_output["classifier_seed"] = seed
    validation_output.to_csv(VALIDATION_PREDICTION_FILES[seed], index=False, lineterminator="\n")

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history_rows),
        "best_validation_auprc_error": best_val_ap,
        "positive_weight": positive_weight,
        "train_views": len(x_train),
        "train_positive_views": positives,
        "train_negative_views": negatives,
        "train_originals": n_train_originals,
        "train_positive_originals": n_train_positive_originals,
        "validation_views": len(x_val),
        "validation_positive_views": val_positives,
        "validation_originals": n_val_originals,
        "validation_positive_originals": n_val_positive_originals,
        "checkpoint": str(CHECKPOINT_FILES[seed]),
        "history": str(HISTORY_FILES[seed]),
        "validation_predictions": str(VALIDATION_PREDICTION_FILES[seed]),
        "feature_file_sha256": train_hash,
        "prediction_file_sha256": prediction_hash,
    }
    print(f"  Selected epoch: {best_epoch}; best validation AUPRC-error: {best_val_ap:.6f}")
    print(f"  Saved checkpoint: {CHECKPOINT_FILES[seed]}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace failure-head outputs from this script (does not alter disease classifiers or input data)",
    )
    args = parser.parse_args()

    make_folders()
    all_inputs = [path for seed in CLASSIFIER_SEEDS for path in (FEATURE_FILES[seed], PREDICTION_FILES[seed])]
    require_files(all_inputs)
    all_outputs = (
        list(CHECKPOINT_FILES.values()) + list(HISTORY_FILES.values()) +
        list(VALIDATION_PREDICTION_FILES.values()) + [SUMMARY_FILE, HASHES_FILE]
    )
    existing_outputs = [path for path in all_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Some failure-head output files already exist; refusing to overwrite them. "
            "Use --overwrite only if you intentionally want to replace these outputs:\n  " +
            "\n  ".join(str(path) for path in existing_outputs)
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("SOURCE-ONLY FAILURE-HEAD TRAINING")
    print("==================================")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Disease classifiers: frozen; no disease-classifier training will occur.")
    print("Target data: not loaded or used.")
    print(f"Architecture: {FAILURE_INPUT_DIM} -> {FAILURE_HIDDEN_UNITS} -> 1")
    print(f"Settings: Adam lr={LEARNING_RATE}, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}")

    summary_lines = [
        "SOURCE-ONLY FAILURE-HEAD TRAINING SUMMARY",
        "=" * 58,
        f"Device: {device}",
        "Disease classifier training: NOT performed",
        "S-test: NOT loaded or used",
        "S-calibration: NOT loaded or used",
        "PP2020 target data: NOT loaded or used",
        "Failure-head architecture: Dense(64, ReLU) -> Dropout(0.30) -> one failure logit",
        f"Input: {FAILURE_INPUT_DIM} values (576 features + 3 raw logits)",
        f"Loss: BCEWithLogitsLoss with positive weight min({FAILURE_POS_WEIGHT_CAP}, train negatives / train positives)",
        f"Hidden-kernel L2 coefficient: {FAILURE_L2}",
        f"Optimizer: Adam; learning rate={LEARNING_RATE}; batch size={BATCH_SIZE}",
        f"Maximum epochs={MAX_EPOCHS}; early-stopping patience={PATIENCE}",
        "Checkpoint selection: strict improvement in validation AUPRC-error; ties retain the earliest best epoch.",
        "Validation AUPRC is view-level and may be unstable because views of an original image are correlated.",
        "",
    ]
    hash_lines = ["FAILURE-HEAD TRAINING INPUT/OUTPUT SHA-256", "=" * 58]
    for seed in CLASSIFIER_SEEDS:
        for path in (FEATURE_FILES[seed], PREDICTION_FILES[seed]):
            hash_lines.append(f"INPUT {sha256_file(path)}  {path.as_posix()}")

    results = []
    try:
        for seed in CLASSIFIER_SEEDS:
            result = train_one_seed(seed, device, args.overwrite)
            results.append(result)
            summary_lines += [
                f"Seed {seed}:",
                f"  Selected epoch: {result['best_epoch']} / {result['epochs_run']} epochs run",
                f"  Best validation AUPRC-error: {result['best_validation_auprc_error']:.8f}",
                f"  Training views: {result['train_views']}; positive={result['train_positive_views']}; negative={result['train_negative_views']}",
                f"  Training originals: {result['train_originals']}; originals with >=1 failure={result['train_positive_originals']}",
                f"  Validation views: {result['validation_views']}; positive={result['validation_positive_views']}",
                f"  Validation originals: {result['validation_originals']}; originals with >=1 failure={result['validation_positive_originals']}",
                f"  Positive-class weight: {result['positive_weight']:.8g}",
                f"  Checkpoint: {result['checkpoint']}",
                f"  History: {result['history']}",
                f"  Validation predictions: {result['validation_predictions']}",
                "",
            ]
            hash_lines.append(f"OUTPUT {sha256_file(CHECKPOINT_FILES[seed])}  {CHECKPOINT_FILES[seed].as_posix()}")
            hash_lines.append(f"OUTPUT {sha256_file(HISTORY_FILES[seed])}  {HISTORY_FILES[seed].as_posix()}")
            hash_lines.append(f"OUTPUT {sha256_file(VALIDATION_PREDICTION_FILES[seed])}  {VALIDATION_PREDICTION_FILES[seed].as_posix()}")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Training stopped. Review the error before attempting to resume.", file=sys.stderr)
        return 1

    SUMMARY_FILE.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
    HASHES_FILE.write_text("\n".join(hash_lines).rstrip() + "\n", encoding="utf-8")
    print("\nAll three failure heads were trained and their best source-validation checkpoints saved.")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Hashes:  {HASHES_FILE}")
    print("Review the summary and validation histories before proceeding to baselines or S-test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
