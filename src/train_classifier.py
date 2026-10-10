"""
train_classifier.py  -  Step 5b: trains the main model (disease classifier) in
the two rounds of the paper draft v2, Section 3.10.

WHAT THIS FILE DOES
-------------------
It teaches the model to say healthy / scab / rust, using ONLY the source images:
  * s_train : the images the model learns from
  * s_val   : images used to check progress, to pick the best version and to
              decide when to stop
It never opens s_reliability, s_calibration, s_test or any PP2020 image.

ROUND 1  "Phase A" (head warm-up)
    Only the 3-output layer learns (the MobileNet part is locked).
    Adam, learning rate 1e-3, at most 12 passes (epochs) over s_train.
    The best version = the one with the lowest validation loss.
    Training stops early if the validation loss has not improved for 3 epochs.
ROUND 2  "Phase B" (fine-tuning)
    The last part of MobileNet and the 3-output layer learn, gently.
    AdamW, learning rate 3e-5, weight decay 1e-5, at most 20 epochs.
    The best version = the one with the highest validation macro-F1
    (the average of the F1 scores of the three classes, so rust counts as much
    as healthy). Early stop after 5 epochs without improvement.
    Phase B starts from the best model of phase A.

LOSS
----
Class-weighted cross-entropy. Rust has far fewer images than healthy, so a
mistake on a rust image counts more. Weight of class c = N / (3 x n_c), where N
is the number of s_train images and n_c the number of s_train images of class c.
The weights are computed from s_train only.

WHAT IT SAVES
-------------
  models/disease_seed<S>_phaseA.pt          best model after round 1
  models/disease_seed<S>_phaseB.pt          best model after round 2 (the final classifier)
  results/disease_seed<S>_training_log.csv  one row per epoch
  results/disease_seed<S>_run_record.json   seed, settings, library versions,
                                            data hashes, parameter counts, results
(models/ is ignored by Git; results/ is kept.)
Existing files are never overwritten unless you add --overwrite.

HOW TO RUN (project root, venv active, GPU PyTorch installed):
    python src/train_classifier.py                  both rounds, seed 42
    python src/train_classifier.py --seed 43        another seed
    python src/train_classifier.py --phase A        round 1 only
    python src/train_classifier.py --phase B        round 2 only (needs the phase A file)
The first run downloads the ImageNet weights (about 10 MB), so it needs internet.
"""

import argparse
import copy
import json
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EVAL_BATCH_SIZE,
    MANIFEST_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    PHASE_A_LR,
    PHASE_A_MAX_EPOCHS,
    PHASE_A_PATIENCE,
    PHASE_B_LR,
    PHASE_B_MAX_EPOCHS,
    PHASE_B_PATIENCE,
    PHASE_B_WEIGHT_DECAY,
    RESULTS_DIR,
    SEED,
    SPLITS_DIR,
    make_folders,
)
from dataset import SourceDataset, make_loader
from model import DiseaseClassifier, batchnorm_statistics, count_parameters, parameter_report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    """Fix every random number generator so the run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def class_weights_from_labels(labels, device):
    """w_c = N / (C x n_c), computed from the s_train labels only."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights = len(labels) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


def first_word(path):
    """The hash stored in a .sha256 sidecar file, or 'not found'."""
    return path.read_text().split()[0] if path.exists() else "not found"


def train_one_epoch(model, loader, optimizer, weights, device):
    """One pass over s_train. Returns the weighted average training loss."""
    model.train()
    loss_sum, weight_sum = 0.0, 0.0
    for images, labels, _ in tqdm(loader, leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        batch_loss = nn.functional.cross_entropy(logits, labels, weight=weights, reduction="sum")
        batch_weight = weights[labels].sum()
        (batch_loss / batch_weight).backward()
        optimizer.step()
        loss_sum += batch_loss.item()
        weight_sum += batch_weight.item()
    return loss_sum / weight_sum


@torch.no_grad()
def evaluate(model, loader, weights, device):
    """Weighted loss, accuracy and macro-F1 on a loader (here: s_val)."""
    model.eval()
    loss_sum, weight_sum = 0.0, 0.0
    all_labels, all_predictions = [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, weight=weights, reduction="sum").item()
        weight_sum += weights[labels].sum().item()
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
    labels_array, predictions_array = np.array(all_labels), np.array(all_predictions)
    return {
        "loss": loss_sum / weight_sum,
        "accuracy": float((labels_array == predictions_array).mean()),
        "macro_f1": float(f1_score(labels_array, predictions_array, average="macro",
                                   labels=list(range(NUM_CLASSES)), zero_division=0)),
        "labels": labels_array,
        "predictions": predictions_array,
    }


def cpu_copy(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def fit_phase(name, model, optimizer, train_loader, val_loader, weights, device,
              max_epochs, patience, monitor, log_rows):
    """Train one phase with early stopping. monitor is 'loss' (lower is better)
    or 'macro_f1' (higher is better). Puts the BEST version back into the model."""
    lower_is_better = monitor == "loss"
    best_value = float("inf") if lower_is_better else -float("inf")
    best_state, best_epoch, epochs_without_improvement = None, 0, 0
    best_metrics = None
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, weights, device)
        metrics = evaluate(model, val_loader, weights, device)
        value = metrics[monitor]
        improved = value < best_value if lower_is_better else value > best_value
        if improved:
            best_value, best_epoch, best_metrics = value, epoch, metrics
            best_state = cpu_copy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_rows.append({
            "phase": name, "epoch": epoch, "train_loss": round(train_loss, 5),
            "val_loss": round(metrics["loss"], 5), "val_accuracy": round(metrics["accuracy"], 5),
            "val_macro_f1": round(metrics["macro_f1"], 5), "best_so_far": improved,
            "seconds": round(time.time() - epoch_start, 1),
        })
        print(f"  [{name}] epoch {epoch:>2}/{max_epochs}  train loss {train_loss:.4f}  "
              f"val loss {metrics['loss']:.4f}  val acc {metrics['accuracy']:.3f}  "
              f"val macro-F1 {metrics['macro_f1']:.3f}  {'<- best' if improved else ''}")
        if epochs_without_improvement >= patience:
            print(f"  [{name}] stopping early: no improvement of val {monitor} for {patience} epochs")
            break

    model.load_state_dict(best_state)
    print(f"  [{name}] best epoch: {best_epoch} (val {monitor} {best_value:.4f})")
    return {
        "best_epoch": best_epoch, "monitor": f"val_{monitor}", "best_value": best_value,
        "epochs_run": epoch, "seconds": round(time.time() - started, 1),
        "val_accuracy": best_metrics["accuracy"], "val_macro_f1": best_metrics["macro_f1"],
        "val_loss": best_metrics["loss"],
    }, best_metrics


def save_checkpoint(path, model, seed, phase, result, weights_list):
    torch.save({"state_dict": cpu_copy(model.state_dict()), "seed": seed, "phase": phase,
                "result": result, "class_weights": weights_list, "class_names": CLASS_NAMES}, path)


def print_confusion(metrics):
    matrix = confusion_matrix(metrics["labels"], metrics["predictions"], labels=list(range(NUM_CLASSES)))
    print("  Confusion matrix on s_val (rows = true class, columns = predicted):")
    print("  " + " " * 10 + "".join(f"{name:>10}" for name in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, matrix):
        print("  " + f"{name:<10}" + "".join(f"{value:>10}" for value in row))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--phase", choices=["A", "B", "both"], default="both")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="image loading processes (0 is safest on Windows)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random start instead of ImageNet weights (tests only)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    make_folders()
    path_a = MODELS_DIR / f"disease_seed{args.seed}_phaseA.pt"
    path_b = MODELS_DIR / f"disease_seed{args.seed}_phaseB.pt"
    log_path = RESULTS_DIR / f"disease_seed{args.seed}_training_log.csv"
    record_path = RESULTS_DIR / f"disease_seed{args.seed}_run_record.json"

    run_a = args.phase in ("A", "both")
    run_b = args.phase in ("B", "both")
    for run, path in ((run_a, path_a), (run_b, path_b)):
        if run and path.exists() and not args.overwrite:
            sys.exit(f"{path} already exists. Use --overwrite only if you really want to train again.")
    if args.phase == "B" and not path_a.exists():
        sys.exit(f"{path_a} not found. Run phase A first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found, training on the CPU will be slow.\n")
    set_seed(args.seed)

    train_set = SourceDataset("s_train", augment=True)
    val_set = SourceDataset("s_val")
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True,
                               num_workers=args.num_workers, seed=args.seed)
    val_loader = make_loader(val_set, EVAL_BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
    weights, counts = class_weights_from_labels(train_set.labels, device)
    weights_list = [round(float(w), 6) for w in weights.cpu()]

    print(f"Seed {args.seed}, device {device}, batch size {BATCH_SIZE}")
    print(f"s_train: {len(train_set)} images {dict(zip(CLASS_NAMES, counts.astype(int).tolist()))}")
    print(f"s_val  : {len(val_set)} images")
    print(f"Class weights (healthy, scab, rust): {weights_list}\n")

    model = DiseaseClassifier(pretrained=not args.no_pretrained).to(device)
    backbone_params, head_params, classifier_params = parameter_report(model)
    print(f"Parameters: MobileNet part {backbone_params:,} + output layer {head_params:,} "
          f"= {classifier_params:,}\n")

    log_rows, results = [], {}
    final_metrics = None

    if run_a:
        print("ROUND 1 (phase A): only the 3-output layer learns")
        model.set_phase_a()
        _, trainable = count_parameters(model)
        print(f"  parameters that can learn: {trainable:,}")
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=PHASE_A_LR)
        results["phase_a"], final_metrics = fit_phase(
            "A", model, optimizer, train_loader, val_loader, weights, device,
            PHASE_A_MAX_EPOCHS, PHASE_A_PATIENCE, "loss", log_rows)
        results["phase_a"]["trainable_parameters"] = trainable
        save_checkpoint(path_a, model, args.seed, "A", results["phase_a"], weights_list)
        print(f"  saved {path_a.name}\n")
    else:
        model.load_state_dict(torch.load(path_a, map_location="cpu")["state_dict"])
        print(f"Loaded {path_a.name}\n")

    if run_b:
        print("ROUND 2 (phase B): last part of MobileNet + output layer learn, gently")
        model.set_phase_b()
        _, trainable = count_parameters(model)
        print(f"  parameters that can learn: {trainable:,} (BatchNorm frozen)")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=PHASE_B_LR, weight_decay=PHASE_B_WEIGHT_DECAY)
        results["phase_b"], final_metrics = fit_phase(
            "B", model, optimizer, train_loader, val_loader, weights, device,
            PHASE_B_MAX_EPOCHS, PHASE_B_PATIENCE, "macro_f1", log_rows)
        results["phase_b"]["trainable_parameters"] = trainable
        save_checkpoint(path_b, model, args.seed, "B", results["phase_b"], weights_list)
        print(f"  saved {path_b.name}\n")

    if final_metrics is not None:
        print_confusion(final_metrics)

    # --- records -------------------------------------------------------------
    if log_rows:
        new_log = pd.DataFrame(log_rows)
        if log_path.exists() and not (run_a and run_b) and not args.overwrite:
            new_log = pd.concat([pd.read_csv(log_path), new_log], ignore_index=True)
        new_log.to_csv(log_path, index=False, lineterminator="\n")

    record = {
        "seed": args.seed, "phase_run": args.phase, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        if device.type == "cuda" else None,
        "pretrained_imagenet_weights": not args.no_pretrained,
        "batch_size": BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
        "class_weights": dict(zip(CLASS_NAMES, weights_list)),
        "train_images": len(train_set), "val_images": len(val_set),
        "parameters": {"mobilenet_part": backbone_params, "output_layer": head_params,
                       "classifier_total": classifier_params,
                       "batchnorm_running_statistics": batchnorm_statistics(model)},
        "manifest_sha256": first_word(MANIFEST_DIR / "manifest.csv.sha256"),
        "split_sha256": first_word(SPLITS_DIR / "split_assignments.csv.sha256"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "results": results,
    }
    if record_path.exists() and args.phase != "both" and not args.overwrite:
        old = json.loads(record_path.read_text())
        old["results"].update(results)
        old["phase_run"] = "A+B"
        record = {**record, "results": old["results"], "phase_run": old["phase_run"]}
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Log   : {log_path}")
    print(f"Record: {record_path}")
    print("Only s_train and s_val were used. No other subset and no PP2020 image was opened.")


if __name__ == "__main__":
    main()