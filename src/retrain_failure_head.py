"""
retrain_failure_head.py  -  Retrains the failure head with a corrected set-up.
The disease classifiers are NOT retrained, and the old failure heads and their
result files are NOT touched. Everything new is saved under new names
("failure_head_scaled_...").

WHY THIS SCRIPT EXISTS
----------------------
failure_head_diagnostic.py showed two problems with the first failure heads:
  1. The 579 input numbers were fed in raw. Numbers of very different sizes made
     the head learn almost nothing useful (a head on raw inputs ranked the
     classifier's mistakes about as well as guessing).
  2. Early stopping on validation AUPRC picked epoch 1 for two seeds, when the
     head had barely started to learn. The validation set holds only 5 to 8
     failing original images, so that score is too noisy to choose an epoch.

WHAT CHANGES (the two deviations from the original protocol)
------------------------------------------------------------
  Change 1  Inputs are standardised: for every one of the 579 input numbers the
            mean is subtracted and the result divided by the standard deviation.
            Mean and standard deviation come from the reliability-TRAIN views
            only. Both are saved with the model, because the same numbers must be
            used later on any other image.
  Change 2  Training runs a FIXED 30 epochs and the model after the last epoch is
            kept. No epoch is chosen with the validation set. The validation
            numbers are recorded for monitoring only.
These two changes must be listed in the paper as deviations, with the reasons above.

WHAT STAYS THE SAME
-------------------
Design of the head (579 -> Dense 64 + ReLU -> Dropout 0.30 -> 1 output), loss
(weighted binary cross-entropy, positive weight capped at 5), L2 penalty on the
hidden layer, Adam with learning rate 5e-4, batch size 64, the same seeds, and
the same data files (results/reliability_v2_*).

WHAT IT NEVER TOUCHES
---------------------
No target (PP2020) image or label, no S-test, S-calibration, S-train or S-val
image. Only the S-reliability views that already exist are read.

OUTPUT FILES (new names; existing files are never overwritten)
--------------------------------------------------------------
  models/failure_head_scaled_seed{seed}.pt            weights + saved mean/std
  results/failure_head_scaled_history_seed{seed}.csv  numbers per epoch (monitoring)
  results/failure_head_scaled_val_predictions_seed{seed}.csv   score of each validation view
  results/failure_head_scaled_summary.txt             readable summary
  results/failure_head_scaled_hashes.txt              SHA-256 of inputs and outputs

HOW TO RUN (project root, venv active):
    python src/retrain_failure_head.py
"""

import argparse
import hashlib
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import FAILURE_POS_WEIGHT_CAP, MODELS_DIR, RESULTS_DIR, make_folders
from failure_head_diagnostic import evaluate, untrained_scores
from train_failure_head import (
    BATCH_SIZE,
    CLASSIFIER_SEEDS,
    FEATURE_FILES,
    LEARNING_RATE,
    PREDICTION_FILES,
    FailureHead,
    l2_hidden_kernel,
    load_and_validate,
    require_files,
    set_random_seeds,
)

FIXED_EPOCHS = 30                      # same number as the original maximum
BOOTSTRAP_SEED = 12345                 # only for the interval in the summary

CHECKPOINTS = {s: MODELS_DIR / f"failure_head_scaled_seed{s}.pt" for s in CLASSIFIER_SEEDS}
HISTORIES = {s: RESULTS_DIR / f"failure_head_scaled_history_seed{s}.csv" for s in CLASSIFIER_SEEDS}
VAL_PREDICTIONS = {s: RESULTS_DIR / f"failure_head_scaled_val_predictions_seed{s}.csv" for s in CLASSIFIER_SEEDS}
SUMMARY_FILE = RESULTS_DIR / "failure_head_scaled_summary.txt"
HASHES_FILE = RESULTS_DIR / "failure_head_scaled_hashes.txt"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_one_seed(seed, device):
    arrays, table = load_and_validate(seed)           # same safety checks as the first training
    train_mask = table["reliability_split"].eq("reliability_train").to_numpy()
    val_mask = ~train_mask

    x_all = np.concatenate([arrays["features"], arrays["logits"]], axis=1).astype(np.float32)
    y_all = arrays["failure_label"].astype(np.float32)
    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]

    # Change 1: standardise with reliability-TRAIN statistics only.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)     # a constant input keeps scale 1
    x_train_scaled = ((x_train - mean) / std).astype(np.float32)
    x_val_scaled = ((x_val - mean) / std).astype(np.float32)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    positive_weight = min(float(FAILURE_POS_WEIGHT_CAP), negatives / positives)

    set_random_seeds(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train_scaled), torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    model = FailureHead().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    x_val_tensor = torch.from_numpy(x_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)
    y_val_int = y_val.astype(np.int64)

    print(f"\nSeed {seed}: train views {len(x_train)} ({positives} failures), "
          f"validation views {len(x_val)} ({int(y_val.sum())} failures), "
          f"positive weight {positive_weight:.3g}, head parameters {parameter_count:,}")
    history = []
    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        total, seen = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y) + l2_hidden_kernel(model)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item()) * len(batch_x)
            seen += len(batch_x)
        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
        history.append({
            "epoch": epoch,
            "train_loss": total / seen,
            "validation_loss": val_loss,
            "validation_auprc_error": float(average_precision_score(y_val_int, val_scores)),
            "validation_auroc_error": float(roc_auc_score(y_val_int, val_scores)),
        })
    print(f"  finished {FIXED_EPOCHS} epochs; kept the last epoch. "
          f"Validation AUPRC {history[-1]['validation_auprc_error']:.3f}, "
          f"AUROC {history[-1]['validation_auroc_error']:.3f} (monitoring only)")

    # ---- save everything under new names ---------------------------------
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_mean": torch.from_numpy(mean), "input_std": torch.from_numpy(std),
        "classifier_seed": seed, "epochs": FIXED_EPOCHS, "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "positive_weight": positive_weight, "parameter_count": parameter_count,
        "recipe": "inputs standardised with reliability-train statistics; fixed 30 epochs; last epoch kept",
        "torch_version": str(torch.__version__),
    }, CHECKPOINTS[seed])
    pd.DataFrame(history).to_csv(HISTORIES[seed], index=False, lineterminator="\n")
    val_table = table.loc[val_mask, ["view_id", "path", "view_type", "failure_label"]].reset_index(drop=True)
    val_table["failure_score"] = val_scores
    val_table.to_csv(VAL_PREDICTIONS[seed], index=False, lineterminator="\n")

    # ---- numbers for the summary (head versus MSP, same views) ----------
    val_paths = table.loc[val_mask, "path"].to_numpy()
    originals = np.unique(val_paths)
    groups_index = {o: np.where(val_paths == o)[0] for o in originals}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
    head_ap, head_lo, head_hi, head_roc = evaluate(y_val_int, val_scores, originals, groups_index, rng)
    msp = untrained_scores(arrays["logits"][val_mask])["MSP"]
    msp_ap, msp_lo, msp_hi, msp_roc = evaluate(y_val_int, msp, originals, groups_index, rng)
    return {
        "seed": seed, "parameters": parameter_count, "positive_weight": positive_weight,
        "train_failures": positives, "val_failures": int(y_val.sum()), "chance": float(y_val.mean()),
        "head": (head_ap, head_lo, head_hi, head_roc), "msp": (msp_ap, msp_lo, msp_hi, msp_roc),
        "first_epoch_ap": history[0]["validation_auprc_error"],
        "val_loss_first_last": (history[0]["validation_loss"], history[-1]["validation_loss"]),
    }


def write_summary(results):
    lines = ["FAILURE HEAD RETRAINED WITH SCALED INPUTS AND A FIXED NUMBER OF EPOCHS", "=" * 74, "",
             "Source reliability-validation views only. No target, S-test or S-calibration data were read.",
             "Deviations from the original protocol: (1) inputs standardised with reliability-train",
             "statistics; (2) fixed 30 epochs, last epoch kept, no epoch chosen on validation.", "",
             f"{'seed':<7}{'head AUPRC [95% interval]':<30}{'head AUROC':>11}{'   MSP AUPRC [95% interval]':<32}{'MSP AUROC':>10}{'chance':>8}"]
    for r in results:
        h, m = r["head"], r["msp"]
        lines.append(f"{r['seed']:<7}{f'{h[0]:.3f} [{h[1]:.3f}-{h[2]:.3f}]':<30}{h[3]:>11.3f}"
                     f"{f'   {m[0]:.3f} [{m[1]:.3f}-{m[2]:.3f}]':<32}{m[3]:>10.3f}{r['chance']:>8.3f}")
    lines += ["", f"Mean head AUPRC {np.mean([r['head'][0] for r in results]):.3f}, "
                  f"mean MSP AUPRC {np.mean([r['msp'][0] for r in results]):.3f}",
              f"Learnable parameters of the head: {results[0]['parameters']:,} "
              f"(calculated from the design: 37,185)", "",
              "Per seed: training failures, validation failures, positive weight, validation loss first -> last epoch"]
    for r in results:
        lines.append(f"  seed {r['seed']}: {r['train_failures']}, {r['val_failures']}, {r['positive_weight']:.3g}, "
                     f"{r['val_loss_first_last'][0]:.3f} -> {r['val_loss_first_last'][1]:.3f}")
    lines += ["", "Notes",
              "  - The validation set holds only 5 to 8 failing original images, so intervals are wide.",
              "  - These numbers are for monitoring and for the paper's source-validation table.",
              "    They must not be used to pick between further variants of the head."]
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace this script's own output files")
    args = parser.parse_args()
    make_folders()
    outputs = (list(CHECKPOINTS.values()) + list(HISTORIES.values()) + list(VAL_PREDICTIONS.values())
               + [SUMMARY_FILE, HASHES_FILE])
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        sys.exit("Outputs of this script already exist (for example " + existing[0].name +
                 "). Use --overwrite only if you want to train again.")
    inputs = [FEATURE_FILES[s] for s in CLASSIFIER_SEEDS] + [PREDICTION_FILES[s] for s in CLASSIFIER_SEEDS]
    require_files(inputs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Source reliability data only; target and S-test are not read.")

    results = [train_one_seed(seed, device) for seed in CLASSIFIER_SEEDS]
    print("\n" + write_summary(results))

    lines = ["SHA-256 checksums for the retrained (scaled) failure heads", "", "INPUT FILES (read-only):"]
    lines += [f"{sha256_file(p)}  {p.as_posix()}" for p in inputs]
    lines += ["", "OUTPUT FILES:"]
    lines += [f"{sha256_file(p)}  {p.as_posix()}" for p in outputs if p != HASHES_FILE]
    HASHES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved new files (names start with failure_head_scaled_) in models/ and results/.")


if __name__ == "__main__":
    main()