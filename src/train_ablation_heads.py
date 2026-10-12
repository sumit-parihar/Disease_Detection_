"""
train_ablation_heads.py  -  Trains the two comparison ("ablation") failure heads
that the protocol asks for, with exactly the same corrected recipe as the
proposed head in retrain_failure_head.py.

THE TWO COMPARISON HEADS
------------------------
  logits-only head    sees only the 3 scores of the disease classifier
                      (tests whether the classifier's own outputs are enough)
  features-only head  sees only the 576 image numbers
                      (tests whether the image numbers alone carry the signal)
The proposed head (576 image numbers + 3 scores) is trained by
retrain_failure_head.py. Comparing the three shows what each kind of input adds.

SAME RECIPE AS THE PROPOSED HEAD
--------------------------------
  * inputs standardised with reliability-TRAIN mean and standard deviation
  * same head design (Dense 64 + ReLU -> Dropout 0.30 -> 1 output; only the size
    of the input differs), same weighted loss (positive weight capped at 5),
    same L2 penalty, Adam learning rate 5e-4, batch size 64
  * a FIXED 30 epochs and the last epoch is kept (no epoch picked on validation)
  * the same three seeds

WHAT IT NEVER TOUCHES
---------------------
No target (PP2020) image or label, no S-test, S-calibration, S-train or S-val
image. Only the existing S-reliability views are read. Nothing existing is
overwritten; all outputs have new names.

OUTPUT FILES
------------
  models/failure_head_scaled_logits_only_seed{seed}.pt   and  ..._features_only_...
  results/failure_head_scaled_{logits_only|features_only}_history_seed{seed}.csv
  results/failure_head_scaled_{logits_only|features_only}_val_predictions_seed{seed}.csv
  results/failure_head_ablation_summary.txt    the three heads and MSP side by side
  results/failure_head_ablation_hashes.txt     SHA-256 of inputs and outputs

HOW TO RUN (project root, venv active):
    python src/train_ablation_heads.py
Run retrain_failure_head.py first, so the proposed head can be shown in the table.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import FAILURE_L2, FAILURE_POS_WEIGHT_CAP, MODELS_DIR, RESULTS_DIR, make_folders
from failure_head_diagnostic import SmallHead, evaluate, untrained_scores
from retrain_failure_head import BOOTSTRAP_SEED, FIXED_EPOCHS, VAL_PREDICTIONS, sha256_file
from train_failure_head import (
    BATCH_SIZE,
    CLASSIFIER_SEEDS,
    FEATURE_FILES,
    LEARNING_RATE,
    PREDICTION_FILES,
    load_and_validate,
    require_files,
    set_random_seeds,
)

VARIANTS = ("logits_only", "features_only")
SUMMARY_FILE = RESULTS_DIR / "failure_head_ablation_summary.txt"
HASHES_FILE = RESULTS_DIR / "failure_head_ablation_hashes.txt"


def path_for(kind, variant, seed):
    if kind == "model":
        return MODELS_DIR / f"failure_head_scaled_{variant}_seed{seed}.pt"
    if kind == "history":
        return RESULTS_DIR / f"failure_head_scaled_{variant}_history_seed{seed}.csv"
    return RESULTS_DIR / f"failure_head_scaled_{variant}_val_predictions_seed{seed}.csv"


def train_variant(variant, arrays, table, seed, device):
    train_mask = table["reliability_split"].eq("reliability_train").to_numpy()
    val_mask = ~train_mask
    matrix = (arrays["logits"] if variant == "logits_only" else arrays["features"]).astype(np.float32)
    y_all = arrays["failure_label"].astype(np.float32)
    x_train, x_val = matrix[train_mask], matrix[val_mask]
    y_train, y_val = y_all[train_mask], y_all[val_mask]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x_train_scaled = ((x_train - mean) / std).astype(np.float32)
    x_val_scaled = ((x_val - mean) / std).astype(np.float32)

    positives = int(y_train.sum())
    positive_weight = min(float(FAILURE_POS_WEIGHT_CAP), (len(y_train) - positives) / positives)

    set_random_seeds(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train_scaled), torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    model = SmallHead(x_train.shape[1]).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_val_tensor = torch.from_numpy(x_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)
    y_val_int = y_val.astype(np.int64)

    history = []
    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        total, seen = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y) + FAILURE_L2 * model.hidden.weight.square().sum()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item()) * len(batch_x)
            seen += len(batch_x)
        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
        history.append({"epoch": epoch, "train_loss": total / seen, "validation_loss": val_loss,
                        "validation_auprc_error": float(average_precision_score(y_val_int, val_scores)),
                        "validation_auroc_error": float(roc_auc_score(y_val_int, val_scores))})

    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_mean": torch.from_numpy(mean), "input_std": torch.from_numpy(std),
        "inputs": variant, "classifier_seed": seed, "epochs": FIXED_EPOCHS, "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "positive_weight": positive_weight, "parameter_count": parameter_count,
        "recipe": "inputs standardised with reliability-train statistics; fixed 30 epochs; last epoch kept",
        "torch_version": str(torch.__version__),
    }, path_for("model", variant, seed))
    pd.DataFrame(history).to_csv(path_for("history", variant, seed), index=False, lineterminator="\n")
    val_table = table.loc[val_mask, ["view_id", "path", "view_type", "failure_label"]].reset_index(drop=True)
    val_table["failure_score"] = val_scores
    val_table.to_csv(path_for("pred", variant, seed), index=False, lineterminator="\n")
    return val_scores, parameter_count


def run_seed(seed, device):
    arrays, table = load_and_validate(seed)
    val_mask = table["reliability_split"].eq("reliability_val").to_numpy()
    y_val = arrays["failure_label"][val_mask].astype(np.int64)
    val_paths = table.loc[val_mask, "path"].to_numpy()
    originals = np.unique(val_paths)
    groups_index = {o: np.where(val_paths == o)[0] for o in originals}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed)

    scores = {"MSP (no training)": untrained_scores(arrays["logits"][val_mask])["MSP"]}
    parameters = {}
    for variant in VARIANTS:
        scores[f"{variant.replace('_', '-')} head"], parameters[variant] = train_variant(
            variant, arrays, table, seed, device)

    # the proposed head (features + logits), if retrain_failure_head.py has been run
    proposed_path = VAL_PREDICTIONS[seed]
    if proposed_path.is_file():
        proposed = pd.read_csv(proposed_path)
        expected = table.loc[val_mask, "view_id"].astype(str).to_numpy()
        if not np.array_equal(proposed["view_id"].astype(str).to_numpy(), expected):
            raise ValueError(f"Seed {seed}: proposed-head predictions are not aligned with the validation views.")
        scores["proposed head (features+logits)"] = proposed["failure_score"].to_numpy()

    rows = []
    for name, score in scores.items():
        ap, low, high, roc = evaluate(y_val, score, originals, groups_index, rng)
        rows.append({"classifier_seed": seed, "method": name, "auprc": ap, "low": low, "high": high, "auroc": roc})
    return rows, float(y_val.mean()), int(y_val.sum()), parameters


def write_summary(table, chance, failures, parameters):
    lines = ["COMPARISON HEADS (same corrected recipe) AND THE PROPOSED HEAD", "=" * 74, "",
             "Source reliability-validation views only. No target, S-test or S-calibration data were read.",
             "AUPRC-error with a 95% interval from resampling original images; higher is better.", ""]
    for seed in sorted(table["classifier_seed"].unique()):
        part = table[table["classifier_seed"] == seed]
        lines.append(f"Classifier seed {seed}: {failures[seed]} failed validation views, chance AUPRC {chance[seed]:.3f}")
        lines.append(f"  {'method':<34}{'AUPRC [95% interval]':<28}{'AUROC':>7}")
        for _, r in part.iterrows():
            lines.append(f"  {r['method']:<34}{f'{r.auprc:.3f} [{r.low:.3f}-{r.high:.3f}]':<28}{r.auroc:>7.3f}")
        lines.append("")
    mean = table.groupby("method", sort=False)[["auprc", "auroc"]].mean().reset_index()
    lines.append("MEAN OVER THE THREE CLASSIFIER SEEDS")
    lines.append(f"  {'method':<34}{'AUPRC':>8}{'AUROC':>8}")
    for _, r in mean.iterrows():
        lines.append(f"  {r['method']:<34}{r.auprc:>8.3f}{r.auroc:>8.3f}")
    lines += ["", f"Learnable parameters: logits-only head {parameters['logits_only']:,}; "
                  f"features-only head {parameters['features_only']:,}; proposed head 37,185.", "",
              "Notes",
              "  - Only 5 to 8 failing original images sit in the validation set, so intervals are wide.",
              "  - These numbers are for the paper's source-validation table. They must not be used",
              "    to pick between further variants."]
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace this script's own output files")
    args = parser.parse_args()
    make_folders()
    outputs = [SUMMARY_FILE, HASHES_FILE]
    for variant in VARIANTS:
        for seed in CLASSIFIER_SEEDS:
            outputs += [path_for("model", variant, seed), path_for("history", variant, seed), path_for("pred", variant, seed)]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        sys.exit("Outputs of this script already exist (for example " + existing[0].name +
                 "). Use --overwrite only if you want to train again.")
    inputs = [FEATURE_FILES[s] for s in CLASSIFIER_SEEDS] + [PREDICTION_FILES[s] for s in CLASSIFIER_SEEDS]
    require_files(inputs)
    if not all(VAL_PREDICTIONS[s].is_file() for s in CLASSIFIER_SEEDS):
        print("Note: the proposed head's predictions were not found, so it will not appear in the table.\n"
              "      Run retrain_failure_head.py first if you want it included.\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Source reliability data only; target and S-test are not read.\n")

    all_rows, chance, failures, parameters = [], {}, {}, {}
    for seed in CLASSIFIER_SEEDS:
        print(f"Training the two comparison heads for classifier seed {seed} ...")
        rows, chance[seed], failures[seed], parameters = run_seed(seed, device)
        all_rows += rows
    table = pd.DataFrame(all_rows)
    print("\n" + write_summary(table, chance, failures, parameters))

    lines = ["SHA-256 checksums for the comparison failure heads", "", "INPUT FILES (read-only):"]
    lines += [f"{sha256_file(p)}  {p.as_posix()}" for p in inputs]
    lines += ["", "OUTPUT FILES:"] + [f"{sha256_file(p)}  {p.as_posix()}" for p in outputs if p != HASHES_FILE]
    HASHES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSaved new files (names start with failure_head_scaled_ or failure_head_ablation_).")


if __name__ == "__main__":
    main()