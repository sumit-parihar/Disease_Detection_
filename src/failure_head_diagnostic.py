"""
failure_head_diagnostic.py  -  A DIAGNOSTIC comparison. It does not create a new
final failure head and it does not change anything that already exists.

QUESTION IT ANSWERS
-------------------
On the source reliability-validation views, simple "how unsure is the classifier?"
scores (MSP, entropy, margin) find the classifier's mistakes much better than the
trained failure head. WHY? This script tests which part of the head's input or
training recipe is responsible, using ONLY files that already exist
(results/reliability_v2_features_seed*.npz and reliability_v2_predictions_seed*.csv).

WHAT IT USES AND WHAT IT NEVER TOUCHES
--------------------------------------
Uses : S-reliability views (reliability_train to learn, reliability_val to score).
Never: S-train, S-val, S-calibration, S-test, or any target (PP2020) image or label.
It does not retrain the disease classifiers and it does not overwrite the saved
failure heads or their result files.

THE COMPLETE LIST OF METHODS (fixed here, before looking at any result)
-----------------------------------------------------------------------
No learning:
  MSP            1 - largest class probability
  Entropy        entropy of the class probabilities
  Margin         1 - (largest minus second-largest probability)
A simple learned reference:
  LogReg-logits  logistic regression on the 3 logits (standardised)
The protocol's small head (Dense 64 -> Dropout 0.30 -> 1 output; same loss, same
positive weight, same L2, same optimiser as train_failure_head.py), with three
different inputs and two training recipes:
  inputs  : logits only (3) | image features only (576) | features + logits (579)
  recipe A: the CURRENT recipe: raw inputs, early stopping on validation
            AUPRC-error with patience 5 (this is how the saved heads were made)
  recipe B: inputs standardised with reliability-TRAIN mean and std, then a
            FIXED 30 epochs with no early stopping and no checkpoint picking
            (so no score is chosen using the validation set)
No other variant is run. If you add variants later, write down why, and report
every variant you tried, not only the best.

HOW TO READ THE RESULT
----------------------
AUPRC-error and AUROC-error measure how well a score ranks the views on which
the classifier is wrong above the views where it is right (higher is better).
"Chance" AUPRC equals the share of failed views (about 0.022).
The 95% interval comes from resampling whole ORIGINAL images (the 13 views of an
image are kept together, because they are not independent).

OUTPUT (in results/, never overwriting existing project files)
--------------------------------------------------------------
  failure_head_diagnostic.csv       one row per method and classifier seed
  failure_head_diagnostic_summary.txt   readable tables

HOW TO RUN (project root, venv active):
    python src/failure_head_diagnostic.py
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    FAILURE_DROPOUT,
    FAILURE_HIDDEN_UNITS,
    FAILURE_L2,
    FAILURE_POS_WEIGHT_CAP,
    RESULTS_DIR,
    make_folders,
)
from train_failure_head import (
    BATCH_SIZE,
    CLASSIFIER_SEEDS,
    FEATURE_FILES,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    PREDICTION_FILES,
    load_and_validate,
    require_files,
    set_random_seeds,
)

OUT_CSV = RESULTS_DIR / "failure_head_diagnostic.csv"
OUT_TXT = RESULTS_DIR / "failure_head_diagnostic_summary.txt"
FIXED_EPOCHS_RECIPE_B = 30
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 12345


# ---------------------------------------------------------------------------
# simple scores that need no learning
# ---------------------------------------------------------------------------
def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def untrained_scores(logits):
    probs = softmax(logits)
    ordered = np.sort(probs, axis=1)
    return {
        "MSP": 1.0 - ordered[:, -1],
        "Entropy": -(probs * np.log(probs + 1e-12)).sum(axis=1),
        "Margin": 1.0 - (ordered[:, -1] - ordered[:, -2]),
    }


# ---------------------------------------------------------------------------
# the small head (same design as the protocol), with a variable input size
# ---------------------------------------------------------------------------
class SmallHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, FAILURE_HIDDEN_UNITS)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(FAILURE_DROPOUT)
        self.output = nn.Linear(FAILURE_HIDDEN_UNITS, 1)

    def forward(self, x):
        return self.output(self.dropout(self.relu(self.hidden(x)))).squeeze(1)


def score_validation(model, x_val, device):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(x_val.to(device))).cpu().numpy()


def train_head(x_train, y_train, x_val, y_val, positive_weight, seed, recipe, device):
    """Return (validation scores, number of epochs actually used / selected)."""
    set_random_seeds(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE,
                        shuffle=True, generator=generator, num_workers=0)
    model = SmallHead(x_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    y_val_np = y_val.numpy().astype(np.int64)

    best_ap, best_scores, best_epoch, waiting = -float("inf"), None, 0, 0
    last_scores = []
    max_epochs = MAX_EPOCHS if recipe == "A" else FIXED_EPOCHS_RECIPE_B
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y) + FAILURE_L2 * model.hidden.weight.square().sum()
            loss.backward()
            optimizer.step()
        scores = score_validation(model, x_val, device)
        if recipe == "A":
            ap = average_precision_score(y_val_np, scores)
            if ap > best_ap:                      # strict improvement, as in the saved heads
                best_ap, best_scores, best_epoch, waiting = ap, scores, epoch, 0
            else:
                waiting += 1
                if waiting >= PATIENCE:
                    break
        else:
            last_scores.append(scores)
    if recipe == "A":
        return best_scores, best_epoch
    return last_scores[-1], FIXED_EPOCHS_RECIPE_B


# ---------------------------------------------------------------------------
# scoring with an interval that keeps the 13 views of one image together
# ---------------------------------------------------------------------------
def evaluate(y, score, originals, groups_index, rng):
    ap = float(average_precision_score(y, score))
    roc = float(roc_auc_score(y, score))
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.choice(originals, len(originals))
        rows = np.concatenate([groups_index[o] for o in chosen])
        if y[rows].sum() == 0 or y[rows].sum() == len(rows):
            continue
        values.append(average_precision_score(y[rows], score[rows]))
    low, high = np.percentile(values, [2.5, 97.5])
    return ap, float(low), float(high), roc


def run_seed(seed, device):
    arrays, table = load_and_validate(seed)            # same safety checks as the real training
    train_mask = table["reliability_split"].eq("reliability_train").to_numpy()
    val_mask = ~train_mask
    features, logits = arrays["features"].astype(np.float32), arrays["logits"].astype(np.float32)
    y_all = arrays["failure_label"].astype(np.int64)
    y_train, y_val = y_all[train_mask], y_all[val_mask]
    positives, negatives = int(y_train.sum()), int(len(y_train) - y_train.sum())
    positive_weight = min(float(FAILURE_POS_WEIGHT_CAP), negatives / positives)

    val_paths = table.loc[val_mask, "path"].to_numpy()
    originals = np.unique(val_paths)
    groups_index = {o: np.where(val_paths == o)[0] for o in originals}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed)

    rows = []

    def record(name, inputs, recipe, score, epoch=""):
        ap, low, high, roc = evaluate(y_val, score, originals, groups_index, rng)
        rows.append({"classifier_seed": seed, "method": name, "inputs": inputs, "recipe": recipe,
                     "auprc_error": ap, "ci_low": low, "ci_high": high, "auroc_error": roc,
                     "epoch_used": epoch, "chance_auprc": float(y_val.mean())})

    for name, score in untrained_scores(logits[val_mask]).items():
        record(name, "logits", "none", score)

    # Imported here, not at the top, so that scripts which only need SmallHead / evaluate
    # do not load these scikit-learn parts (a Windows policy has blocked one of them before).
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(logits[train_mask])
    reference = LogisticRegression(C=1.0, max_iter=2000, class_weight={0: 1.0, 1: positive_weight})
    reference.fit(scaler.transform(logits[train_mask]), y_train)
    record("LogReg-logits", "logits", "none", reference.predict_proba(scaler.transform(logits[val_mask]))[:, 1])

    input_sets = {"logits only": logits, "features only": features,
                  "features+logits": np.concatenate([features, logits], axis=1)}
    for input_name, matrix in input_sets.items():
        for recipe in ("A", "B"):
            if recipe == "A":
                x_train, x_val = matrix[train_mask], matrix[val_mask]
            else:
                mean, std = matrix[train_mask].mean(0), matrix[train_mask].std(0) + 1e-6
                x_train, x_val = (matrix[train_mask] - mean) / std, (matrix[val_mask] - mean) / std
            score, epoch = train_head(
                torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)),
                torch.from_numpy(x_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32)),
                positive_weight, seed, recipe, device)
            label = "A: raw inputs, early stopping" if recipe == "A" else "B: standardised, fixed 30 epochs"
            record("Small head", input_name, label, score, epoch)
    info = {"val_views": int(val_mask.sum()), "val_failures": int(y_val.sum()),
            "val_originals": len(originals), "pos_weight": positive_weight}
    return rows, info


def saved_head_score(seed):
    path = RESULTS_DIR / f"failure_head_history_seed{seed}.csv"
    if not path.is_file():
        return None
    return float(pd.read_csv(path)["validation_auprc_error"].max())


def write_summary(table, infos):
    lines = ["FAILURE-HEAD DIAGNOSTIC (source reliability-validation views only; no target data)", "=" * 78, ""]
    lines.append("AUPRC-error, with 95% interval from resampling original images. Higher is better.")
    lines.append("Chance level = share of failed views.\n")
    for seed in sorted(table["classifier_seed"].unique()):
        part = table[table["classifier_seed"] == seed]
        info = infos[seed]
        saved = saved_head_score(seed)
        lines.append(f"Classifier seed {seed}: {info['val_views']} views, {info['val_failures']} failed views, "
                     f"{info['val_originals']} original images, chance AUPRC {part['chance_auprc'].iloc[0]:.3f}")
        if saved is not None:
            lines.append(f"  (saved failure head from train_failure_head.py: AUPRC {saved:.3f})")
        lines.append(f"  {'method':<16}{'inputs':<17}{'recipe':<35}{'AUPRC [95% interval]':<26}{'AUROC':>7}{'epoch':>7}")
        for _, r in part.iterrows():
            lines.append(f"  {r['method']:<16}{r['inputs']:<17}{r['recipe']:<35}"
                         f"{r['auprc_error']:.3f} [{r['ci_low']:.3f}-{r['ci_high']:.3f}]".ljust(95)
                         + f"{r['auroc_error']:>7.3f}{str(r['epoch_used']):>7}")
        lines.append("")
    mean = (table.groupby(["method", "inputs", "recipe"], sort=False)[["auprc_error", "auroc_error"]].mean().reset_index())
    lines.append("MEAN OVER THE THREE CLASSIFIER SEEDS")
    lines.append(f"  {'method':<16}{'inputs':<17}{'recipe':<35}{'AUPRC':>8}{'AUROC':>8}")
    for _, r in mean.iterrows():
        lines.append(f"  {r['method']:<16}{r['inputs']:<17}{r['recipe']:<35}{r['auprc_error']:>8.3f}{r['auroc_error']:>8.3f}")
    lines += ["", "READING GUIDE",
              "  - If 'logits only' reaches about the MSP level but 'features only' or 'features+logits' is far lower,",
              "    the 576 image features are what holds the head back.",
              "  - If recipe B (standardised, fixed epochs) is clearly better than recipe A for the same inputs,",
              "    the input scaling or the early-stopping rule was part of the problem.",
              "  - If no learned row beats MSP, the learned head has no advantage on source data. Report that.",
              "  - LogReg-logits is a straight-line model of the 3 logits. It cannot express 'largest minus second",
              "    largest', so it is expected to score below MSP. The small head on logits can express it.",
              "  - The validation set holds only a handful of failing original images, so the intervals are wide.",
              "    Do not choose a final head from this table without writing the choice down first."]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace this script's own output files")
    args = parser.parse_args()
    make_folders()
    if (OUT_CSV.exists() or OUT_TXT.exists()) and not args.overwrite:
        sys.exit("Diagnostic outputs already exist. Use --overwrite to replace them.")
    require_files([FEATURE_FILES[s] for s in CLASSIFIER_SEEDS] + [PREDICTION_FILES[s] for s in CLASSIFIER_SEEDS])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Diagnostic only; target and S-test data are not read.\n")

    all_rows, infos = [], {}
    for seed in CLASSIFIER_SEEDS:
        print(f"Running classifier seed {seed} ...")
        rows, info = run_seed(seed, device)
        all_rows += rows
        infos[seed] = info
    table = pd.DataFrame(all_rows)
    table.to_csv(OUT_CSV, index=False, lineterminator="\n")
    print("\n" + write_summary(table, infos))
    print(f"\nSaved: {OUT_CSV.name} and {OUT_TXT.name} in {RESULTS_DIR}")


if __name__ == "__main__":
    main()