"""
evaluate_s_test.py  -  The ONE-TIME evaluation of the frozen system on S-test, the
untouched in-domain (PlantVillage) reference set.

WHAT THE PROTOCOL SAYS
----------------------
S-test is used only after development, the baselines and the calibration are finished
and the system is frozen. It is looked at once. Nothing may be changed because of
what it shows.

WHAT THIS SCRIPT DOES
---------------------
  1. Checks that the frozen system is unchanged (freeze_system.verify_frozen). If any
     frozen file differs from the freeze record, it stops.
  2. Refuses to run if S-test outputs already exist (the "once" rule).
  3. For each of the three classifier seeds, runs every S-test image through the frozen
     classifier and the three frozen failure heads (original images, no augmentation).
  4. Computes, with the frozen temperature and cut-offs (nothing is re-fitted):
       - classification: accuracy, macro-F1, recall per class, confusion matrix
       - calibration of the classifier: NLL, ECE, Brier score, before and after the
         frozen temperature
       - for each of the 7 methods: AURC, change in AURC against MSP (paired interval),
         AUROC-error, AUPRC-error, and coverage / selective risk at the frozen tau80
         and tau90
     Intervals come from 2,000 resamples that keep duplicate groups together
     (group ids from the split file).
  5. Saves everything in results/ (files starting with s_test_).

WHAT TO EXPECT
--------------
The classifiers make very few mistakes on source images, so S-test will contain only a
handful of wrong predictions (possibly none). With no wrong prediction, AUROC and AUPRC
are undefined (shown as nan) and AURC is 0 for every method. This is a limit of the data,
not an error in the code. S-test therefore serves mainly as the classification reference.

WHAT IT NEVER TOUCHES
---------------------
No target (PP2020) image or label. No model or setting is changed or re-fitted.

HOW TO RUN (project root, venv active), once:
    python src/evaluate_s_test.py
There is no --overwrite option on purpose. If a run fails halfway nothing is saved, so it
can simply be run again.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

from calibrate_and_thresholds import METHOD_COLUMNS
from config import CLASS_NAMES, EVAL_BATCH_SIZE, PROJECT_ROOT, RESULTS_DIR, SPLITS_DIR, make_folders
from dataset import SourceDataset, make_loader
from freeze_system import verify_frozen
from frozen_pipeline import HEAD_VARIANTS, load_frozen_classifier, load_frozen_head, run_images, sha256_file
from metrics import (
    aurc,
    auprc_error,
    auroc_error,
    calibration_summary,
    classification_summary,
    delta_aurc,
    metric_with_interval,
    msp_score,
    selective_risk_at,
    softmax,
)
from train_failure_head import CLASSIFIER_SEEDS

SUBSET = "s_test"
RESAMPLES = 2000
SETTINGS_FILE = RESULTS_DIR / "frozen_calibration_settings.json"
SPLIT_FILE = SPLITS_DIR / "split_assignments.csv"

OUT_OUTPUTS = {s: RESULTS_DIR / f"s_test_outputs_seed{s}.csv" for s in CLASSIFIER_SEEDS}
OUT_CLASSIFICATION = RESULTS_DIR / "s_test_classification.csv"
OUT_CONFUSION = RESULTS_DIR / "s_test_confusion_matrices.csv"
OUT_RELIABILITY = RESULTS_DIR / "s_test_reliability.csv"
OUT_SUMMARY = RESULTS_DIR / "s_test_summary.txt"
OUT_HASHES = RESULTS_DIR / "s_test_hashes.txt"

HEAD_COLUMN = {"B5 Logits-only head": "head_logits_only", "A1 Features-only head": "head_features_only",
               "P1 Proposed head": "head_proposed"}


def build_table(out, temperature):
    logits = out["logits"].astype(np.float64)
    probs = softmax(logits)
    ordered = np.sort(probs, axis=1)
    predicted = probs.argmax(axis=1)
    return pd.DataFrame({
        "path": out["paths"],
        "label": [CLASS_NAMES[i] for i in out["label_index"]],
        "label_index": out["label_index"],
        "predicted_index": predicted,
        "correct": (predicted == out["label_index"]).astype(int),
        "logit_healthy": logits[:, 0], "logit_scab": logits[:, 1], "logit_rust": logits[:, 2],
        "msp_score": 1.0 - ordered[:, -1],
        "entropy_score": -(probs * np.log(probs + 1e-12)).sum(axis=1),
        "margin_score": 1.0 - (ordered[:, -1] - ordered[:, -2]),
        "temp_scaled_msp_score": msp_score(logits, temperature),
        "head_proposed": out["head_scores"]["proposed"],
        "head_logits_only": out["head_scores"]["logits_only"],
        "head_features_only": out["head_scores"]["features_only"],
    })


def method_scores(table):
    scores = {}
    for name, column in METHOD_COLUMNS.items():
        scores[name] = table["temp_scaled_msp_score"].to_numpy() if column is None else table[column].to_numpy()
    return scores


def evaluate_seed(seed, device, settings, groups):
    classifier = load_frozen_classifier(seed, device)
    heads = {variant: load_frozen_head(variant, seed, device) for variant in HEAD_VARIANTS}
    dataset = SourceDataset(SUBSET)
    if len(dataset) == 0:
        raise ValueError(f"No images are assigned to {SUBSET}.")
    out = run_images(make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False), classifier, heads, device)
    if list(out["paths"]) != sorted(groups.index):
        raise ValueError("Processed images do not match the S-test list in the split file.")
    cluster = groups.loc[out["paths"]].to_numpy()

    entry = settings[str(seed)]
    temperature = float(entry["temperature"])
    table = build_table(out, temperature)
    errors = 1 - table["correct"].to_numpy()

    summary = classification_summary(table["label_index"], table["predicted_index"], CLASS_NAMES)
    before = calibration_summary(table[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(), table["label_index"].to_numpy(), 1.0)
    after = calibration_summary(table[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(), table["label_index"].to_numpy(), temperature)
    classification_row = {
        "classifier_seed": seed, "images": len(table), "wrong_predictions": int(errors.sum()),
        "accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"],
        **{f"recall_{name}": value for name, value in summary["recall"].items()},
        "temperature_frozen": temperature,
        "nll_before": before["nll"], "nll_after": after["nll"], "ece_before": before["ece"], "ece_after": after["ece"],
        "brier_before": before["brier"], "brier_after": after["brier"],
    }
    confusion_rows = [{"classifier_seed": seed, "true_class": CLASS_NAMES[i], "predicted_class": CLASS_NAMES[j],
                       "count": int(summary["confusion_matrix"][i, j])}
                      for i in range(len(CLASS_NAMES)) for j in range(len(CLASS_NAMES))]

    scores = method_scores(table)
    reliability_rows = []
    for name, score in scores.items():
        value, low, high = metric_with_interval(aurc, errors, score, cluster, RESAMPLES, seed)
        row = {"classifier_seed": seed, "method": name, "aurc": value, "aurc_low": low, "aurc_high": high}
        if name == "B1 MSP":
            row.update({"delta_aurc": 0.0, "delta_low": 0.0, "delta_high": 0.0})
        else:
            d, dl, dh = delta_aurc(errors, score, scores["B1 MSP"], cluster, RESAMPLES, seed)
            row.update({"delta_aurc": d, "delta_low": dl, "delta_high": dh})
        row["auroc_error"] = auroc_error(errors, score)
        row["auprc_error"] = auprc_error(errors, score)
        for tag in (80, 90):
            tau = float(entry["thresholds"][name][f"tau{tag}"])
            result = selective_risk_at(errors, score, tau)
            row.update({f"tau{tag}": tau, f"coverage{tag}": result["coverage"], f"wrong_answered{tag}": result["wrong_answered"],
                        f"selective_risk{tag}": result["selective_risk"]})
        reliability_rows.append(row)
    return table, classification_row, confusion_rows, reliability_rows


def fmt(value, digits=3):
    return "nan" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def write_summary(classification, reliability):
    lines = ["S-TEST EVALUATION OF THE FROZEN SYSTEM (run once)", "=" * 78, "",
             "Original S-test images only. Frozen temperature and cut-offs; nothing was re-fitted.",
             "Intervals: 2,000 resamples that keep duplicate groups together.", "",
             "DISEASE CLASSIFICATION"]
    lines.append(f"  {'seed':<7}{'images':>7}{'wrong':>6}{'accuracy':>10}{'macro-F1':>10}"
                 + "".join(f"{'recall ' + c:>15}" for c in CLASS_NAMES))
    for _, r in classification.iterrows():
        lines.append(f"  {int(r.classifier_seed):<7}{int(r.images):>7}{int(r.wrong_predictions):>6}{fmt(r.accuracy, 4):>10}{fmt(r.macro_f1, 4):>10}"
                     + "".join(f"{fmt(r[f'recall_{c}'], 4):>15}" for c in CLASS_NAMES))
    lines += ["", "CALIBRATION OF THE CLASSIFIER (frozen temperature)",
              f"  {'seed':<7}{'temperature':>12}{'NLL before':>12}{'NLL after':>11}{'ECE before':>12}{'ECE after':>11}"]
    for _, r in classification.iterrows():
        lines.append(f"  {int(r.classifier_seed):<7}{r.temperature_frozen:>12.3f}{r.nll_before:>12.4f}{r.nll_after:>11.4f}{r.ece_before:>12.4f}{r.ece_after:>11.4f}")
    lines += ["", "RELIABILITY (AURC: lower is better; delta AURC negative = better than MSP)"]
    for seed in sorted(reliability["classifier_seed"].unique()):
        part = reliability[reliability["classifier_seed"] == seed]
        lines.append(f"  Classifier seed {seed}")
        lines.append(f"    {'method':<24}{'AURC':>9}{'delta AURC vs MSP [95%]':>32}{'AUROC':>8}{'AUPRC':>8}{'cov80':>7}{'risk80':>8}{'cov90':>7}{'risk90':>8}")
        for _, r in part.iterrows():
            lines.append(f"    {r['method']:<24}{fmt(r.aurc, 5):>9}{f'{fmt(r.delta_aurc, 5)} [{fmt(r.delta_low, 5)}, {fmt(r.delta_high, 5)}]':>32}"
                         f"{fmt(r.auroc_error):>8}{fmt(r.auprc_error):>8}{fmt(r.coverage80):>7}{fmt(r.selective_risk80, 4):>8}{fmt(r.coverage90):>7}{fmt(r.selective_risk90, 4):>8}")
        lines.append("")
    mean = reliability.groupby("method", sort=False)[["aurc", "delta_aurc", "auroc_error", "auprc_error"]].mean(numeric_only=True)
    lines.append("MEAN OVER THE THREE CLASSIFIER SEEDS (the seeds share the same S-test images, so this is not three independent tests)")
    lines.append(f"    {'method':<24}{'AURC':>9}{'delta AURC':>12}{'AUROC':>8}{'AUPRC':>8}")
    for name, r in mean.iterrows():
        lines.append(f"    {name:<24}{fmt(r.aurc, 5):>9}{fmt(r.delta_aurc, 5):>12}{fmt(r.auroc_error):>8}{fmt(r.auprc_error):>8}")
    lines += ["", "Notes",
              "  - 'nan' means the measure is undefined (for example AUROC when no prediction was wrong).",
              "  - With so few wrong predictions these numbers say little about the failure heads.",
              "  - Nothing may be changed because of this evaluation."]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    make_folders()
    outputs = list(OUT_OUTPUTS.values()) + [OUT_CLASSIFICATION, OUT_CONFUSION, OUT_RELIABILITY, OUT_SUMMARY, OUT_HASHES]
    existing = [p for p in outputs if p.exists()]
    if existing:
        sys.exit(f"S-test outputs already exist (for example {existing[0].name}). S-test is evaluated once. "
                 "Deleting them and running again must be a deliberate, documented decision.")
    verify_frozen()                                   # stops if any frozen file changed
    print("Freeze record verified: every frozen file is unchanged.")
    settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    assigned = pd.read_csv(SPLIT_FILE)
    groups = assigned[assigned["subset"] == SUBSET].set_index("path")["group_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Only {SUBSET} images are read; the target is not touched.\n")

    tables, classification_rows, confusion_rows, reliability_rows = {}, [], [], []
    for seed in CLASSIFIER_SEEDS:
        print(f"Evaluating classifier seed {seed} ...")
        table, class_row, conf_rows, rel_rows = evaluate_seed(seed, device, settings, groups)
        tables[seed] = table
        classification_rows.append(class_row)
        confusion_rows += conf_rows
        reliability_rows += rel_rows

    for seed, table in tables.items():
        table.to_csv(OUT_OUTPUTS[seed], index=False, lineterminator="\n")
    classification, reliability = pd.DataFrame(classification_rows), pd.DataFrame(reliability_rows)
    classification.to_csv(OUT_CLASSIFICATION, index=False, lineterminator="\n")
    pd.DataFrame(confusion_rows).to_csv(OUT_CONFUSION, index=False, lineterminator="\n")
    reliability.to_csv(OUT_RELIABILITY, index=False, lineterminator="\n")
    print("\n" + write_summary(classification, reliability))

    lines = ["SHA-256 checksums for the S-test evaluation", "", "OUTPUT FILES:"]
    lines += [f"{sha256_file(p)}  {p.relative_to(PROJECT_ROOT).as_posix()}" for p in outputs if p != OUT_HASHES]
    OUT_HASHES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved files starting with s_test_ in {RESULTS_DIR}")


if __name__ == "__main__":
    main()