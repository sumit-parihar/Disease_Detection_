"""
evaluate_target.py  -  Joins the target labels to the saved frozen predictions and
computes every measure of the protocol. This is the first and only place where the
PP2020 labels are read. Nothing is trained, fitted or changed.

BEFORE IT COMPUTES ANYTHING, IT CHECKS
--------------------------------------
  1. The frozen system is unchanged (freeze_system.verify_frozen).
  2. The label-free manifest is unchanged (its SHA-256 sidecar).
  3. The three prediction files are exactly the ones recorded right after inference
     (results/target_inference_hashes.txt).
  4. The prediction files are COMMITTED in Git. If they are untracked or modified, it stops:
     the protocol says the predictions are saved and recorded before the labels are joined.
  5. No target evaluation output exists yet (the primary analysis is run once).

WHAT IT COMPUTES, PER CLASSIFIER SEED
-------------------------------------
  * disease classification: accuracy, macro-F1, recall per class, confusion matrix,
    errors per class
  * calibration of the classifier before and after the frozen temperature (NLL, ECE, Brier)
  * for each of the 7 methods: AURC, change in AURC against MSP (paired), AUROC-error,
    AUPRC-error, and coverage and selective risk at the frozen tau80 and tau90
  * the PRIMARY endpoint: change in AURC of the proposed head (P1) against MSP (B1),
    per seed and as the mean over the three seeds
  * a SECONDARY check that keeps one image per repeated-photo group

UNCERTAINTY (decision: repeated photos are resampled together)
-------------------------------------------------------------
2,000 resamples. Each resample draws whole clusters (the cluster_id of the label-free
manifest) with replacement, so repeated photos of one leaf move together. The same
resamples are used for every method and every seed, so differences between methods are
paired. The mean-over-seeds interval resamples the clusters once and averages the three
seeds inside each resample.

READING THE RESULT
------------------
AURC: lower is better. Change in AURC against MSP: negative means better than MSP.
A useless ranking has an AURC equal to the error rate. The classifier is wrong on most
target images, so the error rate (and therefore the AURC of a useless ranking) is high.

OUTPUT FILES (results/)
-----------------------
  target_classification.csv, target_confusion_matrices.csv, target_reliability.csv,
  target_reliability_seed_mean.csv, target_sensitivity_one_per_cluster.csv,
  target_summary.txt, target_evaluation_hashes.txt

HOW TO RUN (project root, venv active), once:
    python src/evaluate_target.py
There is no --overwrite option on purpose.
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from calibrate_and_thresholds import METHOD_COLUMNS
from config import CLASS_NAMES, CLASS_TO_INDEX, EXPECTED_TARGET_COUNTS, MANIFEST_DIR, PROJECT_ROOT, RESULTS_DIR, make_folders
from freeze_system import verify_frozen
from frozen_pipeline import sha256_file
from metrics import (
    aurc,
    auprc_error,
    auroc_error,
    calibration_summary,
    classification_summary,
    selective_risk_at,
)
from train_failure_head import CLASSIFIER_SEEDS

RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
PRIMARY_METHOD, REFERENCE_METHOD = "P1 Proposed head", "B1 MSP"

MANIFEST_FILE = MANIFEST_DIR / "target_inference_manifest.csv"
MANIFEST_SIDECAR = MANIFEST_DIR / "target_inference_manifest.csv.sha256"
INFERENCE_HASHES = RESULTS_DIR / "target_inference_hashes.txt"
SETTINGS_FILE = RESULTS_DIR / "frozen_calibration_settings.json"
PREDICTIONS = {s: RESULTS_DIR / f"target_predictions_seed{s}.csv" for s in CLASSIFIER_SEEDS}

OUT_CLASSIFICATION = RESULTS_DIR / "target_classification.csv"
OUT_CONFUSION = RESULTS_DIR / "target_confusion_matrices.csv"
OUT_RELIABILITY = RESULTS_DIR / "target_reliability.csv"
OUT_SEED_MEAN = RESULTS_DIR / "target_reliability_seed_mean.csv"
OUT_SENSITIVITY = RESULTS_DIR / "target_sensitivity_one_per_cluster.csv"
OUT_SUMMARY = RESULTS_DIR / "target_summary.txt"
OUT_HASHES = RESULTS_DIR / "target_evaluation_hashes.txt"
OUTPUTS = [OUT_CLASSIFICATION, OUT_CONFUSION, OUT_RELIABILITY, OUT_SEED_MEAN, OUT_SENSITIVITY, OUT_SUMMARY, OUT_HASHES]


# ---------------------------------------------------------------- pre-checks
def relative(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


def recorded_hashes():
    recorded = {}
    for line in INFERENCE_HASHES.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            recorded[parts[1]] = parts[0]
    return recorded


def precheck():
    existing = [p for p in OUTPUTS if p.exists()]
    if existing:
        sys.exit(f"Target evaluation outputs already exist (for example {existing[0].name}). The primary analysis is run once. "
                 "Deleting them and running again must be a deliberate, documented decision.")
    verify_frozen()
    print("Freeze record verified: every frozen file is unchanged.")
    if not MANIFEST_FILE.is_file() or sha256_file(MANIFEST_FILE) != MANIFEST_SIDECAR.read_text(encoding="utf-8").split()[0]:
        sys.exit("The label-free manifest is missing or has changed.")
    if not INFERENCE_HASHES.is_file():
        sys.exit("results/target_inference_hashes.txt not found. Run run_target_inference.py first.")
    recorded = recorded_hashes()
    for seed, path in PREDICTIONS.items():
        if not path.is_file():
            sys.exit(f"Missing prediction file: {path.name}")
        if recorded.get(relative(path)) != sha256_file(path):
            sys.exit(f"{path.name} differs from the hash recorded right after inference.")
    print("Prediction files verified: identical to the hashes recorded after inference.")
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--"] + [relative(p) for p in PREDICTIONS.values()] + [relative(INFERENCE_HASHES)],
                                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
        if status.returncode == 0 and status.stdout.strip():
            sys.exit("The prediction files are not committed in Git yet. Commit and push them first, then run this script again.")
        if status.returncode != 0:
            print("WARNING: Git status could not be read, so the commit check was skipped.")
        else:
            print("Prediction files are committed in Git.")
    except (OSError, subprocess.SubprocessError):
        print("WARNING: Git is not available, so the commit check was skipped.")


# ---------------------------------------------------------------- resampling
def make_resamples(cluster_ids, draws, seed):
    unique = np.unique(cluster_ids)
    groups = [np.flatnonzero(cluster_ids == c) for c in unique]
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(draws):
        chosen = rng.integers(0, len(groups), len(groups))
        resamples.append(np.concatenate([groups[g] for g in chosen]))
    return resamples


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def boot(function, errors, score, resamples):
    return np.array([function(errors[i], score[i]) for i in resamples])


# ---------------------------------------------------------------- main work
def method_scores(table):
    return {name: (table["temp_scaled_msp_score"] if column is None else table[column]).to_numpy(dtype=float)
            for name, column in METHOD_COLUMNS.items()}


def main():
    make_folders()
    precheck()

    manifest = pd.read_csv(MANIFEST_FILE, keep_default_na=False)
    labels_table = pd.read_csv(MANIFEST_DIR / "manifest.csv", keep_default_na=False)
    labels_table = labels_table[labels_table["dataset"] == "target"].set_index("path")
    settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    print("\nJoining the target labels now.")

    seeds_data = {}
    for seed in CLASSIFIER_SEEDS:
        table = pd.read_csv(PREDICTIONS[seed], keep_default_na=False)
        if table["path"].tolist() != manifest["path"].tolist():
            sys.exit(f"Seed {seed}: prediction rows do not match the manifest.")
        label_names = labels_table.loc[table["path"], "label"].tolist()
        if any(name not in CLASS_NAMES for name in label_names):
            sys.exit("A non-eligible label was found among the predicted images.")
        true_index = np.array([CLASS_TO_INDEX[n] for n in label_names])
        seeds_data[seed] = {"table": table, "true": true_index, "pred": table["predicted_index"].to_numpy(),
                            "errors": (table["predicted_index"].to_numpy() != true_index).astype(int)}
    counts = {name: int((seeds_data[CLASSIFIER_SEEDS[0]]["true"] == CLASS_TO_INDEX[name]).sum()) for name in CLASS_NAMES}
    if counts != EXPECTED_TARGET_COUNTS:
        sys.exit(f"Joined class counts {counts} differ from the published {EXPECTED_TARGET_COUNTS}.")

    cluster_ids = manifest["cluster_id"].to_numpy()
    resamples = make_resamples(cluster_ids, RESAMPLES, BOOTSTRAP_SEED)

    classification_rows, confusion_rows, reliability_rows = [], [], []
    boot_aurc = {}
    for seed in CLASSIFIER_SEEDS:
        data = seeds_data[seed]
        table, errors = data["table"], data["errors"]
        entry = settings[str(seed)]
        logits = table[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=float)
        summary = classification_summary(data["true"], data["pred"], CLASS_NAMES)
        before, after = calibration_summary(logits, data["true"], 1.0), calibration_summary(logits, data["true"], float(entry["temperature"]))
        classification_rows.append({
            "classifier_seed": seed, "images": len(errors), "wrong_predictions": int(errors.sum()), "error_rate": float(errors.mean()),
            "accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"],
            **{f"recall_{n}": v for n, v in summary["recall"].items()}, "temperature_frozen": float(entry["temperature"]),
            "nll_before": before["nll"], "nll_after": after["nll"], "ece_before": before["ece"], "ece_after": after["ece"],
            "brier_before": before["brier"], "brier_after": after["brier"]})
        for i, true_name in enumerate(CLASS_NAMES):
            row_total = int(summary["confusion_matrix"][i].sum())
            for j, pred_name in enumerate(CLASS_NAMES):
                confusion_rows.append({"classifier_seed": seed, "true_class": true_name, "predicted_class": pred_name,
                                       "count": int(summary["confusion_matrix"][i, j]), "true_class_images": row_total})

        scores = method_scores(table)
        for name, score in scores.items():
            boot_aurc[(seed, name)] = boot(aurc, errors, score, resamples)
        for name, score in scores.items():
            point = aurc(errors, score)
            low, high = interval(boot_aurc[(seed, name)])
            delta_boot = boot_aurc[(seed, name)] - boot_aurc[(seed, REFERENCE_METHOD)]
            delta = point - aurc(errors, scores[REFERENCE_METHOD])
            dl, dh = interval(delta_boot)
            roc_boot, pr_boot = boot(auroc_error, errors, score, resamples), boot(auprc_error, errors, score, resamples)
            row = {"classifier_seed": seed, "method": name, "error_rate": float(errors.mean()),
                   "aurc": point, "aurc_low": low, "aurc_high": high, "delta_aurc": delta, "delta_low": dl, "delta_high": dh,
                   "auroc_error": auroc_error(errors, score), "auroc_low": interval(roc_boot)[0], "auroc_high": interval(roc_boot)[1],
                   "auprc_error": auprc_error(errors, score), "auprc_low": interval(pr_boot)[0], "auprc_high": interval(pr_boot)[1]}
            for tag in (80, 90):
                tau = float(entry["thresholds"][name][f"tau{tag}"])
                result = selective_risk_at(errors, score, tau)
                row.update({f"tau{tag}": tau, f"coverage{tag}": result["coverage"], f"answered{tag}": result["answered"],
                            f"wrong_answered{tag}": result["wrong_answered"], f"selective_risk{tag}": result["selective_risk"]})
            reliability_rows.append(row)
        print(f"  seed {seed}: done")

    reliability = pd.DataFrame(reliability_rows)

    # mean over the three seeds, one joint resampling
    seed_mean_rows = []
    for name in METHOD_COLUMNS:
        mean_aurc = np.mean([boot_aurc[(s, name)] for s in CLASSIFIER_SEEDS], axis=0)
        mean_delta = np.mean([boot_aurc[(s, name)] - boot_aurc[(s, REFERENCE_METHOD)] for s in CLASSIFIER_SEEDS], axis=0)
        part = reliability[reliability["method"] == name]
        point_delta = float(part["delta_aurc"].mean())
        dl, dh = interval(mean_delta)
        al, ah = interval(mean_aurc)
        seed_mean_rows.append({"method": name, "mean_aurc": float(part["aurc"].mean()), "mean_aurc_low": al, "mean_aurc_high": ah,
                               "mean_delta_aurc": point_delta, "mean_delta_low": dl, "mean_delta_high": dh,
                               "mean_auroc_error": float(part["auroc_error"].mean()), "mean_auprc_error": float(part["auprc_error"].mean())})
    seed_mean = pd.DataFrame(seed_mean_rows)

    # secondary check: one image per cluster
    first = np.array([np.flatnonzero(cluster_ids == c)[0] for c in np.unique(cluster_ids)])
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    subset_resamples = [rng.integers(0, len(first), len(first)) for _ in range(RESAMPLES)]
    sensitivity_rows = []
    for seed in CLASSIFIER_SEEDS:
        errors = seeds_data[seed]["errors"][first]
        scores = {n: s[first] for n, s in method_scores(seeds_data[seed]["table"]).items()}
        msp_boot = boot(aurc, errors, scores[REFERENCE_METHOD], subset_resamples)
        for name, score in scores.items():
            b = boot(aurc, errors, score, subset_resamples)
            dl, dh = interval(b - msp_boot)
            sensitivity_rows.append({"classifier_seed": seed, "method": name, "images": len(first), "aurc": aurc(errors, score),
                                     "delta_aurc": aurc(errors, score) - aurc(errors, scores[REFERENCE_METHOD]),
                                     "delta_low": dl, "delta_high": dh})
    sensitivity = pd.DataFrame(sensitivity_rows)

    classification, confusion = pd.DataFrame(classification_rows), pd.DataFrame(confusion_rows)
    classification.to_csv(OUT_CLASSIFICATION, index=False, lineterminator="\n")
    confusion.to_csv(OUT_CONFUSION, index=False, lineterminator="\n")
    reliability.to_csv(OUT_RELIABILITY, index=False, lineterminator="\n")
    seed_mean.to_csv(OUT_SEED_MEAN, index=False, lineterminator="\n")
    sensitivity.to_csv(OUT_SENSITIVITY, index=False, lineterminator="\n")
    summary = write_summary(classification, confusion, reliability, seed_mean, sensitivity)
    print("\n" + summary)

    lines = ["SHA-256 checksums for the target evaluation", "", "OUTPUT FILES:"]
    lines += [f"{sha256_file(p)}  {relative(p)}" for p in OUTPUTS if p != OUT_HASHES]
    OUT_HASHES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved files starting with target_ in {RESULTS_DIR}")


def fmt(value, digits=3):
    return "nan" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def write_summary(classification, confusion, reliability, seed_mean, sensitivity):
    L = ["TARGET (PP2020) EVALUATION OF THE FROZEN SYSTEM", "=" * 86, "",
         "Primary set: 1,730 images (516 healthy, 592 scab, 622 rust); multiple_diseases images were excluded before inference.",
         "Intervals: 2,000 resamples of repeated-photo clusters, shared by all methods and seeds.", "",
         "DISEASE CLASSIFICATION",
         f"  {'seed':<7}{'wrong':>7}{'error rate':>12}{'accuracy':>10}{'macro-F1':>10}" + "".join(f"{'recall ' + c:>15}" for c in CLASS_NAMES)]
    for _, r in classification.iterrows():
        L.append(f"  {int(r.classifier_seed):<7}{int(r.wrong_predictions):>7}{fmt(r.error_rate, 4):>12}{fmt(r.accuracy, 4):>10}{fmt(r.macro_f1, 4):>10}"
                 + "".join(f"{fmt(r[f'recall_{c}'], 4):>15}" for c in CLASS_NAMES))
    L += ["", "CONFUSION MATRICES (rows: true class, columns: predicted class)"]
    for seed in sorted(confusion["classifier_seed"].unique()):
        part = confusion[confusion["classifier_seed"] == seed]
        L.append(f"  seed {seed}")
        L.append(f"    {'':<10}" + "".join(f"{c:>10}" for c in CLASS_NAMES))
        for t in CLASS_NAMES:
            L.append(f"    {t:<10}" + "".join(f"{int(part[(part.true_class == t) & (part.predicted_class == p)]['count'].iloc[0]):>10}" for p in CLASS_NAMES))
    L += ["", "CALIBRATION OF THE CLASSIFIER (frozen temperature)",
          f"  {'seed':<7}{'NLL before':>12}{'NLL after':>11}{'ECE before':>12}{'ECE after':>11}{'Brier before':>14}{'Brier after':>13}"]
    for _, r in classification.iterrows():
        L.append(f"  {int(r.classifier_seed):<7}{r.nll_before:>12.4f}{r.nll_after:>11.4f}{r.ece_before:>12.4f}{r.ece_after:>11.4f}{r.brier_before:>14.4f}{r.brier_after:>13.4f}")
    L += ["", f"PRIMARY ENDPOINT: change in AURC of the proposed head against MSP (negative = better than MSP)"]
    for seed in sorted(reliability["classifier_seed"].unique()):
        r = reliability[(reliability.classifier_seed == seed) & (reliability.method == PRIMARY_METHOD)].iloc[0]
        L.append(f"  seed {seed}: {r.delta_aurc:+.4f} [{r.delta_low:+.4f}, {r.delta_high:+.4f}]   (AURC {r.aurc:.4f}; MSP {reliability[(reliability.classifier_seed == seed) & (reliability.method == REFERENCE_METHOD)].iloc[0].aurc:.4f}; error rate {r.error_rate:.4f})")
    m = seed_mean[seed_mean.method == PRIMARY_METHOD].iloc[0]
    L.append(f"  mean over the three seeds: {m.mean_delta_aurc:+.4f} [{m.mean_delta_low:+.4f}, {m.mean_delta_high:+.4f}]")
    L += ["", "RELIABILITY BY SEED (AURC lower is better; AUROC and AUPRC: higher is better)"]
    for seed in sorted(reliability["classifier_seed"].unique()):
        part = reliability[reliability["classifier_seed"] == seed]
        L.append(f"  Classifier seed {seed}   (a useless ranking has AURC = error rate = {part.error_rate.iloc[0]:.4f})")
        L.append(f"    {'method':<24}{'AURC':>8}{'delta AURC [95%]':>28}{'AUROC':>8}{'AUPRC':>8}{'cov80':>7}{'risk80':>8}{'cov90':>7}{'risk90':>8}")
        for _, r in part.iterrows():
            L.append(f"    {r['method']:<24}{fmt(r.aurc, 4):>8}{f'{r.delta_aurc:+.4f} [{r.delta_low:+.4f}, {r.delta_high:+.4f}]':>28}"
                     f"{fmt(r.auroc_error):>8}{fmt(r.auprc_error):>8}{fmt(r.coverage80):>7}{fmt(r.selective_risk80, 3):>8}{fmt(r.coverage90):>7}{fmt(r.selective_risk90, 3):>8}")
        L.append("")
    L.append("MEAN OVER THE THREE CLASSIFIER SEEDS (the seeds share the same images)")
    L.append(f"    {'method':<24}{'AURC':>8}{'delta AURC [95%]':>28}{'AUROC':>8}{'AUPRC':>8}")
    for _, r in seed_mean.iterrows():
        L.append(f"    {r['method']:<24}{fmt(r.mean_aurc, 4):>8}{f'{r.mean_delta_aurc:+.4f} [{r.mean_delta_low:+.4f}, {r.mean_delta_high:+.4f}]':>28}{fmt(r.mean_auroc_error):>8}{fmt(r.mean_auprc_error):>8}")
    L += ["", f"SECONDARY CHECK: one image per repeated-photo cluster ({int(sensitivity.images.iloc[0])} images), change in AURC against MSP"]
    for seed in sorted(sensitivity["classifier_seed"].unique()):
        r = sensitivity[(sensitivity.classifier_seed == seed) & (sensitivity.method == PRIMARY_METHOD)].iloc[0]
        L.append(f"  seed {seed}, proposed head: {r.delta_aurc:+.4f} [{r.delta_low:+.4f}, {r.delta_high:+.4f}]")
    L += ["", "Notes",
          "  - The primary analysis stops here. Anything further is exploratory and must be labelled as such.",
          "  - Nothing may be changed in the system, the thresholds or the analysis in response to these results."]
    OUT_SUMMARY.write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


if __name__ == "__main__":
    main()