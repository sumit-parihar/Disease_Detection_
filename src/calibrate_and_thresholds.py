"""
calibrate_and_thresholds.py  -  Fits the temperature and the 80% / 90% cut-offs on
S-calibration, using ONLY the numbers saved by extract_calibration_outputs.py.
No image is opened, no model is run, and nothing is trained.

THE TWO THINGS THAT ARE SET HERE
--------------------------------
1. TEMPERATURE (one number per classifier seed). It softens or sharpens the
   classifier's confidence so that "90% sure" is right about 90% of the time. It is
   found by minimising the negative log-likelihood on S-calibration. The scores it
   produces are the baseline B4 (temperature-scaled MSP).
2. CUT-OFFS tau80 and tau90 (one pair per method and seed). A method gives every
   image a "chance this answer is wrong" score. The system ANSWERS an image when its
   score is at or below the cut-off. tau80 is chosen so that 80% of the S-calibration
   images are answered, tau90 so that 90% are answered.
   The same rule is used for every method, which keeps the comparison fair. Each
   method gets its own number, because each score has its own scale.

THE EXACT RULE FOR A CUT-OFF (fixed here, before any target result)
-------------------------------------------------------------------
For coverage c and n images, take k = ceil(c * n) and set the cut-off to the k-th
smallest score. Images with a score at or below the cut-off are answered. The
coverage actually reached is printed next to each cut-off.

THE SEVEN METHODS
-----------------
  B1 MSP             1 - largest class probability
  B2 Entropy         entropy of the class probabilities
  B3 Margin          1 - (largest minus second-largest probability)
  B4 Temp-scaled MSP like B1, after dividing the logits by the fitted temperature
  B5 Logits-only head     failure head that sees only the 3 scores
  A1 Features-only head   failure head that sees only the 576 image numbers
  P1 Proposed head        failure head that sees image numbers and scores

A WARNING THAT GOES INTO THE PAPER
----------------------------------
S-calibration holds only a handful of wrong predictions (2, 1 and 1 for the three
seeds). The temperature uses the wrong predictions, so it is uncertain. The script
therefore also prints an interval for the temperature from resampling the images.
The cut-offs depend only on how the scores rank the images, so they are not affected.

OUTPUT FILES (results/, new names, nothing existing is overwritten)
-------------------------------------------------------------------
  temperature_scaling.csv              temperature, interval, before/after numbers
  calibration_thresholds.csv           cut-offs and reached coverage per method and seed
  frozen_calibration_settings.json     the numbers later steps must use unchanged
  calibration_summary.txt              readable summary
  calibration_hashes.txt               SHA-256 of inputs and outputs

HOW TO RUN (project root, venv active):
    python src/calibrate_and_thresholds.py
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from config import COVERAGE_LEVELS, PROJECT_ROOT, RESULTS_DIR, make_folders
from frozen_pipeline import sha256_file
from train_failure_head import CLASSIFIER_SEEDS

TEMPERATURE_FILE = RESULTS_DIR / "temperature_scaling.csv"
THRESHOLD_FILE = RESULTS_DIR / "calibration_thresholds.csv"
SETTINGS_FILE = RESULTS_DIR / "frozen_calibration_settings.json"
SUMMARY_FILE = RESULTS_DIR / "calibration_summary.txt"
HASHES_FILE = RESULTS_DIR / "calibration_hashes.txt"

BETA_LOW, BETA_HIGH = 0.05, 20.0       # search range for 1 / temperature
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 2026

METHOD_COLUMNS = {
    "B1 MSP": "msp_score",
    "B2 Entropy": "entropy_score",
    "B3 Margin": "margin_score",
    "B4 Temp-scaled MSP": None,                 # computed here
    "B5 Logits-only head": "head_logits_only",
    "A1 Features-only head": "head_features_only",
    "P1 Proposed head": "head_proposed",
}


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits, labels, temperature):
    """Negative log-likelihood, computed from the log-softmax directly. (A small constant
    inside the log would flatten the penalty for confident mistakes and spoil the search.)"""
    scaled = logits / temperature
    top = scaled.max(axis=1, keepdims=True)
    log_probs = scaled - (top + np.log(np.exp(scaled - top).sum(axis=1, keepdims=True)))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(logits, labels):
    """Golden-section search on beta = 1 / temperature (the loss is convex in beta).
    Returns (temperature, reached_search_limit)."""
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    low, high = BETA_LOW, BETA_HIGH
    c, d = high - ratio * (high - low), low + ratio * (high - low)
    fc, fd = nll(logits, labels, 1.0 / c), nll(logits, labels, 1.0 / d)
    for _ in range(80):
        if fc < fd:
            high, d, fd = d, c, fc
            c = high - ratio * (high - low)
            fc = nll(logits, labels, 1.0 / c)
        else:
            low, c, fc = c, d, fd
            d = low + ratio * (high - low)
            fd = nll(logits, labels, 1.0 / d)
    beta = (low + high) / 2.0
    return 1.0 / beta, bool(beta < BETA_LOW * 1.01 or beta > BETA_HIGH * 0.99)


def expected_calibration_error(probs, labels, bins=15):
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def brier_score(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def threshold_for_coverage(scores, coverage):
    k = int(np.ceil(coverage * len(scores)))
    return float(np.sort(scores)[k - 1])


def process_seed(seed, rng):
    path = RESULTS_DIR / f"s_calibration_outputs_seed{seed}.csv"
    table = pd.read_csv(path)
    logits = table[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64)
    labels = table["label_index"].to_numpy(dtype=np.int64)
    wrong = (table["correct"].to_numpy() == 0)

    temperature, at_limit = fit_temperature(logits, labels)
    if nll(logits, labels, temperature) > nll(logits, labels, 1.0) + 1e-9:
        raise ValueError(f"Seed {seed}: the fitted temperature is worse than temperature 1. The fit is not valid.")
    # interval for the temperature: resample images
    boot = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        pick = rng.integers(0, len(labels), len(labels))
        boot.append(fit_temperature(logits[pick], labels[pick])[0])
    low, high = np.percentile(boot, [2.5, 97.5])
    share_at_limit = float(np.mean([t <= 1.0 / BETA_HIGH * 1.01 for t in boot]))

    before, after = softmax(logits), softmax(logits / temperature)
    temp_row = {
        "classifier_seed": seed, "images": len(labels), "wrong_predictions": int(wrong.sum()),
        "temperature": temperature, "interval_low": float(low), "interval_high": float(high),
        "resamples_at_search_limit": share_at_limit, "fit_reached_search_limit": at_limit,
        "nll_before": nll(logits, labels, 1.0), "nll_after": nll(logits, labels, temperature),
        "ece_before": expected_calibration_error(before, labels), "ece_after": expected_calibration_error(after, labels),
        "brier_before": brier_score(before, labels), "brier_after": brier_score(after, labels),
    }

    scores = {name: (table[col].to_numpy(dtype=np.float64) if col else 1.0 - after.max(axis=1))
              for name, col in METHOD_COLUMNS.items()}
    threshold_rows, settings = [], {}
    for name, values in scores.items():
        row = {"classifier_seed": seed, "method": name}
        settings[name] = {}
        for coverage in COVERAGE_LEVELS:
            tag = f"{int(round(coverage * 100))}"
            tau = threshold_for_coverage(values, coverage)
            answered = values <= tau
            row[f"tau{tag}"] = tau
            row[f"coverage{tag}"] = float(answered.mean())
            row[f"wrong_answered{tag}"] = int((answered & wrong).sum())
            settings[name][f"tau{tag}"] = tau
        threshold_rows.append(row)
    return temp_row, threshold_rows, {"temperature": temperature, "thresholds": settings}


def write_summary(temp_table, threshold_table):
    lines = ["CALIBRATION AND CUT-OFFS (fitted on S-calibration only)", "=" * 70, "",
             "TEMPERATURE SCALING (one number per classifier seed)",
             f"  {'seed':<7}{'wrong':>6}{'temperature':>13}{'95% interval':>18}{'NLL before':>12}{'NLL after':>11}{'ECE before':>12}{'ECE after':>11}"]
    for _, r in temp_table.iterrows():
        lines.append(f"  {int(r.classifier_seed):<7}{int(r.wrong_predictions):>6}{r.temperature:>13.3f}"
                     f"{f'[{r.interval_low:.2f}-{r.interval_high:.2f}]':>18}{r.nll_before:>12.4f}{r.nll_after:>11.4f}"
                     f"{r.ece_before:>12.4f}{r.ece_after:>11.4f}")
    lines += ["", "  The temperature rests on very few wrong predictions, so it is uncertain (see the interval).",
              "  'Resamples at search limit' in temperature_scaling.csv shows how often a resample had no wrong prediction.", "",
              "CUT-OFFS: an image is answered when its score is at or below the cut-off.",
              "Same rule for every method: the k-th smallest score, k = ceil(coverage x images).", ""]
    for seed in sorted(threshold_table["classifier_seed"].unique()):
        part = threshold_table[threshold_table["classifier_seed"] == seed]
        lines.append(f"Classifier seed {seed}")
        lines.append(f"  {'method':<24}{'tau80':>10}{'cov80':>8}{'wrong80':>9}{'tau90':>10}{'cov90':>8}{'wrong90':>9}")
        for _, r in part.iterrows():
            lines.append(f"  {r['method']:<24}{r.tau80:>10.4f}{r.coverage80:>8.3f}{int(r.wrong_answered80):>9}"
                         f"{r.tau90:>10.4f}{r.coverage90:>8.3f}{int(r.wrong_answered90):>9}")
        lines.append("")
    lines += ["'wrong80' / 'wrong90' = wrong predictions that are still answered on S-calibration (tiny numbers,",
              "so they say little). These settings are now frozen: later steps must use them unchanged."]
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace this script's own output files")
    args = parser.parse_args()
    make_folders()
    outputs = [TEMPERATURE_FILE, THRESHOLD_FILE, SETTINGS_FILE, SUMMARY_FILE, HASHES_FILE]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        sys.exit("Outputs of this script already exist (for example " + existing[0].name +
                 "). Use --overwrite only if you want to run again.")
    inputs = [RESULTS_DIR / f"s_calibration_outputs_seed{s}.csv" for s in CLASSIFIER_SEEDS]
    missing = [p.name for p in inputs if not p.is_file()]
    if missing:
        sys.exit(f"Missing input files: {missing}. Run extract_calibration_outputs.py first.")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    temp_rows, threshold_rows, settings = [], [], {}
    for seed in CLASSIFIER_SEEDS:
        temp_row, thresholds, seed_settings = process_seed(seed, rng)
        temp_rows.append(temp_row)
        threshold_rows += thresholds
        settings[str(seed)] = seed_settings
    temp_table, threshold_table = pd.DataFrame(temp_rows), pd.DataFrame(threshold_rows)
    temp_table.to_csv(TEMPERATURE_FILE, index=False, lineterminator="\n")
    threshold_table.to_csv(THRESHOLD_FILE, index=False, lineterminator="\n")
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(write_summary(temp_table, threshold_table))

    lines = ["SHA-256 checksums for the calibration step", "", "INPUT FILES (read-only):"]
    lines += [f"{sha256_file(p)}  {p.relative_to(PROJECT_ROOT).as_posix()}" for p in inputs]
    lines += ["", "OUTPUT FILES:"] + [f"{sha256_file(p)}  {p.relative_to(PROJECT_ROOT).as_posix()}"
                                       for p in outputs if p != HASHES_FILE]
    HASHES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved new files in {RESULTS_DIR}")


if __name__ == "__main__":
    main()