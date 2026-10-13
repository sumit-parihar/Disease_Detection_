"""
metrics.py  -  All the measures of the protocol, as plain functions. It reads no
file and trains nothing. Other scripts (S-test and target evaluation) import it, so
that every dataset is measured in exactly the same way.

TWO KINDS OF MEASURES
---------------------
A. "How well does the system know when it is wrong?"  (the reliability measures)
   Every image has an ERROR flag (1 = the classifier was wrong) and a SCORE
   (higher = less trustworthy). A good score ranks the wrong answers above the
   right ones.
   * AURC               Sort the images from most to least trusted. At each point the
                        "selective risk" is the share of wrong answers among the images
                        answered so far. AURC is the average of that risk over all
                        points. LOWER IS BETTER. (If the ranking were useless, AURC
                        would equal the overall error rate.)
   * Delta AURC         AURC of a method minus AURC of MSP, on the same images.
                        NEGATIVE means better than MSP.
   * AUROC-error        Chance that a random wrong answer gets a higher score than a
                        random right answer (0.5 = guessing, 1 = perfect).
   * AUPRC-error        Like AUROC but focused on finding the wrong answers; its
                        guessing level equals the share of wrong answers.
   * Selective risk at a cut-off tau   Answer an image when its score <= tau; report the
                        share answered (coverage) and the error rate among the answered.
B. "How good is the disease classifier itself?"
   accuracy, macro-F1, recall per class, confusion matrix, and the calibration
   measures ECE, Brier score and NLL.

TIES
----
If several images have exactly the same score, the risk-coverage curve uses the average
over all possible orders of those images. The result therefore does not depend on the
order of the rows in the file. If ALL scores are equal, AURC equals the overall error
rate (the level of a useless ranking).

UNCERTAINTY
-----------
bootstrap_interval() resamples images, and can resample whole GROUPS of related images
together (duplicates, or the 13 views of one image), so related images are not treated
as independent. The protocol plans about 2,000 resamples.

SELF-TEST
---------
    python src/metrics.py
checks the functions on toy examples with known answers, and on your real source
reliability-validation views, where the AUPRC and AUROC of MSP must equal the numbers
reported earlier. It prints only; it writes no file.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from calibrate_and_thresholds import brier_score, expected_calibration_error, nll, softmax

COVERAGE_TAGS = (80, 90)


# ---------------------------------------------------------------------------
# A. reliability measures
# ---------------------------------------------------------------------------
def risk_coverage_curve(errors, scores):
    """Return (coverage, selective_risk) after answering 1, 2, ... n images.
    Images with exactly the same score are ordered randomly: the risk at each point is the
    AVERAGE over all possible orders of the tied images. This does not depend on the order
    of the rows in the file. (Without ties it is the ordinary risk-coverage curve.)"""
    errors = np.asarray(errors, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(errors)
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_errors = scores[order], errors[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])   # first index of each tie group
    sizes = np.diff(np.r_[starts, n])
    errors_in_group = np.add.reduceat(sorted_errors, starts)
    errors_before = np.r_[0.0, np.cumsum(errors_in_group)[:-1]]
    position_in_group = np.arange(n) - np.repeat(starts, sizes) + 1
    expected_errors = np.repeat(errors_before, sizes) + position_in_group / np.repeat(sizes, sizes) * np.repeat(errors_in_group, sizes)
    answered = np.arange(1, n + 1)
    return answered / n, expected_errors / answered


def aurc(errors, scores):
    return float(risk_coverage_curve(errors, scores)[1].mean())


def auroc_error(errors, scores):
    errors = np.asarray(errors)
    if errors.min() == errors.max():
        return float("nan")
    return float(roc_auc_score(errors, scores))


def auprc_error(errors, scores):
    errors = np.asarray(errors)
    if errors.sum() == 0:
        return float("nan")
    return float(average_precision_score(errors, scores))


def selective_risk_at(errors, scores, tau):
    """Answer images whose score is at or below tau."""
    errors, scores = np.asarray(errors), np.asarray(scores)
    answered = scores <= tau
    count = int(answered.sum())
    return {
        "coverage": count / len(errors),
        "answered": count,
        "wrong_answered": int(errors[answered].sum()),
        "selective_risk": float(errors[answered].mean()) if count else float("nan"),
    }


# ---------------------------------------------------------------------------
# uncertainty by resampling
# ---------------------------------------------------------------------------
def bootstrap_interval(statistic, n_items, clusters=None, resamples=2000, seed=0, level=95):
    """statistic(index_array) -> float. Resample items, or whole clusters when
    `clusters` (one label per item) is given. Returns (low, high); resamples for
    which the statistic is not defined (nan) are skipped."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(n_items)]
    else:
        clusters = np.asarray(clusters)
        groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups[g] for g in chosen])
        value = statistic(index)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    tail = (100 - level) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def delta_aurc(errors, scores_method, scores_msp, clusters=None, resamples=2000, seed=0):
    """AURC(method) - AURC(MSP) on the same images, with a paired interval."""
    errors, scores_method, scores_msp = map(np.asarray, (errors, scores_method, scores_msp))
    value = aurc(errors, scores_method) - aurc(errors, scores_msp)
    low, high = bootstrap_interval(
        lambda i: aurc(errors[i], scores_method[i]) - aurc(errors[i], scores_msp[i]),
        len(errors), clusters, resamples, seed)
    return float(value), low, high


def metric_with_interval(function, errors, scores, clusters=None, resamples=2000, seed=0):
    errors, scores = np.asarray(errors), np.asarray(scores)
    value = function(errors, scores)
    low, high = bootstrap_interval(lambda i: function(errors[i], scores[i]), len(errors), clusters, resamples, seed)
    return float(value), low, high


# ---------------------------------------------------------------------------
# B. disease-classifier measures
# ---------------------------------------------------------------------------
def classification_summary(true_index, predicted_index, class_names):
    true_index, predicted_index = np.asarray(true_index), np.asarray(predicted_index)
    k = len(class_names)
    matrix = np.zeros((k, k), dtype=int)
    for t, p in zip(true_index, predicted_index):
        matrix[t, p] += 1
    recall = {name: (float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else float("nan"))
              for i, name in enumerate(class_names)}
    return {
        "accuracy": float((true_index == predicted_index).mean()),
        "macro_f1": float(f1_score(true_index, predicted_index, labels=list(range(k)), average="macro", zero_division=0)),
        "recall": recall,
        "confusion_matrix": matrix,
    }


def calibration_summary(logits, labels, temperature=1.0):
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    probs = softmax(logits / temperature)
    return {"nll": nll(logits, labels, temperature),
            "ece": expected_calibration_error(probs, labels),
            "brier": brier_score(probs, labels)}


def msp_score(logits, temperature=1.0):
    """1 - largest class probability (higher = less trustworthy)."""
    return 1.0 - softmax(np.asarray(logits, dtype=np.float64) / temperature).max(axis=1)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def _toy_checks():
    print("1. Toy examples with known answers")
    errors = np.array([0, 0, 0, 1])
    good = np.array([0.1, 0.2, 0.3, 0.9])           # the wrong answer is ranked last
    bad = np.array([0.9, 0.3, 0.2, 0.1])            # the wrong answer is ranked first
    ok = True
    ok &= _check("AURC of a perfect ranking = 0.0625", abs(aurc(errors, good) - 0.0625) < 1e-12)
    ok &= _check("AURC of the worst ranking = 0.5208", abs(aurc(errors, bad) - (1 + 0.5 + 1 / 3 + 0.25) / 4) < 1e-12)
    ok &= _check("AUROC-error perfect = 1, worst = 0", auroc_error(errors, good) == 1.0 and auroc_error(errors, bad) == 0.0)
    ok &= _check("AUPRC-error perfect = 1", auprc_error(errors, good) == 1.0)
    result = selective_risk_at(errors, good, 0.35)
    ok &= _check("Selective risk at a cut-off: coverage 0.75, no wrong answer",
                 result["coverage"] == 0.75 and result["wrong_answered"] == 0 and result["selective_risk"] == 0.0)
    summary = classification_summary([0, 0, 1, 2], [0, 1, 1, 2], ["a", "b", "c"])
    ok &= _check("Accuracy 0.75 and confusion matrix", summary["accuracy"] == 0.75 and summary["confusion_matrix"][0, 1] == 1)
    delta, low, high = delta_aurc(errors, good, bad, resamples=200)
    ok &= _check("Delta AURC (perfect minus worst) is negative", delta < 0)
    tied = np.array([0.5, 0.5, 0.5, 0.5])
    ok &= _check("All scores equal: AURC equals the error rate (0.25)", abs(aurc(errors, tied) - 0.25) < 1e-12)
    rng = np.random.default_rng(0)
    big_errors = (rng.random(300) < 0.1).astype(int)
    big_scores = np.round(rng.random(300), 1)                      # many exact ties
    shuffled = rng.permutation(300)
    ok &= _check("With many ties, shuffling the rows does not change AURC",
                 abs(aurc(big_errors, big_scores) - aurc(big_errors[shuffled], big_scores[shuffled])) < 1e-12)
    return ok


def _real_data_check():
    from config import RESULTS_DIR
    print("\n2. Real source reliability-validation views (should reproduce the earlier numbers)")
    ok = True
    for seed in (42, 123, 2026):
        prediction_file = RESULTS_DIR / f"reliability_v2_predictions_seed{seed}.csv"
        if not prediction_file.is_file():
            print(f"  (seed {seed}: {prediction_file.name} not found, skipped)")
            continue
        table = pd.read_csv(prediction_file)
        table = table[table["reliability_split"] == "reliability_val"].reset_index(drop=True)
        errors = table["failure_label"].to_numpy()
        logits = table[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy() if "logit_healthy" in table else None
        if logits is None:
            print(f"  (seed {seed}: no logit columns, skipped)")
            continue
        msp = msp_score(logits)
        clusters = table["path"].to_numpy()
        print(f"\n  Seed {seed}: {len(errors)} views, {int(errors.sum())} wrong, "
              f"{len(np.unique(clusters))} original images (resampled together)")
        print(f"    {'method':<26}{'AURC':>8}{'delta AURC vs MSP [95%]':>30}{'AUROC':>8}{'AUPRC':>8}")
        print(f"    {'MSP':<26}{aurc(errors, msp):>8.4f}{'reference':>30}{auroc_error(errors, msp):>8.3f}{auprc_error(errors, msp):>8.3f}")
        files = {"Proposed head": f"failure_head_scaled_val_predictions_seed{seed}.csv",
                 "Logits-only head": f"failure_head_scaled_logits_only_val_predictions_seed{seed}.csv",
                 "Features-only head": f"failure_head_scaled_features_only_val_predictions_seed{seed}.csv"}
        for name, filename in files.items():
            path = RESULTS_DIR / filename
            if not path.is_file():
                continue
            scores = pd.read_csv(path)
            if len(scores) != len(errors) or not np.array_equal(scores["failure_label"].to_numpy(), errors):
                ok &= _check(f"{name}: rows aligned with the validation views", False)
                continue
            s = scores["failure_score"].to_numpy()
            d, lo, hi = delta_aurc(errors, s, msp, clusters, resamples=500, seed=seed)
            print(f"    {name:<26}{aurc(errors, s):>8.4f}{f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]':>30}{auroc_error(errors, s):>8.3f}{auprc_error(errors, s):>8.3f}")
    print("\n  Compare the MSP rows with the earlier results: seed 42 AUPRC 0.336 and AUROC 0.960,")
    print("  seed 123 AUPRC 0.467 and AUROC 0.978, seed 2026 AUPRC 0.372 and AUROC 0.975.")
    return ok


if __name__ == "__main__":
    passed = _toy_checks()
    passed &= _real_data_check()
    print("\n" + ("SELF-TEST FINISHED: the toy checks passed." if passed else "SELF-TEST: some checks FAILED."))