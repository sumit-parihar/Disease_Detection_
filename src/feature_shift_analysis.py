"""
feature_shift_analysis.py  -  EXPLORATORY analysis (not part of the prespecified protocol).

QUESTION
--------
On the target, the failure heads that use the 576 image numbers were no better than MSP, while
the head that sees only the classifier's 3 scores was clearly better. Do the field images look
very different from the lab images in that 576-number image description?

WHAT IT DOES (all numbers already exist; no model is run, nothing is trained)
-----------------------------------------------------------------------------
For each classifier seed:
  1. Takes the 576 image numbers of the 254 S-calibration images (the source reference) and of
     the 1,730 target images (from the saved .npz files).
  2. Standardises both with the source mean and standard deviation of every one of the 576 numbers.
  3. Measures how far each image is from the source:
       - centre distance: distance to the source mean, divided by sqrt(576)
         (a source image typically has a value near 1)
       - nearest-source distance: distance to the closest source image
         (for a source image: the closest OTHER source image)
  4. Counts how many of the 576 numbers have a target average more than 1 source standard
     deviation away from the source average.
  5. Asks whether those distances predict which target predictions are wrong (AUROC), and
     shows next to them the AUROC of MSP, B5 and P1 from results/target_reliability.csv.
  6. Projects source and target onto the first 2 principal components of the source images
     (for a scatter plot).

IMPORTANT LIMITS
----------------
  * This analysis was decided AFTER the target results were known. It is exploratory and must be
    labelled that way in the paper. It must not be used to change anything.
  * The source reference is S-calibration (254 images). Source distances are measured against the
    source mean computed from the same images, so they are slightly optimistic (small).

OUTPUT FILES
------------
  results/feature_shift_summary.csv, results/feature_shift_summary.txt
  figure_data/feature_shift_pca.csv   (seed, domain, pc1, pc2, correct; no image names)

HOW TO RUN (project root, venv active):
    python src/feature_shift_analysis.py
"""

import argparse
import sys

import numpy as np
import pandas as pd

from config import CLASS_NAMES, MANIFEST_DIR, PROJECT_ROOT, RESULTS_DIR

SEEDS = (42, 123, 2026)
OUT_CSV = RESULTS_DIR / "feature_shift_summary.csv"
OUT_TXT = RESULTS_DIR / "feature_shift_summary.txt"
OUT_PCA = PROJECT_ROOT / "figure_data" / "feature_shift_pca.csv"


def auroc(positive_flags, score):
    """Chance that a random positive has a higher score than a random negative (ties count half)."""
    flags = np.asarray(positive_flags).astype(bool)
    n_pos, n_neg = int(flags.sum()), int((~flags).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[flags].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def nearest_distance(a, b, exclude_self=False):
    """Euclidean distance from every row of a to its nearest row of b."""
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    d2 = np.maximum(d2, 0.0)
    if exclude_self:
        np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min(axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (OUT_CSV.exists() or OUT_TXT.exists()) and not args.overwrite:
        sys.exit("feature_shift_summary files already exist. Use --overwrite to rebuild them.")
    OUT_PCA.parent.mkdir(exist_ok=True)

    manifest = pd.read_csv(MANIFEST_DIR / "manifest.csv", keep_default_na=False)
    labels = manifest[manifest["dataset"] == "target"].set_index("path")["label"]
    class_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    reliability = pd.read_csv(RESULTS_DIR / "target_reliability.csv")

    rows, pca_rows = [], []
    for seed in SEEDS:
        source = np.load(RESULTS_DIR / f"s_calibration_features_seed{seed}.npz")["features"].astype(np.float64)
        target_file = np.load(RESULTS_DIR / f"target_features_seed{seed}.npz")
        target = target_file["features"].astype(np.float64)
        predictions = pd.read_csv(RESULTS_DIR / f"target_predictions_seed{seed}.csv", keep_default_na=False)
        if list(target_file["paths"]) != predictions["path"].tolist():
            sys.exit(f"Seed {seed}: feature file and prediction file are not in the same order.")
        true_index = np.array([class_index[labels[p]] for p in predictions["path"]])
        wrong = (predictions["predicted_index"].to_numpy() != true_index)

        mean, std = source.mean(axis=0), source.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        zs, zt = (source - mean) / std, (target - mean) / std
        dim = source.shape[1]

        centre_s, centre_t = np.linalg.norm(zs, axis=1) / np.sqrt(dim), np.linalg.norm(zt, axis=1) / np.sqrt(dim)
        nn_s, nn_t = nearest_distance(zs, zs, exclude_self=True), nearest_distance(zt, zs)
        shifted = float((np.abs(zt.mean(axis=0)) > 1.0).mean())

        reference = reliability[reliability["classifier_seed"] == seed].set_index("method")["auroc_error"]
        rows.append({
            "classifier_seed": seed, "source_images": len(zs), "target_images": len(zt),
            "centre_distance_source_median": float(np.median(centre_s)), "centre_distance_source_p95": float(np.percentile(centre_s, 95)),
            "centre_distance_target_median": float(np.median(centre_t)), "centre_distance_target_p5": float(np.percentile(centre_t, 5)),
            "target_images_beyond_source_p95_centre": float((centre_t > np.percentile(centre_s, 95)).mean()),
            "nearest_source_distance_source_median": float(np.median(nn_s)), "nearest_source_distance_target_median": float(np.median(nn_t)),
            "target_images_beyond_source_p95_nearest": float((nn_t > np.percentile(nn_s, 95)).mean()),
            "features_shifted_more_than_1sd": shifted,
            "auroc_error_centre_distance": auroc(wrong, centre_t), "auroc_error_nearest_distance": auroc(wrong, nn_t),
            "auroc_error_MSP": float(reference["B1 MSP"]), "auroc_error_B5": float(reference["B5 Logits-only head"]),
            "auroc_error_P1": float(reference["P1 Proposed head"]),
        })

        centred = zs - zs.mean(axis=0)
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        components = vt[:2].T
        rows[-1]["source_variance_in_first_2_pcs"] = float((singular[:2] ** 2).sum() / (singular ** 2).sum())
        for domain, z, flag in (("source", zs, None), ("target", zt, (~wrong).astype(int))):
            p = (z - zs.mean(axis=0)) @ components
            for i in range(len(p)):
                pca_rows.append({"seed": seed, "domain": domain, "pc1": float(p[i, 0]), "pc2": float(p[i, 1]),
                                 "correct": "" if flag is None else int(flag[i])})
        print(f"  seed {seed}: done")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False, lineterminator="\n")
    pd.DataFrame(pca_rows).to_csv(OUT_PCA, index=False, lineterminator="\n")

    L = ["FEATURE-SHIFT ANALYSIS (EXPLORATORY: decided after the target results were known)", "=" * 80, "",
         "Source reference: 254 S-calibration images. Target: 1,730 images. 576 image numbers per image,",
         "standardised with the source mean and standard deviation.", ""]
    L.append(f"{'seed':<7}{'centre dist. source':>21}{'centre dist. target':>21}{'beyond source p95':>19}{'nearest dist. source':>22}{'nearest dist. target':>22}{'shifted >1 SD':>15}")
    for _, r in summary.iterrows():
        L.append(f"{int(r.classifier_seed):<7}{r.centre_distance_source_median:>21.2f}{r.centre_distance_target_median:>21.2f}"
                 f"{100 * r.target_images_beyond_source_p95_centre:>18.1f}%{r.nearest_source_distance_source_median:>22.2f}"
                 f"{r.nearest_source_distance_target_median:>22.2f}{100 * r.features_shifted_more_than_1sd:>14.1f}%")
    L += ["", "Medians; 'beyond source p95' = share of target images farther from the source mean than 95% of source images.", "",
          "CAN THE DISTANCE PREDICT WRONG PREDICTIONS? (AUROC on the target, 0.5 = no better than guessing)",
          f"{'seed':<7}{'centre dist.':>14}{'nearest dist.':>15}{'MSP':>8}{'B5 (scores only)':>19}{'P1 (proposed)':>16}"]
    for _, r in summary.iterrows():
        L.append(f"{int(r.classifier_seed):<7}{r.auroc_error_centre_distance:>14.3f}{r.auroc_error_nearest_distance:>15.3f}"
                 f"{r.auroc_error_MSP:>8.3f}{r.auroc_error_B5:>19.3f}{r.auroc_error_P1:>16.3f}")
    L += ["", "This analysis is exploratory. It does not change any result and must be labelled exploratory in the paper."]
    OUT_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"\nSaved {OUT_CSV.name}, {OUT_TXT.name} and {OUT_PCA.name}")


if __name__ == "__main__":
    main()