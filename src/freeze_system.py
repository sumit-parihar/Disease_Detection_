"""
freeze_system.py  -  Freezes the whole system before any S-test or target image is
evaluated. It records a SHA-256 hash of every file that defines the system, so
that nothing can change quietly afterwards.

WHAT GETS FROZEN
----------------
  * the 3 disease classifiers (one per seed)
  * the 9 failure heads (3 per seed: proposed, logits-only, features-only)
  * the calibration settings (temperatures and the tau80 / tau90 cut-offs) and their tables
  * the split file and the master manifest
  * the code that defines the pipeline and the measures
  * the Git commit, whether there are uncommitted changes, and the library versions

MODES
-----
  python src/freeze_system.py            create the freeze record (run it once)
  python src/freeze_system.py --verify   check that every frozen file is unchanged

Later scripts (S-test and target evaluation) call verify_frozen() and stop if anything
differs from the record.

BEFORE IT FREEZES, IT CHECKS
----------------------------
  * every file above exists
  * the calibration settings are complete (3 seeds x 7 methods x tau80 and tau90)
  * no S-test or target result files exist yet (the freeze must come BEFORE them)

OUTPUT (in results/)
--------------------
  frozen_system_manifest.json         the record
  frozen_system_manifest.json.sha256  hash of the record

The freeze record is meant to stay fixed: the script refuses to overwrite it.
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys

from config import MANIFEST_DIR, MODELS_DIR, PROJECT_ROOT, RESULTS_DIR, SPLITS_DIR, make_folders
from frozen_pipeline import HEAD_VARIANTS, classifier_path, head_path, sha256_file
from train_failure_head import CLASSIFIER_SEEDS

FREEZE_FILE = RESULTS_DIR / "frozen_system_manifest.json"
FREEZE_SIDECAR = RESULTS_DIR / "frozen_system_manifest.json.sha256"

CODE_FILES = ["config.py", "dataset.py", "model.py", "frozen_pipeline.py", "failure_head_diagnostic.py",
              "train_failure_head.py", "retrain_failure_head.py", "train_ablation_heads.py",
              "extract_calibration_outputs.py", "calibrate_and_thresholds.py", "metrics.py"]
METHODS = ["B1 MSP", "B2 Entropy", "B3 Margin", "B4 Temp-scaled MSP", "B5 Logits-only head",
           "A1 Features-only head", "P1 Proposed head"]


def relative(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


def frozen_files():
    files = [SPLITS_DIR / "split_assignments.csv", MANIFEST_DIR / "manifest.csv",
             RESULTS_DIR / "frozen_calibration_settings.json", RESULTS_DIR / "temperature_scaling.csv",
             RESULTS_DIR / "calibration_thresholds.csv"]
    for seed in CLASSIFIER_SEEDS:
        files.append(classifier_path(seed))
        files += [head_path(variant, seed) for variant in HEAD_VARIANTS]
        files.append(RESULTS_DIR / f"s_calibration_outputs_seed{seed}.csv")
    files += [PROJECT_ROOT / "src" / name for name in CODE_FILES]
    return files


def git_state():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    if not commit or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        commit = None                                   # no repository, or no commit yet
    status = run("status", "--porcelain")
    return {"commit": commit or "not available",
            "uncommitted_changes": (None if status is None else bool(status)),
            "uncommitted_file_count": (None if status is None else len(status.splitlines()))}


def library_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "numpy", "pandas", "sklearn", "PIL"):
        try:
            module = __import__(name)
            versions[name] = str(module.__version__)
        except Exception:
            versions[name] = "not available"
    return versions


def preflight():
    problems = []
    for path in frozen_files():
        if not path.is_file():
            problems.append(f"missing file: {relative(path)}")
    settings_path = RESULTS_DIR / "frozen_calibration_settings.json"
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        for seed in CLASSIFIER_SEEDS:
            entry = settings.get(str(seed))
            if not entry or "temperature" not in entry:
                problems.append(f"calibration settings: no temperature for seed {seed}")
                continue
            for method in METHODS:
                taus = entry.get("thresholds", {}).get(method, {})
                if "tau80" not in taus or "tau90" not in taus:
                    problems.append(f"calibration settings: seed {seed}, '{method}' lacks tau80 or tau90")
    for pattern in ("s_test_*", "target_*", "*_target_*"):
        found = sorted(RESULTS_DIR.glob(pattern))
        if found:
            problems.append(f"result files already exist ({found[0].name}); the freeze must come BEFORE S-test or target results")
    return problems


def verify_frozen(raise_on_change=True):
    """Compare every frozen file with the freeze record. Returns a list of differences."""
    if not FREEZE_FILE.is_file():
        raise FileNotFoundError(f"{FREEZE_FILE.name} not found. Run freeze_system.py first.")
    if FREEZE_SIDECAR.is_file():
        recorded = FREEZE_SIDECAR.read_text(encoding="utf-8").split()[0]
        if sha256_file(FREEZE_FILE) != recorded:
            raise ValueError("The freeze record itself has been changed (its hash no longer matches).")
    record = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    differences = []
    for rel, expected in record["files"].items():
        path = PROJECT_ROOT / rel
        if not path.is_file():
            differences.append(f"missing: {rel}")
        elif sha256_file(path) != expected:
            differences.append(f"changed: {rel}")
    if differences and raise_on_change:
        raise ValueError("The frozen system has changed:\n  " + "\n  ".join(differences))
    return differences


def create_freeze():
    problems = preflight()
    if problems:
        print("The system cannot be frozen yet:")
        for problem in problems:
            print("  -", problem)
        sys.exit(1)
    git = git_state()
    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Freeze of the full system before any S-test or target evaluation",
        "classifier_seeds": list(CLASSIFIER_SEEDS),
        "git": git,
        "libraries": library_versions(),
        "files": {relative(path): sha256_file(path) for path in frozen_files()},
    }
    FREEZE_FILE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    FREEZE_SIDECAR.write_text(f"{sha256_file(FREEZE_FILE)}  {FREEZE_FILE.name}\n", encoding="utf-8")
    print(f"Frozen {len(record['files'])} files.")
    print(f"Git commit: {git['commit']}")
    if git["uncommitted_changes"]:
        print(f"WARNING: {git['uncommitted_file_count']} file(s) have uncommitted changes. Commit them, and note that the freeze")
        print("         records the files as they are on disk now, not as they are in Git.")
    print(f"Freeze record: {relative(FREEZE_FILE)}")
    print(f"Its SHA-256  : {sha256_file(FREEZE_FILE)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check that the frozen files are unchanged")
    args = parser.parse_args()
    make_folders()
    if args.verify:
        differences = verify_frozen(raise_on_change=False)
        if differences:
            print("The frozen system has CHANGED:")
            for line in differences:
                print("  -", line)
            sys.exit(1)
        print("OK: every frozen file matches the freeze record.")
        return
    if FREEZE_FILE.exists():
        sys.exit(f"{FREEZE_FILE.name} already exists. A freeze is meant to stay fixed. Use --verify to check it.")
    create_freeze()


if __name__ == "__main__":
    main()