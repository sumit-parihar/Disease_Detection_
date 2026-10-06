"""
config.py  -  Single place for every fixed value used in the project.

WHAT THIS FILE DOES
-------------------
It does not run any experiment. It only stores constants (folder paths, class
names, random seed, split proportions, image size) that every other file will
import. Example in other files:  from config import SEED, SOURCE_CLASSES

WHY WE NEED IT
--------------
1. Reproducibility: the paper says the split is "reproducibly seeded". One seed
   stored here means every file uses the same seed.
2. No hidden changes: if a value changes, it changes in one place only.
3. Only values that are written in the paper are stored here. Values the paper
   does not give (learning rate, batch size, epochs) are NOT here yet. We will
   decide them later using source data only, never target data.

All paths are built from the project root, so the project works from any
folder, for example D:\\Capstone Project.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
# This file lives in <root>/src/config.py, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data: downloaded once and never modified (paper, Section 3.3).
DATA_DIR = PROJECT_ROOT / "data"
PLANTVILLAGE_DIR = DATA_DIR / "plantvillage" / "raw" / "color"   # source
PP2020_DIR = DATA_DIR / "pp2020"                                  # target
PP2020_CSV = PP2020_DIR / "train.csv"                             # target labels

# Generated outputs (these are created by our scripts, not downloaded).
MANIFEST_DIR = PROJECT_ROOT / "manifests"   # audit results, file hashes
SPLITS_DIR = PROJECT_ROOT / "splits"        # which image is in which subset
MODELS_DIR = PROJECT_ROOT / "models"        # trained weights
RESULTS_DIR = PROJECT_ROOT / "results"      # tables and figures

# ---------------------------------------------------------------------------
# 2. REPRODUCIBILITY
# ---------------------------------------------------------------------------
# One fixed seed so that random choices (splits, weight init) can be repeated.
# The paper says "reproducibly seeded" but does not give a number, so 42 is
# our own choice. We fix it now, before seeing any result, and never change it.
SEED = 42

# ---------------------------------------------------------------------------
# 3. CLASSES (paper, Section 3.2)
# ---------------------------------------------------------------------------
# The three disease states shared by both datasets. The ORDER matters: the
# model outputs 3 numbers, and index 0/1/2 must always mean the same class.
CLASS_NAMES = ["healthy", "scab", "rust"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# PlantVillage folder name for each class. Black rot is left out on purpose:
# the target dataset has no black rot class (paper, Section 3.2).
SOURCE_FOLDERS = {
    "healthy": "Apple___healthy",
    "scab": "Apple___Apple_scab",
    "rust": "Apple___Cedar_apple_rust",
}

# Published counts quoted in the paper. These are only EXPECTED values; the
# audit step will count the real files and compare.
EXPECTED_SOURCE_COUNTS = {"healthy": 1645, "scab": 630, "rust": 275}  # 2,550
EXPECTED_TARGET_COUNTS = {"healthy": 516, "scab": 592, "rust": 622}   # 1,730
EXPECTED_TARGET_MULTIPLE = 91   # excluded from the primary analysis only

# Plant Pathology 2020 stores labels as 4 yes/no columns in train.csv.
# The first three match our classes; "multiple_diseases" has no match.
TARGET_LABEL_COLUMNS = ["healthy", "scab", "rust"]
TARGET_MULTIPLE_COLUMN = "multiple_diseases"

# ---------------------------------------------------------------------------
# 4. SOURCE SPLIT PROPORTIONS (paper, Section 3.4)
# ---------------------------------------------------------------------------
# PlantVillage images are divided into 5 subsets. Each has one job.
SPLIT_PROPORTIONS = {
    "s_train": 0.50,         # fit the disease classifier
    "s_val": 0.10,           # model selection and early stopping
    "s_reliability": 0.15,   # train and validate the failure-prediction head
    "s_calibration": 0.10,   # temperature scaling and source-only thresholds
    "s_test": 0.15,          # untouched in-domain reference
}

# ---------------------------------------------------------------------------
# 5. MODEL SETTINGS (paper, Sections 3.5 and 3.6)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224            # images are resized to 224 x 224, 3 colour channels
NUM_CLASSES = 3
FEATURE_DIM = 576           # MobileNetV3-Small features after global pooling
DISEASE_DROPOUT = 0.20      # dropout before the 3-unit disease layer
FAILURE_INPUT_DIM = FEATURE_DIM + NUM_CLASSES   # 576 + 3 = 579
FAILURE_HIDDEN_UNITS = 64
FAILURE_DROPOUT = 0.30
FAILURE_L2 = 1e-4
FAILURE_POS_WEIGHT_CAP = 5  # cap on the error-class weight in the loss

# Source-selected coverage levels used to test threshold transfer (RQ4).
COVERAGE_LEVELS = [0.80, 0.90]


def make_folders():
    """Create the output folders if they do not exist yet.

    Raw data folders are NOT created here, because raw data must come from
    the download scripts and must never be changed afterwards.
    """
    for folder in (MANIFEST_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Running "python src/config.py" creates the folders and prints the paths,
    # so you can check that the project root is correct.
    make_folders()
    print("Project root :", PROJECT_ROOT)
    print("Source data  :", PLANTVILLAGE_DIR)
    print("Target data  :", PP2020_DIR)
    print("Folders ready: manifests, splits, models, results")