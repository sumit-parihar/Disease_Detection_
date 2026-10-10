"""
dataset.py  -  Step 4: loads the images and turns them into numbers the neural
network can read (tensors). Used by every training and evaluation script.

WHAT THIS FILE DOES
-------------------
1. Reads which image belongs to which subset from splits/split_assignments.csv
   (made by make_splits.py) and gives you the images of ONE subset at a time.
2. Applies the image preparation ("transform") described in the paper.
3. Gives the images to the model in small groups called batches.

THE TWO TRANSFORMS
------------------
Both transforms first do the same preparation:
    open image -> make it RGB -> resize to 224 x 224
    -> scale pixel values from 0..255 to 0..1
    -> subtract the ImageNet mean and divide by the ImageNet std, per colour.
The last step is needed because the MobileNetV3-Small we start from was
pretrained on ImageNet with exactly this scaling.

  eval transform  : only the preparation above. Used for s_val, s_reliability,
                    s_calibration, s_test and for the target (PP2020).
  train transform : the preparation plus the mild augmentation of the paper
                    (Section 3.8): horizontal flip, rotation up to +/-15
                    degrees, zoom +/-10%, contrast +/-10%. ONLY for s_train.
                    The paper says augmentation is never applied to
                    validation, calibration or test sets, so the code refuses
                    to use it on any other subset.

CHOICES THE PAPER DOES NOT SPECIFY (made here, before any target result)
------------------------------------------------------------------------
  * Resizing: the whole image is resized straight to 224 x 224, for BOTH
    datasets, with the same rule. Source images are square (256 x 256); the
    target images are not, so they get squeezed a little. The other options
    (cutting out the centre, adding borders) would remove parts of the leaf
    or add fake borders. Whatever is chosen, it must be the same for source and
    target and must not be changed after seeing target results.
  * Flip probability 0.5 (standard for "horizontal flips").
  * Rotated-in corners are filled with black.
  * No automatic correction of photo orientation (EXIF) is applied, for either
    dataset.
  * Normalisation uses the ImageNet mean/std (PyTorch convention for
    pretrained MobileNetV3).

THE TARGET DATASET AND THE SEAL
-------------------------------
TargetDataset gives only the image and its path. It never gives labels. The
paper says target labels may be used only after predictions are frozen and
saved, so labels will be read by the final evaluation script, not here.

HOW TO CHECK IT (project root, venv active):
    python src/dataset.py          quick check (a few images per subset) and a
                                   GPU memory test with batch size 32
    python src/dataset.py --full   opens every image of every subset

GPU TEST
--------
If a GPU is found, the check also sends one augmented batch (config.BATCH_SIZE
= 32) through an UNTRAINED MobileNetV3-Small, forwards and backwards, and
prints the peak GPU memory. This is only a memory test: nothing is trained or
saved, and no pretrained weights are downloaded.
"""

import argparse
import sys

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    IMAGE_SIZE,
    MANIFEST_DIR,
    PROJECT_ROOT,
    SEED,
    SPLIT_PROPORTIONS,
    SPLITS_DIR,
)

SPLIT_FILE = SPLITS_DIR / "split_assignments.csv"
MANIFEST_FILE = MANIFEST_DIR / "manifest.csv"
SUBSETS = list(SPLIT_PROPORTIONS.keys())
TRAIN_SUBSET = "s_train"          # the only subset that may be augmented

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentation limits from the paper (Section 3.8)
FLIP_PROBABILITY = 0.5            # not given in the paper (standard value)
ROTATION_DEGREES = 15
ZOOM_RANGE = (0.9, 1.1)           # zoom up to +/-10%
CONTRAST_RANGE = 0.1              # contrast factor 0.9 to 1.1


# ---------------------------------------------------------------------------
# PART 1: transforms
# ---------------------------------------------------------------------------
def _preparation():
    """Steps shared by every transform (after any augmentation)."""
    return [
        T.ToTensor(),                                  # 0..255 -> 0..1, shape 3 x H x W
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def build_eval_transform():
    """Resize to 224 x 224, scale, normalise. No randomness."""
    return T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE))] + _preparation())


def build_train_transform():
    """Same as the eval transform, plus the mild random augmentation."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=FLIP_PROBABILITY),
        T.RandomAffine(degrees=ROTATION_DEGREES, scale=ZOOM_RANGE, fill=0),
        T.ColorJitter(contrast=CONTRAST_RANGE),
    ] + _preparation())


def load_image(relative_path):
    """Open an image from its path in the manifest (relative to the project
    root) and make sure it has 3 colour channels."""
    return Image.open(PROJECT_ROOT / relative_path).convert("RGB")


# ---------------------------------------------------------------------------
# PART 2: datasets
# ---------------------------------------------------------------------------
class SourceDataset(Dataset):
    """The PlantVillage images of ONE subset.
    Each item is (image tensor, class index 0/1/2, image path)."""

    def __init__(self, subset, augment=False):
        if subset not in SUBSETS:
            raise ValueError(f"Unknown subset '{subset}'. Choose from {SUBSETS}.")
        if augment and subset != TRAIN_SUBSET:
            raise ValueError("Augmentation is only allowed on s_train "
                             "(paper: not on validation, calibration or test sets).")
        if not SPLIT_FILE.exists():
            sys.exit(f"{SPLIT_FILE} not found. Run make_splits.py first.")

        table = pd.read_csv(SPLIT_FILE)
        table = table[table["subset"] == subset].sort_values("path").reset_index(drop=True)
        self.subset = subset
        self.paths = table["path"].tolist()
        self.label_names = table["label"].tolist()
        self.labels = [CLASS_TO_INDEX[name] for name in self.label_names]
        self.transform = build_train_transform() if augment else build_eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = self.transform(load_image(self.paths[index]))
        return image, self.labels[index], self.paths[index]


class TargetDataset(Dataset):
    """All labelled PP2020 train images (1,821) for the frozen pipeline.
    Each item is (image tensor, image path). NO labels are returned: the paper
    keeps target labels sealed until predictions are frozen and saved."""

    def __init__(self):
        if not MANIFEST_FILE.exists():
            sys.exit(f"{MANIFEST_FILE} not found. Run audit_dataset.py first.")
        manifest = pd.read_csv(MANIFEST_FILE, keep_default_na=False)
        manifest["decode_ok"] = manifest["decode_ok"].astype(str) == "True"
        target = manifest[(manifest["dataset"] == "target") & manifest["decode_ok"]]
        self.paths = sorted(target["path"].tolist())
        self.transform = build_eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_image(self.paths[index])), self.paths[index]


# ---------------------------------------------------------------------------
# PART 3: loaders (batches)
# ---------------------------------------------------------------------------
def make_loader(dataset, batch_size, shuffle=False, num_workers=0, seed=SEED):
    """Wrap a dataset so the model receives batches.
    batch_size has no default on purpose: the paper does not fix it, so it is
    chosen later, using source data only.
    shuffle=True mixes the order each epoch (use it only for training); the
    mixing is seeded so a run can be repeated.
    num_workers=0 loads images in the main process (safest on Windows)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def get_source_loader(subset, batch_size, train=False, num_workers=0):
    """Loader for one source subset. train=True means: augmentation + shuffling
    (allowed for s_train only)."""
    dataset = SourceDataset(subset, augment=train)
    return make_loader(dataset, batch_size, shuffle=train, num_workers=num_workers)


# ---------------------------------------------------------------------------
# PART 4: self-check
# ---------------------------------------------------------------------------
def check_tensor(image):
    ok = tuple(image.shape) == (3, IMAGE_SIZE, IMAGE_SIZE) and image.dtype == torch.float32
    return ok and bool(torch.isfinite(image).all())


def gpu_check(batch_size):
    """Send one augmented batch through an untrained MobileNetV3-Small on the GPU
    (forward + backward) to confirm the batch size fits in GPU memory."""
    print("\nGPU test:")
    if not torch.cuda.is_available():
        print("  No GPU found. PyTorch will use the CPU (slow).")
        print("  Usually this means the CPU-only PyTorch was installed. Install the")
        print("  CUDA build from pytorch.org for your NVIDIA GPU.")
        return True
    from torchvision.models import mobilenet_v3_small

    device = torch.device("cuda")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")
    try:
        model = mobilenet_v3_small(weights=None).to(device)   # untrained, no download
        model.train()
        loader = make_loader(SourceDataset(TRAIN_SUBSET, augment=True), batch_size, shuffle=True)
        images, _, _ = next(iter(loader))
        images = images.to(device)
        torch.cuda.reset_peak_memory_stats()
        model(images).float().mean().backward()               # throwaway pass
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  Batch of {len(images)}: forward + backward OK, "
              f"peak GPU memory {peak_gb:.2f} GB of {total_gb:.1f} GB")
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"  CUDA out of memory with batch size {batch_size}. "
              "Set BATCH_SIZE = 16 in config.py and run again.")
        return False


def self_check(full):
    print("Checking the data loader...\n")
    table = pd.read_csv(SPLIT_FILE) if SPLIT_FILE.exists() else sys.exit(
        f"{SPLIT_FILE} not found. Run make_splits.py first.")
    all_ok = True

    print(f"{'subset':<15}{'images':>8}{'in file':>9}   classes (healthy / scab / rust)")
    for subset in SUBSETS:
        dataset = SourceDataset(subset)
        in_file = int((table["subset"] == subset).sum())
        counts = [dataset.labels.count(CLASS_TO_INDEX[name]) for name in CLASS_NAMES]
        status = "OK" if len(dataset) == in_file else "MISMATCH"
        all_ok &= status == "OK"
        print(f"{subset:<15}{len(dataset):>8}{in_file:>9}   {counts}  {status}")

        indices = range(len(dataset)) if full else range(min(4, len(dataset)))
        for index in indices:
            image, label, path = dataset[index]
            if not check_tensor(image):
                print(f"   PROBLEM with {path}: shape {tuple(image.shape)}")
                all_ok = False

    # the training transform (augmentation) on a few s_train images
    train_set = SourceDataset(TRAIN_SUBSET, augment=True)
    batch_images, batch_labels, _ = next(iter(make_loader(train_set, batch_size=4, shuffle=True)))
    train_ok = tuple(batch_images.shape) == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    all_ok &= train_ok
    print(f"\nTraining batch (augmented, shuffled): shape {tuple(batch_images.shape)}, "
          f"labels {batch_labels.tolist()}  {'OK' if train_ok else 'PROBLEM'}")

    # augmentation must be refused on other subsets
    try:
        SourceDataset("s_test", augment=True)
        print("PROBLEM: augmentation was allowed on s_test")
        all_ok = False
    except ValueError:
        print("Augmentation on s_test is refused  OK")

    # the target dataset: only images, no labels
    target = TargetDataset()
    item = target[0]
    target_ok = len(item) == 2 and check_tensor(item[0])
    if full:
        for index in range(len(target)):
            target_ok &= check_tensor(target[index][0])
    all_ok &= target_ok
    print(f"Target dataset: {len(target)} images, each item = (image, path), no labels  "
          f"{'OK' if target_ok else 'PROBLEM'}")

    all_ok &= gpu_check(BATCH_SIZE)

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED, send this output for help"))
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="open every image of every subset (slower)")
    arguments = parser.parse_args()
    sys.exit(0 if self_check(arguments.full) else 1)