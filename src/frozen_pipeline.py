"""
frozen_pipeline.py  -  Shared tools to run the FROZEN system on a set of images:
  frozen disease classifier  ->  3 scores + 576 image numbers
  frozen failure heads       ->  "chance this answer is wrong" scores

Nothing in this file trains or changes a model. Other scripts import it, so that
S-calibration, S-test and the target images are all processed in exactly the same
way.

WHAT IT CHECKS BEFORE USING A MODEL
-----------------------------------
  * Disease classifier: the checkpoint file must have the same SHA-256 as the one
    recorded when the reliability data were created
    (results/reliability_generation_hashes_v2.txt), and its seed, phase and class
    order must match. If anything differs, it stops.
  * Failure head: the checkpoint must be for the right classifier seed and the
    right inputs, and must load into the head design without mismatch.

THE THREE FAILURE HEADS
-----------------------
  proposed       image numbers + 3 scores (579 inputs)   models/failure_head_scaled_seed{seed}.pt
  logits_only    3 scores only                           models/failure_head_scaled_logits_only_seed{seed}.pt
  features_only  576 image numbers only                  models/failure_head_scaled_features_only_seed{seed}.pt
Each head standardises its inputs with the mean and standard deviation saved in its
own checkpoint (computed from the reliability-training views).
"""

import hashlib

import numpy as np
import torch

from config import CLASS_NAMES, MODELS_DIR, RESULTS_DIR
from failure_head_diagnostic import SmallHead
from model import DiseaseClassifier

HEAD_VARIANTS = ("proposed", "logits_only", "features_only")
CLASSIFIER_HASH_FILE = RESULTS_DIR / "reliability_generation_hashes_v2.txt"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classifier_path(seed):
    return MODELS_DIR / f"disease_seed{seed}_phaseB.pt"


def head_path(variant, seed):
    if variant == "proposed":
        return MODELS_DIR / f"failure_head_scaled_seed{seed}.pt"
    return MODELS_DIR / f"failure_head_scaled_{variant}_seed{seed}.pt"


def recorded_classifier_hash(seed):
    """The hash written when the reliability data were created."""
    if not CLASSIFIER_HASH_FILE.is_file():
        raise FileNotFoundError(f"{CLASSIFIER_HASH_FILE} not found, so the classifier cannot be verified.")
    wanted = f"models/disease_seed{seed}_phaseB.pt"
    for line in CLASSIFIER_HASH_FILE.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == wanted:
            return parts[0]
    raise ValueError(f"No recorded hash for {wanted} in {CLASSIFIER_HASH_FILE.name}.")


def load_frozen_classifier(seed, device):
    path = classifier_path(seed)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
    actual, recorded = sha256_file(path), recorded_classifier_hash(seed)
    if actual != recorded:
        raise ValueError(f"{path.name} does not match the recorded SHA-256.\n  file    : {actual}\n  recorded: {recorded}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != seed or checkpoint.get("phase") != "B":
        raise ValueError(f"{path.name}: metadata mismatch (seed={checkpoint.get('seed')}, phase={checkpoint.get('phase')}).")
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError(f"{path.name}: class order {checkpoint.get('class_names')} differs from {CLASS_NAMES}.")
    model = DiseaseClassifier(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def load_frozen_head(variant, seed, device):
    """Return (head, input_mean, input_std) for one failure-head variant."""
    path = head_path(variant, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Failure-head checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("classifier_seed") != seed:
        raise ValueError(f"{path.name}: belongs to classifier seed {checkpoint.get('classifier_seed')}, expected {seed}.")
    if variant != "proposed" and checkpoint.get("inputs") != variant:
        raise ValueError(f"{path.name}: inputs are '{checkpoint.get('inputs')}', expected '{variant}'.")
    mean, std = checkpoint["input_mean"].float(), checkpoint["input_std"].float()
    head = SmallHead(int(mean.numel()))
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()
    head.requires_grad_(False)
    return head.to(device), mean.to(device), std.to(device)


def head_input(variant, features, logits):
    if variant == "proposed":
        return torch.cat([features, logits], dim=1)
    return logits if variant == "logits_only" else features


@torch.inference_mode()
def run_images(loader, classifier, heads, device):
    """Run every image of a loader through the frozen classifier and heads.
    The loader may return (image, label, path) or (image, path).
    heads: {variant: (head, mean, std)}."""
    paths, labels, feature_list, logit_list = [], [], [], []
    scores = {variant: [] for variant in heads}
    for batch in loader:
        if len(batch) == 3:
            images, batch_labels, batch_paths = batch
            labels.extend(int(x) for x in batch_labels)
        else:
            images, batch_paths = batch
        paths.extend(batch_paths)
        features = classifier.extract_features(images.to(device))
        logits = classifier.fc(classifier.dropout(features))
        feature_list.append(features.cpu().numpy())
        logit_list.append(logits.cpu().numpy())
        for variant, (head, mean, std) in heads.items():
            x = (head_input(variant, features, logits) - mean) / std
            scores[variant].append(torch.sigmoid(head(x)).cpu().numpy())
    return {
        "paths": np.asarray(paths, dtype=np.str_),
        "label_index": np.asarray(labels, dtype=np.int64) if labels else None,
        "features": np.concatenate(feature_list).astype(np.float32),
        "logits": np.concatenate(logit_list).astype(np.float32),
        "head_scores": {v: np.concatenate(s).astype(np.float64) for v, s in scores.items()},
    }