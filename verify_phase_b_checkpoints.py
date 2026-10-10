from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import DiseaseClassifier
from config import IMAGE_SIZE, FEATURE_DIM, NUM_CLASSES

checkpoints = {
    42: ROOT / "models" / "disease_seed42_phaseB.pt",
    123: ROOT / "models" / "disease_seed123_phaseB.pt",
    2026: ROOT / "models" / "disease_seed2026_phaseB.pt",
}

print(f"PyTorch version: {torch.__version__}")
print("Verifying Phase B disease-classifier checkpoints...\n")

for seed, path in checkpoints.items():
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    if checkpoint.get("seed") != seed:
        raise ValueError(
            f"{path.name}: expected seed {seed}, "
            f"found {checkpoint.get('seed')}"
        )

    if checkpoint.get("phase") != "B":
        raise ValueError(
            f"{path.name}: expected Phase B, "
            f"found {checkpoint.get('phase')}"
        )

    model = DiseaseClassifier(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    sample = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.inference_mode():
        features = model.extract_features(sample)
        logits = model(sample)

    assert features.shape == (1, FEATURE_DIM), features.shape
    assert logits.shape == (1, NUM_CLASSES), logits.shape
    assert torch.isfinite(features).all(), "Non-finite features detected"
    assert torch.isfinite(logits).all(), "Non-finite logits detected"

    print(
        f"Seed {seed}: PASS | "
        f"Phase {checkpoint['phase']} | "
        f"Features: {tuple(features.shape)} | "
        f"Logits: {tuple(logits.shape)}"
    )

print("\nAll three Phase B checkpoints passed verification.")
