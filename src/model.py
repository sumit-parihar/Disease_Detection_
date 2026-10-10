"""
model.py  -  Step 5a: the main model (the disease classifier).

WHAT THIS FILE DOES
-------------------
It defines the model that looks at a leaf photo and answers: healthy, scab or
rust (paper draft v2, Section 3.7). It only DEFINES the model. Training is in
train_classifier.py.

THE MODEL
---------
  photo (3 x 224 x 224)
    -> MobileNetV3-Small, already trained on ImageNet photos
       (the part that "reads" the picture; it ends with 576 numbers per image)
    -> average over the picture  -> 576 numbers
    -> dropout 0.20              (randomly hides 20% of the numbers while
                                  training, so the model does not memorise)
    -> one linear layer with 3 outputs ("logits": one score each for
       healthy, scab, rust)
The class with the highest score is the prediction.

TWO TRAINING MODES (paper Section 3.10)
---------------------------------------
  Phase A: only the 3-output layer can change; the MobileNet part is locked.
  Phase B: the 3-output layer plus the LAST part of MobileNet can change
           (the final three inverted-residual blocks and the final 1x1
           convolution, modules 9, 10, 11 and 12 of the network).
In both phases the "BatchNorm" layers stay frozen: they keep the statistics
they learned on ImageNet and never update them.

extract_features() returns the 576 numbers per image. The failure head (a later
step) will use them together with the 3 scores.

PARAMETER COUNT (paper Appendix B)
----------------------------------
The paper planned 939,120 (MobileNet part) + 1,731 (output layer) = 940,851,
and says the exact number must be confirmed from the real implementation.
The PyTorch (torchvision) MobileNetV3-Small MobileNet part has 927,008
learnable parameters. The paper's 939,120 is 927,008 plus 12,112 BatchNorm
"running statistics" (numbers that are stored but never learned). Keras-style
parameter summaries add them; PyTorch does not. Running this file prints both
ways of counting, so the paper can state which one it reports.

HOW TO CHECK IT (project root, venv active):
    python src/model.py
(builds the model without downloading weights, prints the parameter counts and
runs the checks)
"""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from config import DISEASE_DROPOUT, FEATURE_DIM, IMAGE_SIZE, NUM_CLASSES

# Modules of torchvision's MobileNetV3-Small "features" list that are trained in
# phase B: three inverted-residual blocks (9, 10, 11) and the final 1x1 conv (12).
PHASE_B_MODULES = (9, 10, 11, 12)

# Numbers planned in the paper (Appendix B)
PAPER_BACKBONE_PARAMS = 939_120
PAPER_HEAD_PARAMS = 1_731
PAPER_CLASSIFIER_PARAMS = 940_851


class DiseaseClassifier(nn.Module):
    def __init__(self, pretrained=True):
        """pretrained=True downloads the ImageNet weights the first time (about
        10 MB) and needs internet. pretrained=False gives a random model, only
        for quick tests."""
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features   # the MobileNet part
        self.pool = nn.AdaptiveAvgPool2d(1)                            # average over the picture
        self.dropout = nn.Dropout(DISEASE_DROPOUT)
        self.fc = nn.Linear(FEATURE_DIM, NUM_CLASSES)                  # 576 -> 3 scores

    # --- forward passes -----------------------------------------------------
    def extract_features(self, images):
        """576 numbers per image (before dropout and the output layer)."""
        return torch.flatten(self.pool(self.features(images)), 1)

    def forward(self, images):
        """3 scores (logits) per image: healthy, scab, rust."""
        return self.fc(self.dropout(self.extract_features(images)))

    # --- freezing -----------------------------------------------------------
    def set_phase_a(self):
        """Lock the whole MobileNet part. Only the output layer can learn."""
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        for parameter in self.fc.parameters():
            parameter.requires_grad = True

    def set_phase_b(self):
        """Also let the last part of MobileNet learn (BatchNorm stays frozen)."""
        self.set_phase_a()
        for index in PHASE_B_MODULES:
            for parameter in self.features[index].parameters():
                parameter.requires_grad = True
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def train(self, mode=True):
        """Normal train()/eval() switch, except that the BatchNorm layers of the
        MobileNet part ALWAYS stay in evaluation mode (frozen statistics)."""
        super().train(mode)
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self


def count_parameters(model):
    """Return (all parameters, parameters that can currently learn)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def batchnorm_statistics(model):
    """Number of BatchNorm running-statistics values (stored, not learned).
    Keras-style parameter counts include them, PyTorch parameter counts do not."""
    return sum(buffer.numel() for name, buffer in model.named_buffers()
               if not name.endswith("num_batches_tracked"))


def parameter_report(model):
    backbone = sum(p.numel() for p in model.features.parameters())
    head = sum(p.numel() for p in model.fc.parameters())
    return backbone, head, backbone + head


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def self_check():
    print("Building the model (no weights downloaded)...\n")
    model = DiseaseClassifier(pretrained=False)
    backbone, head, total = parameter_report(model)

    print(f"{'part':<28}{'this model':>12}{'paper plan':>12}{'difference':>12}")
    for name, mine, paper in [("MobileNet part", backbone, PAPER_BACKBONE_PARAMS),
                              ("3-output layer", head, PAPER_HEAD_PARAMS),
                              ("disease classifier total", total, PAPER_CLASSIFIER_PARAMS)]:
        print(f"{name:<28}{mine:>12,}{paper:>12,}{mine - paper:>12,}")
    statistics = batchnorm_statistics(model)
    reconciled = backbone + statistics == PAPER_BACKBONE_PARAMS
    print(f"BatchNorm running statistics (stored, not learned): {statistics:,}")
    print(f"MobileNet part counted the Keras way = {backbone + statistics:,}  "
          f"{'(matches the paper plan)' if reconciled else '(does NOT match the paper plan)'}")
    print("The paper's figure counts the running statistics, PyTorch does not.\n")

    ok = True
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    model.eval()
    features = model.extract_features(images)
    logits = model(images)
    shape_ok = tuple(features.shape) == (4, FEATURE_DIM) and tuple(logits.shape) == (4, NUM_CLASSES)
    ok &= shape_ok
    print(f"Features per image: {tuple(features.shape[1:])}, scores per image: "
          f"{tuple(logits.shape[1:])}  {'OK' if shape_ok else 'PROBLEM'}")

    model.set_phase_a()
    _, trainable_a = count_parameters(model)
    a_ok = trainable_a == head
    ok &= a_ok
    print(f"Phase A can change {trainable_a:,} parameters (expected {head:,})  "
          f"{'OK' if a_ok else 'PROBLEM'}")

    model.set_phase_b()
    _, trainable_b = count_parameters(model)
    block_params = sum(p.numel() for index in PHASE_B_MODULES
                       for p in model.features[index].parameters())
    batchnorm_params = sum(p.numel() for index in PHASE_B_MODULES
                           for m in model.features[index].modules()
                           if isinstance(m, nn.BatchNorm2d) for p in m.parameters())
    expected_b = head + block_params - batchnorm_params
    b_ok = trainable_b == expected_b
    ok &= b_ok
    print(f"Phase B can change {trainable_b:,} parameters (expected {expected_b:,}, "
          f"BatchNorm excluded)  {'OK' if b_ok else 'PROBLEM'}")

    model.train()
    bn_modes = [m.training for m in model.features.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_ok = len(bn_modes) > 0 and not any(bn_modes)
    ok &= bn_ok
    print(f"BatchNorm layers stay frozen in train mode ({len(bn_modes)} layers)  "
          f"{'OK' if bn_ok else 'PROBLEM'}")
    dropout_ok = model.dropout.training
    ok &= dropout_ok
    print(f"Dropout is active in train mode  {'OK' if dropout_ok else 'PROBLEM'}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED, send this output for help"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)