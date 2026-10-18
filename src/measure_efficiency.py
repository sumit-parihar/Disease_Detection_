"""
measure_efficiency.py  -  Measures how heavy the system is: parameters, file size,
FLOPs/MACs and inference time of the disease classifier and of the failure heads.
It uses random numbers as input (no real image, no label) and changes nothing.

WHAT IT MEASURES (classifier seed 42 is used; the architecture is the same for every seed)
------------------------------------------------------------------------------------------
  * learnable parameters of the classifier and of each failure head
  * size of each checkpoint file on disk (MB)
  * FLOPs and MACs for ONE image (counted by PyTorch for convolutions and matrix products only;
    MACs = FLOPs / 2)
  * inference time per image, in milliseconds, for batch size 1 and batch size 32, on the CPU
    and (if available) on the GPU: classifier, each failure head, and the extra cost of the
    proposed head as a percentage of the classifier
Timings depend on the computer. The hardware and the PyTorch version are written into the output.

OUTPUT (results/): efficiency.csv and efficiency_summary.txt

HOW TO RUN (project root, venv active):
    python src/measure_efficiency.py
"""

import argparse
import platform
import sys
import time

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from config import IMAGE_SIZE, RESULTS_DIR
from frozen_pipeline import HEAD_VARIANTS, classifier_path, head_path, head_input, load_frozen_classifier, load_frozen_head

SEED = 42
WARMUP = 20
RUNS = {1: 200, 32: 50}
OUT_CSV = RESULTS_DIR / "efficiency.csv"
OUT_TXT = RESULTS_DIR / "efficiency_summary.txt"


def learnable(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(function, *inputs):
    with FlopCounterMode(display=False) as counter:
        function(*inputs)
    return counter.get_total_flops()


def timed(function, device, runs):
    for _ in range(WARMUP):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000.0          # milliseconds per batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (OUT_CSV.exists() or OUT_TXT.exists()) and not args.overwrite:
        sys.exit("Efficiency files already exist. Use --overwrite to measure again.")

    rows = []

    def add(item, value, unit, device="", batch=""):
        rows.append({"item": item, "value": value, "unit": unit, "device": device, "batch_size": batch})

    cpu = torch.device("cpu")
    classifier = load_frozen_classifier(SEED, cpu)
    heads = {v: load_frozen_head(v, SEED, cpu) for v in HEAD_VARIANTS}

    add("classifier learnable parameters", learnable(classifier), "parameters")
    for v, (head, _, _) in heads.items():
        add(f"failure head ({v}) learnable parameters", learnable(head), "parameters")
    add("classifier checkpoint file", round(classifier_path(SEED).stat().st_size / 1e6, 3), "MB")
    for v in HEAD_VARIANTS:
        add(f"failure head ({v}) checkpoint file", round(head_path(v, SEED).stat().st_size / 1e6, 3), "MB")

    def run_classifier(x):
        features = classifier.extract_features(x)
        return features, classifier.fc(classifier.dropout(features))

    with torch.inference_mode():
        x1 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        flops_classifier = count_flops(lambda: run_classifier(x1))
        features1, logits1 = run_classifier(x1)
        add("classifier FLOPs per image", flops_classifier, "FLOPs")
        add("classifier MACs per image", flops_classifier // 2, "MACs")
        for v, (head, mean, std) in heads.items():
            flops_head = count_flops(lambda: head((head_input(v, features1, logits1) - mean) / std))
            add(f"failure head ({v}) FLOPs per image", flops_head, "FLOPs")
            add(f"failure head ({v}) MACs per image", flops_head // 2, "MACs")

    devices = [cpu] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for device in devices:
        classifier_d = load_frozen_classifier(SEED, device)
        heads_d = {v: load_frozen_head(v, SEED, device) for v in HEAD_VARIANTS}
        for batch, runs in RUNS.items():
            x = torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            with torch.inference_mode():
                def run():
                    f = classifier_d.extract_features(x)
                    return f, classifier_d.fc(classifier_d.dropout(f))
                features, logits = run()
                t_classifier = timed(run, device, runs) / batch
                add("classifier time per image", round(t_classifier, 4), "ms", device.type, batch)
                for v, (head, mean, std) in heads_d.items():
                    t_head = timed(lambda: torch.sigmoid(head((head_input(v, features, logits) - mean) / std)), device, runs) / batch
                    add(f"failure head ({v}) time per image", round(t_head, 4), "ms", device.type, batch)
                    if v == "proposed":
                        add("proposed head extra time (share of classifier time)", round(100 * t_head / t_classifier, 3), "%", device.type, batch)
                        add("classifier + proposed head time per image", round(t_classifier + t_head, 4), "ms", device.type, batch)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False, lineterminator="\n")
    lines = ["EFFICIENCY OF THE FROZEN SYSTEM (random input; no image or label used)", "=" * 70, "",
             f"Python {platform.python_version()}, PyTorch {torch.__version__}, CPU threads {torch.get_num_threads()}",
             f"Processor: {platform.processor() or platform.machine()}",
             f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}", ""]
    for _, r in table.iterrows():
        where = f"  [{r.device}, batch {r.batch_size}]" if r.device != "" else ""
        lines.append(f"{r['item']:<58}{r['value']:>16,}  {r['unit']}{where}")
    lines += ["", "FLOPs are counted by PyTorch for convolutions and matrix products only. Timings depend on this computer."]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()