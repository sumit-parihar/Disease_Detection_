"""
check_environment.py  -  Confirms the computer is ready before we write
any real code.

WHAT THIS FILE DOES
-------------------
It only reads and prints. It does not change anything. It checks:
  1. The Python version.
  2. That every library we need can be imported (and prints its version).
  3. Whether PyTorch can see your NVIDIA GPU (RTX 3050).
  4. Whether the raw dataset folders are in the expected place.

WHY WE NEED IT
--------------
A missing library or a PyTorch install without GPU support is much easier to
find now than in the middle of training.

HOW TO RUN (from D:\\Capstone Project, with the venv active):
    python src/check_environment.py
"""

import importlib
import sys

from config import PLANTVILLAGE_DIR, PP2020_DIR, SOURCE_FOLDERS

# Libraries we will use later, and what each one is for.
REQUIRED = {
    "numpy": "arrays and maths",
    "pandas": "tables (manifest, train.csv)",
    "PIL": "opening images (installed as Pillow)",
    "sklearn": "metrics such as AUROC",
    "scipy": "statistics and bootstrap",
    "matplotlib": "plots",
    "imagehash": "pHash near-duplicate detection",
    "tqdm": "progress bars",
    "torch": "deep learning",
    "torchvision": "MobileNetV3-Small and image transforms",
}


def check_python():
    print(f"Python version : {sys.version.split()[0]}")


def check_libraries():
    """Try to import each library and report its version or the problem."""
    print("\nLibraries:")
    all_ok = True
    for name, purpose in REQUIRED.items():
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "version not shown")
            print(f"  OK       {name:<12} {version:<18} ({purpose})")
        except ImportError:
            all_ok = False
            print(f"  MISSING  {name:<12} ({purpose})")
    return all_ok


def check_gpu():
    """Ask PyTorch whether a CUDA GPU is available."""
    print("\nGPU:")
    try:
        import torch
    except ImportError:
        print("  PyTorch is not installed, so the GPU cannot be checked.")
        return
    if torch.cuda.is_available():
        print(f"  CUDA available : yes")
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory     : {memory_gb:.1f} GB")
    else:
        print("  CUDA available : NO. Training will use the CPU (slow).")
        print("  Usually this means the CPU-only PyTorch was installed.")


def check_data_folders():
    """Check that the raw dataset folders exist. Counting images and
    comparing with the paper is the job of the audit step, not this file."""
    print("\nData folders:")
    for class_name, folder in SOURCE_FOLDERS.items():
        path = PLANTVILLAGE_DIR / folder
        status = "found" if path.exists() else "NOT FOUND"
        print(f"  {status:<10} {path}")
    status = "found" if PP2020_DIR.exists() else "NOT FOUND"
    print(f"  {status:<10} {PP2020_DIR}  (Plant Pathology 2020)")


if __name__ == "__main__":
    check_python()
    libraries_ok = check_libraries()
    check_gpu()
    check_data_folders()
    if not libraries_ok:
        print("\nSome libraries are missing. Run: pip install -r requirements.txt")