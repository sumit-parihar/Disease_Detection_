import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
DEST = Path("data") / "plantvillage"
CLASSES = ["Apple___healthy", "Apple___Apple_scab", "Apple___Cedar_apple_rust"]
EXPECTED = {"Apple___healthy": 1645, "Apple___Apple_scab": 630, "Apple___Cedar_apple_rust": 275}
GIT_DEFAULT = r"C:\Program Files\Git\cmd"


def find_git():
    if shutil.which("git"):
        return True
    if Path(GIT_DEFAULT, "git.exe").exists():
        os.environ["PATH"] += os.pathsep + GIT_DEFAULT
        return True
    return False


def ensure_git():
    if find_git():
        return
    print("Git not found. Installing with winget...")
    if not shutil.which("winget"):
        sys.exit("winget is not available. Install Git manually from https://git-scm.com/download/win and run this script again.")
    subprocess.run(
        ["winget", "install", "--id", "Git.Git", "-e", "--source", "winget",
         "--accept-package-agreements", "--accept-source-agreements"],
        check=True,
    )
    if not find_git():
        sys.exit("Git was installed but is not visible yet. Close this window, open a new one, and run the script again.")


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ensure_git()
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if not DEST.exists():
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(DEST)])

    patterns = [f"raw/color/{c}/*" for c in CLASSES]
    run(["git", "sparse-checkout", "set", "--no-cone", *patterns], cwd=DEST)
    run(["git", "checkout"], cwd=DEST)

    print("\nVerification:")
    for c in CLASSES:
        n = len([p for p in (DEST / "raw" / "color" / c).glob("*") if p.is_file()])
        status = "OK" if n == EXPECTED[c] else "CHECK"
        print(f"  {c}: {n} files (expected {EXPECTED[c]}) {status}")
    print(f"\nSaved in: {DEST.resolve()}")


if __name__ == "__main__":
    main()