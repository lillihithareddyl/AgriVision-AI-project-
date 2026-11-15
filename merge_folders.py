"""Merge class folders from several source folders into one (e.g. PlantDoc train + test).
  python merge_folders.py --src PlantDoc/train PlantDoc/test --dst data/plant/field
"""
import argparse, shutil
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--src", nargs="+", required=True); ap.add_argument("--dst", required=True)
a = ap.parse_args(); n = 0
for s in a.src:
    for cdir in Path(s).iterdir():
        if cdir.is_dir():
            out = Path(a.dst) / cdir.name; out.mkdir(parents=True, exist_ok=True)
            for f in cdir.rglob("*"):
                if f.is_file():
                    shutil.copy(f, out / f"{Path(s).name}_{f.name}"); n += 1
print("copied", n, "files to", a.dst)
