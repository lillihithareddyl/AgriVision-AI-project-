"""Run this FIRST: shows which classes match between lab and field folders."""
import argparse
from data_utils import scan, load_class_map

ap = argparse.ArgumentParser()
ap.add_argument("--lab_dir", required=True)
ap.add_argument("--field_dir", required=True)
a = ap.parse_args()
cm = load_class_map()
lab, fld = scan(a.lab_dir, cm), scan(a.field_dir, cm)
shared = sorted(set(lab) & set(fld))
print(f"\nLab classes: {len(lab)} | Field classes: {len(fld)} | MATCHED: {len(shared)}")
for c in shared:
    print(f"  {c:45s} lab={len(lab[c]):5d} field={len(fld[c]):4d}")
print("\nOnly in lab  :", sorted(set(lab) - set(fld)))
print("Only in field:", sorted(set(fld) - set(lab)))
print("\nIf a field class is really the same as a lab class, add it to class_map.json, e.g.")
print('  "bell_pepper_leaf_spot": "pepper_bell_bacterial_spot"')
