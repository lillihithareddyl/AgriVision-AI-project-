"""Dataset scanning, class matching and splitting (no torch needed)."""
import json, random, re
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HERE = Path(__file__).parent


def load_class_map(path=None):
    p = Path(path) if path else HERE / "class_map.json"
    if not p.exists():
        return {}
    return {k: v for k, v in json.load(open(p)).items() if not k.startswith("_")}


def norm(name, cmap):
    n = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    n = re.sub(r"_leaf$", "", n)
    return cmap.get(n, n)


def scan(root, cmap, cap=0, seed=42):
    """Return {class_name: [image paths]} for root/<class>/**/*.jpg (nested train/test folders are merged)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    items = defaultdict(list)
    for cdir in sorted(root.iterdir()):
        if cdir.is_dir():
            items[norm(cdir.name, cmap)] += [str(p) for p in cdir.rglob("*") if p.suffix.lower() in EXT]
    rng = random.Random(seed)
    for c in items:
        rng.shuffle(items[c])
        if cap:
            items[c] = items[c][:cap]
    return dict(items)


def build_splits(lab_dir, field_dir=None, cmap=None, cap_lab=0, cap_field=0, min_per_class=10, seed=42):
    """Train on ALL lab classes. Field data is restricted to classes that also exist in the lab set."""
    cmap = cmap if cmap is not None else load_class_map()
    lab = scan(lab_dir, cmap, cap_lab, seed)
    classes = sorted(c for c, v in lab.items() if len(v) >= min_per_class)
    idx = {c: i for i, c in enumerate(classes)}
    lab_s = [(p, idx[c]) for c in classes for p in lab[c]]
    tr, tmp = train_test_split(lab_s, test_size=0.3, stratify=[y for _, y in lab_s], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=[y for _, y in tmp], random_state=seed)
    out = dict(classes=classes, lab_train=tr, lab_val=va, lab_test=te,
               field_pool=[], field_test=[], field_classes=[])
    print(f"[data] lab classes: {len(classes)} | train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    if field_dir:
        fld = scan(field_dir, cmap, cap_field, seed)
        shared = sorted(c for c in classes if c in fld and len(fld[c]) >= max(5, min_per_class // 2))
        fld_s = [(p, idx[c]) for c in shared for p in fld[c]]
        if len(shared) >= 2:
            pool, ft = train_test_split(fld_s, test_size=0.6, stratify=[y for _, y in fld_s], random_state=seed)
            out.update(field_pool=pool, field_test=ft, field_classes=shared)
        print(f"[data] field classes matched: {len(shared)} | field pool/test = {len(out['field_pool'])}/{len(out['field_test'])}")
        print("[data] only in field (add to class_map.json if same class):", sorted(set(fld) - set(classes))[:20])
    return out
