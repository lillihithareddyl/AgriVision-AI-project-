"""Evaluate one model, or an ENSEMBLE of several, on any folder of class sub-folders.
  python evaluate.py --ckpts models/plant.pt --data_dir data/plant/field
  python evaluate.py --ckpts runs_a/plant.pt runs_b/plant.pt --data_dir data/plant/field   (ensemble, same classes)
"""
import argparse, time, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
from sklearn.metrics import classification_report
from data_utils import scan, load_class_map
from dl_utils import *
from train import plot_cm

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--out", default="runs/eval")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
loaded = [load_checkpoint(p) for p in a.ckpts]
classes = loaded[0][1]
assert all(c == classes for _, c, _ in loaded), "Ensemble members must share the same class list"
idx = {c: i for i, c in enumerate(classes)}
data = scan(a.data_dir, load_class_map())
samples = [(p, idx[c]) for c in classes if c in data for p in data[c]]
print(f"{len(samples)} images from {len({y for _, y in samples})} matching classes")
ens, t0 = None, time.time()
for (m, _, S) in loaded:
    P, L = predict_probs(m, samples, get_transforms(S)["eval"], a.bs, S)
    ens = P if ens is None else ens + P
ens /= len(loaded); sec = time.time() - t0
acc, f1 = metrics_from_probs(ens, L)
print(f"\nACCURACY {acc*100:.2f}% | MACRO-F1 {f1*100:.2f}% | {len(samples)*len(loaded)/sec:.1f} img/s")
labels = sorted(set(L.tolist()))
print(classification_report(L, ens.argmax(1), labels=labels, target_names=[classes[i] for i in labels], zero_division=0, digits=4))
plot_cm(L, ens.argmax(1), classes, f"{a.out}/cm_eval.png", "evaluation")
json.dump(dict(acc=float(acc), macro_f1=float(f1), n=len(samples), models=a.ckpts), open(f"{a.out}/eval.json", "w"), indent=2)
