"""Classical ML baseline (colour histograms + Random Forest / SVM) to compare against deep learning.
  python baseline_ml.py --lab_dir data/plant/lab --field_dir data/plant/field --cap_lab 200
"""
import argparse, json, os, time
import cv2, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from data_utils import build_splits


def feats(path):
    img = cv2.imread(path)
    if img is None:
        return np.zeros(512 + 12, dtype=np.float32)
    img = cv2.resize(img, (128, 128)); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    h = h / (h.sum() + 1e-9)
    st = np.concatenate([img.reshape(-1, 3).mean(0), img.reshape(-1, 3).std(0), hsv.reshape(-1, 3).mean(0), hsv.reshape(-1, 3).std(0)]) / 255
    return np.concatenate([h, st]).astype(np.float32)


def xy(samples):
    return np.stack([feats(p) for p, _ in samples]), np.array([y for _, y in samples])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab_dir", required=True); ap.add_argument("--field_dir", default=None)
    ap.add_argument("--cap_lab", type=int, default=200); ap.add_argument("--cap_field", type=int, default=0)
    ap.add_argument("--out", default="runs/baseline_ml.json")
    a = ap.parse_args()
    D = build_splits(a.lab_dir, a.field_dir, cap_lab=a.cap_lab, cap_field=a.cap_field)
    Xtr, ytr = xy(D["lab_train"]); Xte, yte = xy(D["lab_test"])
    Xf, yf = xy(D["field_test"]) if D["field_test"] else (None, None)
    res = {}
    for name, clf in {"RandomForest": RandomForestClassifier(300, n_jobs=-1, random_state=42),
                      "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))}.items():
        t0 = time.time(); clf.fit(Xtr, ytr); r = {}
        p = clf.predict(Xte); r["lab_acc"] = float(accuracy_score(yte, p)); r["lab_macro_f1"] = float(f1_score(yte, p, average="macro"))
        if Xf is not None:
            p = clf.predict(Xf); r["field_acc"] = float(accuracy_score(yf, p))
            r["field_macro_f1"] = float(f1_score(yf, p, average="macro", labels=sorted(set(yf.tolist())), zero_division=0))
        r["seconds"] = round(time.time() - t0, 1); res[name] = r
        print(name, {k: (round(v * 100, 2) if "acc" in k or "f1" in k else v) for k, v in r.items()})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
