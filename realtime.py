"""Real-time plant disease / fruit defect detection from a webcam (or a video file).
  python realtime.py --plant models/plant.pt --fruit models/fruit.pt
Keys: 1 = plant mode | 2 = fruit mode | 3 = auto (most confident model) | s = save screenshot | q / ESC = quit
Hold the leaf or fruit inside the green box, close to the camera, in good light.
"""
import argparse, time, os
from collections import deque
import cv2, numpy as np, torch
from dl_utils import load_checkpoint, preprocess_array, DEV

ap = argparse.ArgumentParser()
ap.add_argument("--plant", default="models/plant.pt")
ap.add_argument("--fruit", default="models/fruit.pt")
ap.add_argument("--cam", default="0", help="camera index (0,1,..) or path to a video file")
ap.add_argument("--threshold", type=float, default=0.60, help="below this confidence show 'uncertain'")
ap.add_argument("--smooth", type=int, default=5, help="average predictions over last N frames")
ap.add_argument("--box", type=float, default=0.7, help="size of the green box (fraction of the shorter side)")
a = ap.parse_args()

models = {}
for name, path in (("plant", a.plant), ("fruit", a.fruit)):
    if os.path.exists(path):
        m, classes, S = load_checkpoint(path); models[name] = (m, classes, S); print(f"loaded {name}: {len(classes)} classes")
    else:
        print(f"(no {name} model at {path}, that mode is disabled)")
if not models:
    raise SystemExit("No models found. Train first with train.py")

pretty = lambda s: s.replace("_", " ")
mode = "auto" if len(models) > 1 else next(iter(models))
buf = {k: deque(maxlen=a.smooth) for k in models}
cap = cv2.VideoCapture(int(a.cam) if a.cam.isdigit() else a.cam)
if not cap.isOpened():
    raise SystemExit("Cannot open camera/video. Try --cam 1, close other apps using the camera, or allow camera access.")
os.makedirs("screenshots", exist_ok=True); fps, t_prev = 0.0, time.time()

@torch.no_grad()
def infer(name, rgb):
    m, classes, S = models[name]
    p = m(preprocess_array(rgb, S)).float().softmax(1)[0].cpu().numpy()
    buf[name].append(p); return np.mean(buf[name], axis=0), classes

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if a.cam.isdigit():
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]; s = int(min(h, w) * a.box); x0, y0 = (w - s) // 2, (h - s) // 2
    rgb = cv2.cvtColor(frame[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2RGB)
    names = list(models) if mode == "auto" else [mode]
    best = None
    for n in names:
        p, classes = infer(n, rgb)
        if best is None or p.max() > best[1].max():
            best = (n, p, classes)
    n, p, classes = best
    top = np.argsort(p)[::-1][:3]
    conf = float(p[top[0]]); ok_conf = conf >= a.threshold
    color = (0, 200, 0) if ok_conf else (0, 165, 255)
    cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), color, 2)
    cv2.rectangle(frame, (0, 0), (w, 120), (30, 30, 30), -1)
    title = f"{n.upper()}: {pretty(classes[top[0]])}" if ok_conf else "Uncertain - move closer / better light"
    cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for i, k in enumerate(top):
        y = 55 + i * 22
        cv2.rectangle(frame, (10, y - 12), (10 + int(220 * p[k]), y + 4), (200, 160, 60), -1)
        cv2.putText(frame, f"{pretty(classes[k])[:34]} {p[k]*100:.0f}%", (240, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    now = time.time(); fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6); t_prev = now
    cv2.putText(frame, f"mode:{mode}  {fps:.1f} FPS  [1 plant 2 fruit 3 auto  s save  q quit]", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Lab-to-Field: Plant & Fruit Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        break
    if key == ord("1") and "plant" in models: mode = "plant"
    if key == ord("2") and "fruit" in models: mode = "fruit"
    if key == ord("3") and len(models) > 1: mode = "auto"
    if key == ord("s"):
        fn = f"screenshots/shot_{int(time.time())}.png"; cv2.imwrite(fn, frame); print("saved", fn)
cap.release(); cv2.destroyAllWindows()
