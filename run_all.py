"""ONE command: trains plant + fruit models, runs the ML baseline, builds all graphs and opens results/dashboard.html.

python run_all.py --plant_lab data/plant/lab --plant_field data/plant/field --fruit_lab data/fruit/lab --fruit_field data/fruit/field
Options: --models efficientnet_b0 resnet50   --epochs 8   --cap_lab 300 (faster)   --quick (2-minute test)
"""
import argparse, subprocess, sys
from make_dashboard import build

ap = argparse.ArgumentParser()
ap.add_argument("--plant_lab"); ap.add_argument("--plant_field"); ap.add_argument("--fruit_lab"); ap.add_argument("--fruit_field")
ap.add_argument("--models", nargs="+", default=["efficientnet_b0"]); ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--cap_lab", type=int, default=0); ap.add_argument("--img_size", type=int, default=224)
ap.add_argument("--quick", action="store_true"); ap.add_argument("--skip_baseline", action="store_true")
a = ap.parse_args()
py = sys.executable
tasks = [(t, l, f) for t, l, f in (("plant", a.plant_lab, a.plant_field), ("fruit", a.fruit_lab, a.fruit_field)) if l]
if not tasks:
    raise SystemExit("Give at least --plant_lab or --fruit_lab")
for t, lab, fld in tasks:
    for i, m in enumerate(a.models):
        cmd = [py, "train.py", "--task", t, "--lab_dir", lab, "--model", m, "--epochs", str(a.epochs),
               "--cap_lab", str(a.cap_lab), "--img_size", str(a.img_size), "--ckpt", f"models/{t}.pt" if i == 0 else f"models/{t}_{m}.pt"]
        if fld: cmd += ["--field_dir", fld]
        if a.quick: cmd.append("--quick")
        print("\n>>>", " ".join(cmd)); subprocess.run(cmd, check=True)
    if not a.skip_baseline:
        cmd = [py, "baseline_ml.py", "--lab_dir", lab, "--cap_lab", str(a.cap_lab or 200), "--out", f"runs/baseline_{t}.json"]
        if fld: cmd += ["--field_dir", fld]
        print("\n>>>", " ".join(cmd)); subprocess.run(cmd, check=True)
build("runs", "results", show=True)
print("\nNow run the camera demo:  python realtime.py")
