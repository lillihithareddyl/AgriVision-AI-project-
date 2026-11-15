"""Builds graphs, a summary table and ONE results page (results/dashboard.html) from runs/*/metrics.json."""
import argparse, base64, glob, json, os, webbrowser
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(runs):
    rows = []
    for f in sorted(glob.glob(f"{runs}/*/metrics.json")):
        m = json.load(open(f)); m["run_dir"] = os.path.dirname(f); rows.append(m)
    base = {}
    for f in glob.glob(f"{runs}/baseline_*.json"):
        base[os.path.basename(f)[9:-5]] = json.load(open(f))
    return rows, base


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def build(runs="runs", out="results", show=True):
    os.makedirs(out, exist_ok=True)
    rows, base = load(runs)
    if not rows:
        raise SystemExit(f"No metrics.json found in {runs}/. Train first (python run_all.py ...).")
    tasks = sorted({r["task"] for r in rows})
    # ---- summary table
    tab = []
    for r in rows:
        best_k = max(r.get("kshot", {}).items(), key=lambda kv: int(kv[0]), default=(None, None))
        tab.append({"Task": r["task"], "Model": r["model"], "Aug": r["aug"], "Params (M)": r["params_millions"],
                    "Lab acc %": round(r["lab_acc"] * 100, 2), "Lab macro-F1 %": round(r["lab_macro_f1"] * 100, 2),
                    "Field acc %": round(r["field_acc"] * 100, 2) if "field_acc" in r else None,
                    "Gap (pts)": round(r["gap"] * 100, 2) if "gap" in r else None,
                    f"Field acc % after {best_k[0]}-shot": round(best_k[1]["acc"] * 100, 2) if best_k[0] else None,
                    "Train min": r["train_minutes"]})
    for t, d in base.items():
        for name, v in d.items():
            tab.append({"Task": t, "Model": name + " (classical ML)", "Aug": "-", "Params (M)": None,
                        "Lab acc %": round(v["lab_acc"] * 100, 2), "Lab macro-F1 %": round(v["lab_macro_f1"] * 100, 2),
                        "Field acc %": round(v["field_acc"] * 100, 2) if "field_acc" in v else None,
                        "Gap (pts)": round((v["lab_acc"] - v["field_acc"]) * 100, 2) if "field_acc" in v else None})
    df = pd.DataFrame(tab); df.to_csv(f"{out}/summary.csv", index=False)
    figs = []

    # ---- Figure 1: lab vs field vs few-shot, per task
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.2), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        rs = [r for r in rows if r["task"] == t]; x = np.arange(len(rs)); w = 0.27
        lab = [r["lab_acc"] * 100 for r in rs]; fld = [r.get("field_acc", np.nan) * 100 for r in rs]
        ks = [max(r["kshot"].items(), key=lambda kv: int(kv[0]))[1]["acc"] * 100 if r.get("kshot") else np.nan for r in rs]
        for off, vals, lab_name, col in ((-w, lab, "Lab test", "#2a9d8f"), (0, fld, "Field test (zero-shot)", "#e76f51"), (w, ks, "Field + few-shot", "#264653")):
            b = ax.bar(x + off, vals, w, label=lab_name, color=col)
            for rect, v in zip(b, vals):
                if not np.isnan(v): ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([f"{r['model']}\n({r['aug']})" for r in rs], fontsize=8)
        ax.set_ylim(0, 108); ax.set_ylabel("accuracy %"); ax.set_title(f"{t}: lab vs field accuracy"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); p = f"{out}/1_lab_vs_field_accuracy.png"; fig.savefig(p, dpi=170); plt.close(fig); figs.append(("Lab vs field accuracy (the generalization gap)", p))

    # ---- Figure 2: few-shot curves
    if any(r.get("kshot") for r in rows):
        fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 3.8), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            for r in [r for r in rows if r["task"] == t and "field_acc" in r]:
                ks = sorted(r.get("kshot", {}), key=int)
                ax.plot([0] + [int(k) for k in ks], [r["field_acc"] * 100] + [r["kshot"][k]["acc"] * 100 for k in ks], marker="o", label=f"{r['model']} ({r['aug']})")
            ax.set_xlabel("field images per class used for fine-tuning"); ax.set_ylabel("field accuracy %"); ax.set_title(f"{t}: how much field data is needed"); ax.legend(fontsize=7)
        fig.tight_layout(); p = f"{out}/2_few_shot_curve.png"; fig.savefig(p, dpi=170); plt.close(fig); figs.append(("Effect of a few field images", p))

    # ---- Figure 3: classical ML vs deep learning
    if base:
        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            sub = df[(df.Task == t)].dropna(subset=["Field acc %"]); x = np.arange(len(sub)); w = 0.38
            ax.bar(x - w / 2, sub["Lab acc %"], w, label="Lab", color="#2a9d8f"); ax.bar(x + w / 2, sub["Field acc %"], w, label="Field", color="#e76f51")
            ax.set_xticks(x); ax.set_xticklabels([m.replace(" (classical ML)", "\n(ML)") for m in sub["Model"]], fontsize=7, rotation=20)
            ax.set_ylim(0, 105); ax.set_ylabel("accuracy %"); ax.set_title(f"{t}: classical ML vs deep learning"); ax.legend(fontsize=7)
        fig.tight_layout(); p = f"{out}/3_ml_vs_deep_learning.png"; fig.savefig(p, dpi=170); plt.close(fig); figs.append(("Classical ML vs deep learning", p))

    # ---- HTML page
    h = ["<html><head><meta charset='utf-8'><title>Lab-to-Field Gap: Results</title><style>body{font-family:Arial;max-width:1150px;margin:30px auto;padding:0 16px;color:#222}"
         "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:5px 9px;text-align:center}th{background:#f0f0f0}img{max-width:100%;margin:8px 0}"
         "h2{border-bottom:2px solid #2a9d8f;padding-bottom:4px;margin-top:34px}.row{display:flex;flex-wrap:wrap;gap:12px}.row div{flex:1 1 360px}</style></head><body>",
         "<h1>The Laboratory-to-Field Gap: Results</h1><p>Generated automatically from your training runs.</p><h2>Summary table</h2>", df.to_html(index=False, na_rep="-")]
    for title, p in figs:
        h.append(f"<h2>{title}</h2><img src='data:image/png;base64,{b64(p)}'>")
    for r in rows:
        h.append(f"<h2>Details: {r['task']} / {r['model']} / {r['aug']}</h2><div class='row'>")
        for fn, cap in (("curves.png", "Training curves"), ("cm_lab.png", "Confusion matrix (lab)"), ("cm_field.png", "Confusion matrix (field)"), ("kshot.png", "Few-shot")):
            fp = os.path.join(r["run_dir"], fn)
            if os.path.exists(fp): h.append(f"<div><b>{cap}</b><br><img src='data:image/png;base64,{b64(fp)}'></div>")
        h.append("</div>")
    h.append("</body></html>")
    page = f"{out}/dashboard.html"; open(page, "w", encoding="utf-8").write("\n".join(h))
    print("\n" + df.to_string(index=False)); print(f"\nDashboard: {os.path.abspath(page)}\nGraphs + summary.csv in: {os.path.abspath(out)}")
    if show: webbrowser.open("file://" + os.path.abspath(page))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results"); ap.add_argument("--no_open", action="store_true")
    a = ap.parse_args(); build(a.runs, a.out, show=not a.no_open)
