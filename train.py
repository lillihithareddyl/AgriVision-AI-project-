"""Train a model on LAB images, test in-domain (lab) and cross-domain (field), optional few-shot field fine-tuning.

Example (plants):
  python train.py --task plant --lab_dir data/plant/lab --field_dir data/plant/field --model efficientnet_b0 --epochs 8
Example (fruit):
  python train.py --task fruit --lab_dir data/fruit/lab --field_dir data/fruit/field --model efficientnet_b0 --epochs 8
Quick smoke test on a laptop:  add --quick
"""
import argparse, copy, json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from data_utils import build_splits
from dl_utils import *


def plot_cm(L, P, names, path, title):
    labels = sorted(set(L.tolist()) | set(P.tolist()))
    cm = confusion_matrix(L, P, labels=labels, normalize="true")
    n = len(labels); fig, ax = plt.subplots(figsize=(max(5, 0.35 * n + 3),) * 2)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([names[i] for i in labels], rotation=90, fontsize=6)
    ax.set_yticklabels([names[i] for i in labels], fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def write_report(L, P, names, path):
    labels = sorted(set(L.tolist()))
    txt = classification_report(L, P, labels=labels, target_names=[names[i] for i in labels], zero_division=0, digits=4)
    open(path, "w").write(txt); return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="plant or fruit (just a name for outputs)")
    ap.add_argument("--lab_dir", required=True)
    ap.add_argument("--field_dir", default=None)
    ap.add_argument("--model", default="efficientnet_b0",
                    help="any timm model: efficientnet_b0, resnet50, mobilenetv3_large_100, vit_base_patch16_224, convnext_tiny ...")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--aug", default="strong", choices=["basic", "strong"])
    ap.add_argument("--cap_lab", type=int, default=0, help="max images per lab class (0 = all)")
    ap.add_argument("--cap_field", type=int, default=0)
    ap.add_argument("--kshot", type=int, nargs="*", default=[5, 10, 20], help="field images per class for fine-tuning")
    ap.add_argument("--ft_epochs", type=int, default=5)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny run to test the pipeline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--ckpt", default=None, help="where to save the final model (default models/<task>.pt)")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.ft_epochs, a.cap_lab, a.cap_field, a.kshot = 2, 2, 60, 40, [5]
    seed_all(a.seed); S = a.img_size
    run = os.path.join(a.out, f"{a.task}_{a.model}_{a.aug}"); os.makedirs(run, exist_ok=True)
    print(f"device: {DEV} | run folder: {run}")

    D = build_splits(a.lab_dir, a.field_dir, cap_lab=a.cap_lab, cap_field=a.cap_field, seed=a.seed)
    classes, TF = D["classes"], get_transforms(S)
    model = make_model(a.model, len(classes), pretrained=not a.no_pretrained)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model {a.model}: {nparams/1e6:.2f}M parameters, {len(classes)} classes")

    t0 = time.time()
    model, hist = fit(model, D["lab_train"], D["lab_val"], TF[a.aug], a.epochs, a.lr, a.bs, S, tag=a.task)
    train_min = (time.time() - t0) / 60

    res = dict(task=a.task, model=a.model, aug=a.aug, params_millions=round(nparams / 1e6, 2),
               classes=len(classes), epochs=a.epochs, lr=a.lr, batch_size=a.bs, img_size=S,
               train_minutes=round(train_min, 1), n_lab_test=len(D["lab_test"]))
    P, L = predict_probs(model, D["lab_test"], TF["eval"], a.bs, S)
    res["lab_acc"], res["lab_macro_f1"] = [float(v) for v in metrics_from_probs(P, L)]
    print(f"\nLAB TEST: acc {res['lab_acc']*100:.2f}% | macro-F1 {res['lab_macro_f1']*100:.2f}%")
    print(write_report(L, P.argmax(1), classes, f"{run}/report_lab.txt"))
    plot_cm(L, P.argmax(1), classes, f"{run}/cm_lab.png", f"{a.task} {a.model}: lab test")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(hist["loss"], marker="o"); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist["train_acc"], marker="o", label="train"); ax[1].plot(hist["val_acc"], marker="o", label="val")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{run}/curves.png", dpi=200); plt.close(fig)

    ck = a.ckpt or f"models/{a.task}.pt"
    if D["field_test"]:
        Pf, Lf = predict_probs(model, D["field_test"], TF["eval"], a.bs, S)
        res["field_acc"], res["field_macro_f1"] = [float(v) for v in metrics_from_probs(Pf, Lf)]
        res["gap"] = res["lab_acc"] - res["field_acc"]; res["n_field_test"] = len(D["field_test"])
        print(f"FIELD TEST (zero-shot): acc {res['field_acc']*100:.2f}% | macro-F1 {res['field_macro_f1']*100:.2f}% | gap {res['gap']*100:.2f} pts")
        write_report(Lf, Pf.argmax(1), classes, f"{run}/report_field.txt")
        plot_cm(Lf, Pf.argmax(1), classes, f"{run}/cm_field.png", f"{a.task} {a.model}: field test (zero-shot)")
        save_checkpoint(ck, model, a.model, classes, S, extra=dict(lab_acc=res["lab_acc"], field_acc=res["field_acc"]))
        res["kshot"] = {}
        for k in a.kshot:
            m2 = copy.deepcopy(model); shots = sample_k(D["field_pool"], k, a.seed)
            m2, _ = fit(m2, shots, None, TF["strong"], a.ft_epochs, a.ft_lr, a.bs, S, tag=f"{k}-shot")
            Pk, Lk = predict_probs(m2, D["field_test"], TF["eval"], a.bs, S)
            acc, f1 = metrics_from_probs(Pk, Lk); res["kshot"][k] = dict(acc=float(acc), macro_f1=float(f1))
            print(f"FIELD +{k}-shot fine-tune: acc {acc*100:.2f}% | macro-F1 {f1*100:.2f}%")
            save_checkpoint(ck.replace(".pt", f"_field{k}shot.pt"), m2, a.model, classes, S)
            del m2
        if res["kshot"]:
            ks = sorted(res["kshot"]); plt.figure(figsize=(4.5, 3.2))
            plt.plot([0] + ks, [res["field_acc"]] + [res["kshot"][k]["acc"] for k in ks], marker="o")
            plt.xlabel("field images per class"); plt.ylabel("field accuracy"); plt.title("Effect of few-shot field data")
            plt.tight_layout(); plt.savefig(f"{run}/kshot.png", dpi=200); plt.close()
    else:
        save_checkpoint(ck, model, a.model, classes, S, extra=dict(lab_acc=res["lab_acc"]))
    json.dump(res, open(f"{run}/metrics.json", "w"), indent=2)
    print(f"\nSaved model: {ck}\nSaved metrics/plots in: {run}")


if __name__ == "__main__":
    main()
