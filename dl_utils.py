"""PyTorch helpers: transforms, datasets, models, training, evaluation, checkpoints."""
import copy, os, random
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import timm

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_transforms(S):
    norm = T.Normalize(MEAN, STD)
    return {
        "eval": T.Compose([T.Resize((S, S)), T.ToTensor(), norm]),
        "basic": T.Compose([T.Resize((S, S)), T.RandomHorizontalFlip(), T.ToTensor(), norm]),
        "strong": T.Compose([
            T.RandomResizedCrop(S, scale=(0.4, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandAugment(num_ops=2, magnitude=9), T.ColorJitter(0.4, 0.4, 0.4, 0.05),
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3), T.RandomGrayscale(0.05),
            T.ToTensor(), norm, T.RandomErasing(p=0.4, scale=(0.02, 0.25), value="random")]),
    }


class ImgDS(Dataset):
    def __init__(self, samples, tfm, S=224):
        self.s, self.t, self.S = samples, tfm, S

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        p, y = self.s[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.S, self.S))
        return self.t(img), y


def make_loader(samples, tfm, bs, S, train=False, workers=None):
    workers = (0 if os.name == "nt" else 2) if workers is None else workers
    ds = ImgDS(samples, tfm, S)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=workers, pin_memory=AMP,
                      drop_last=train and len(ds) > bs)


def make_model(name, n_classes, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, num_classes=n_classes).to(DEV)


@torch.no_grad()
def predict_probs(model, samples, tfm, bs, S):
    model.eval(); P, L = [], []
    for x, y in make_loader(samples, tfm, bs, S):
        with torch.autocast(device_type=DEV.type, enabled=AMP):
            P.append(model(x.to(DEV)).float().softmax(1).cpu())
        L.append(y)
    return torch.cat(P).numpy(), torch.cat(L).numpy()


def metrics_from_probs(P, L):
    pred = P.argmax(1)
    return accuracy_score(L, pred), f1_score(L, pred, average="macro", labels=sorted(set(L.tolist())), zero_division=0)


def fit(model, train_s, val_s, tfm, epochs, lr, bs, S, tag=""):
    dl = make_loader(train_s, tfm, bs, S, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(dl)))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    hist = dict(loss=[], train_acc=[], val_acc=[])
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train(); tot, correct, n = 0.0, 0, 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast(device_type=DEV.type, enabled=AMP):
                out = model(x); loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
        hist["loss"].append(tot / n); hist["train_acc"].append(correct / n)
        msg = f"[{tag}] epoch {ep+1}/{epochs} loss {tot/n:.4f} train_acc {correct/n:.4f}"
        if val_s:
            P, L = predict_probs(model, val_s, get_transforms(S)["eval"], bs, S)
            va = float((P.argmax(1) == L).mean()); hist["val_acc"].append(va); msg += f" val_acc {va:.4f}"
            if va > best:
                best, best_state = va, copy.deepcopy(model.state_dict())
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def sample_k(pool, k, seed=0):
    rng = random.Random(seed); by = defaultdict(list)
    for p, y in pool:
        by[y].append((p, y))
    out = []
    for v in by.values():
        rng.shuffle(v); out += v[:k]
    return out


def save_checkpoint(path, model, name, classes, S, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(dict(model=name, classes=classes, img_size=S, state_dict=model.state_dict(), extra=extra or {}), path)


def load_checkpoint(path):
    ck = torch.load(path, map_location=DEV)
    m = make_model(ck["model"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["classes"], ck["img_size"]


def preprocess_array(rgb, S):
    """rgb: HxWx3 uint8 numpy array -> 1x3xSxS tensor."""
    return get_transforms(S)["eval"](Image.fromarray(rgb)).unsqueeze(0).to(DEV)
