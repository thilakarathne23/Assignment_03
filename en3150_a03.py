"""
EN3150 Assignment 03 - Resource-constrained CNN for edge image classification
Dataset : EuroSAT RGB (native 64x64, 10 land-use classes)  -- not CIFAR-10
Models  : Model A (standard CNN), Model B (depthwise separable CNN, <100k params),
          + two pre-trained lightweight nets (MobileNetV2, SqueezeNet1.1)
Run     : python en3150_a03.py            (GPU optional; downloads data + weights once)
Outputs : ./results/  (loss curves, confusion matrices, results.json)

Install : pip install torch torchvision scikit-learn matplotlib seaborn
"""
import os, json, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 230646
EPOCHS = 20            # custom models (>= 20 required)
FT_EPOCHS = 10         # fine-tuning of pre-trained nets
BATCH = 128
OUT = "results"
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- data (Q1)
def get_loaders():
    tf = transforms.Compose([
        transforms.Resize((64, 64)),           # EuroSAT is already 64x64; enforces max size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.EuroSAT(root="data", download=True, transform=tf)
    n = len(ds)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    n_te = n - n_tr - n_va
    g = torch.Generator().manual_seed(SEED)
    tr, va, te = random_split(ds, [n_tr, n_va, n_te], generator=g)
    mk = lambda d, sh: DataLoader(d, batch_size=BATCH, shuffle=sh, num_workers=2, pin_memory=True)
    print(f"Split sizes  train={n_tr}  val={n_va}  test={n_te}  classes={ds.classes}")
    return mk(tr, True), mk(va, False), mk(te, False), ds.classes


# ---------------------------------------------------------------- models (Q2)
class ModelA(nn.Module):
    """Standard CNN: [Conv3x3-ReLU-MaxPool] x3 -> FC. ReLU = compare-with-zero, no exponentials."""
    def __init__(self, nc=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 64->32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16->8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nc))

    def forward(self, x):
        return self.classifier(self.features(x))


class DSConv(nn.Module):
    """Depthwise 3x3 (groups=in) + pointwise 1x1, each followed by BN + ReLU6."""
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu6(self.bn1(self.dw(x)))
        return F.relu6(self.bn2(self.pw(x)))


class ModelB(nn.Module):
    """Lightweight CNN with depthwise separable convs, global average pooling (no big FC)."""
    def __init__(self, nc=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU6())
        self.body = nn.Sequential(
            DSConv(16, 32), nn.MaxPool2d(2),     # 64->32
            DSConv(32, 64), nn.MaxPool2d(2),     # 32->16
            DSConv(64, 128), nn.MaxPool2d(2),    # 16->8
            DSConv(128, 128),
        )
        self.head = nn.Linear(128, nc)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.head(x.mean(dim=(2, 3)))


def build_pretrained(name, nc=10):
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.last_channel, nc)
    elif name == "squeezenet1_1":
        m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Conv2d(512, nc, kernel_size=1)
        m.num_classes = nc
    else:
        raise ValueError(name)
    return m


# ---------------------------------------------------------------- utilities
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def size_on_disk(m, path):
    torch.save(m.state_dict(), path)
    return os.path.getsize(path) / 1024  # KB


def count_macs(model, shape=(1, 3, 64, 64)):
    """Approximate MACs for Conv2d and Linear layers via forward hooks."""
    total = [0]
    def conv_hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        total[0] += out.numel() * k
    def lin_hook(mod, inp, out):
        total[0] += mod.in_features * mod.out_features
    hs = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d): hs.append(mod.register_forward_hook(conv_hook))
        if isinstance(mod, nn.Linear): hs.append(mod.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape).to(next(model.parameters()).device))
    for h in hs: h.remove()
    return total[0]


# ---------------------------------------------------------------- training (Q3, Q4)
def evaluate_loss(model, loader, crit):
    model.eval(); tot, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            tot += crit(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot / n, correct / n


def train(model, tr, va, optimizer, epochs, tag):
    crit = nn.CrossEntropyLoss()
    model.to(DEVICE)
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    for ep in range(epochs):
        model.train(); t0 = time.time(); run, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); optimizer.step()
            run += loss.item() * len(y); n += len(y)
        if DEVICE == "cuda": torch.cuda.synchronize()
        et = time.time() - t0
        vl, vacc = evaluate_loss(model, va, crit)
        hist["train_loss"].append(run / n); hist["val_loss"].append(vl)
        hist["val_acc"].append(vacc); hist["epoch_time"].append(et)
        print(f"[{tag}] ep {ep+1:02d}/{epochs}  train {run/n:.4f}  val {vl:.4f}  acc {vacc:.4f}  {et:.1f}s")
    return hist


def plot_curves(hists, fname, title):
    plt.figure(figsize=(7, 4.5))
    for tag, h in hists.items():
        plt.plot(h["train_loss"], label=f"{tag} train")
        plt.plot(h["val_loss"], "--", label=f"{tag} val")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.title(title)
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=200); plt.close()


def test_metrics(model, te, classes, tag):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in te:
            ps += model(x.to(DEVICE)).argmax(1).cpu().tolist(); ys += y.tolist()
    acc = accuracy_score(ys, ps)
    p, r, f, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion matrix - {tag}")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"cm_{tag}.png"), dpi=200); plt.close()
    per_class = precision_recall_fscore_support(ys, ps, zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f,
            "precision_per_class": per_class[0].tolist(), "recall_per_class": per_class[1].tolist(),
            "confusion_matrix": cm.tolist()}


def make_opt(name, params, lr):
    if name == "adam":         return torch.optim.Adam(params, lr=lr)
    if name == "sgd":          return torch.optim.SGD(params, lr=lr)
    if name.startswith("sgd_m"):  # e.g. sgd_m0.9
        return torch.optim.SGD(params, lr=lr, momentum=float(name[5:]))
    raise ValueError(name)


# ---------------------------------------------------------------- main
def main():
    seed_all()
    tr, va, te, classes = get_loaders()
    results = {}

    # ---- Q2: architecture summary
    for name, cls in [("ModelA", ModelA), ("ModelB", ModelB)]:
        m = cls(len(classes))
        print(f"{name}: params={count_params(m):,}  MACs={count_macs(m):,}")
        for n_, p_ in m.named_parameters():
            print(f"   {n_:35s} {tuple(p_.shape)}  {p_.numel():,}")
    assert count_params(ModelB()) <= 100_000, "Model B exceeds 100k parameters!"

    # ---- Q3: optimizer comparison on Model B (Adam vs SGD vs SGD+momentum, plus momentum sweep)
    opt_cfg = {"adam_lr1e-3": ("adam", 1e-3), "sgd_lr1e-2": ("sgd", 1e-2),
               "sgd_m0.9_lr1e-2": ("sgd_m0.9", 1e-2)}
    sweep = {f"sgd_m{b}_lr1e-2": (f"sgd_m{b}", 1e-2) for b in (0.5, 0.99)}
    opt_hists, opt_res = {}, {}
    for tag, (o, lr) in {**opt_cfg, **sweep}.items():
        seed_all()
        m = ModelB(len(classes))
        h = train(m, tr, va, make_opt(o, m.parameters(), lr), EPOCHS, f"B/{tag}")
        opt_hists[tag] = h
        opt_res[tag] = {"final_val_loss": h["val_loss"][-1], "best_val_acc": max(h["val_acc"]),
                        "epochs_to_90pct_val": next((i + 1 for i, a in enumerate(h["val_acc"]) if a >= .9), None)}
    plot_curves(opt_hists, "optimizer_comparison.png", "Model B: optimizer comparison")
    results["optimizers"] = opt_res
    # choose the best optimizer by validation accuracy (state your justification in the report)
    best_tag = max(opt_cfg, key=lambda t: opt_res[t]["best_val_acc"])
    best_o, best_lr = opt_cfg[best_tag]
    print("Selected optimizer for final training:", best_tag)

    # ---- Q4: final training of A and B
    hists = {}
    for name, cls in [("ModelA", ModelA), ("ModelB", ModelB)]:
        seed_all()
        m = cls(len(classes))
        h = train(m, tr, va, make_opt(best_o, m.parameters(), best_lr), EPOCHS, name)
        hists[name] = h
        r = test_metrics(m.to(DEVICE), te, classes, name)
        r.update({"params": count_params(m), "size_kb": size_on_disk(m, f"{OUT}/{name}.pt"),
                  "sec_per_epoch": float(np.mean(h["epoch_time"])), "macs": count_macs(m)})
        results[name] = r
    plot_curves(hists, "loss_A_vs_B.png", "Train/validation loss: Model A vs Model B")

    # ---- Q5: fine-tune pretrained lightweight nets
    ft_hists = {}
    for name in ["mobilenet_v2", "squeezenet1_1"]:
        seed_all()
        m = build_pretrained(name, len(classes))
        h = train(m, tr, va, torch.optim.Adam(m.parameters(), lr=1e-3), FT_EPOCHS, name)
        ft_hists[name] = h
        r = test_metrics(m.to(DEVICE), te, classes, name)
        r.update({"params": count_params(m), "size_mb": size_on_disk(m, f"{OUT}/{name}.pt") / 1024,
                  "sec_per_epoch": float(np.mean(h["epoch_time"])), "macs": count_macs(m)})
        results[name] = r
    plot_curves(ft_hists, "loss_finetune.png", "Fine-tuning loss curves")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=========== SUMMARY ===========")
    for k in ["ModelA", "ModelB", "mobilenet_v2", "squeezenet1_1"]:
        r = results[k]
        print(f"{k:15s} params={r['params']:>10,}  acc={r['accuracy']:.4f}  "
              f"prec={r['precision_macro']:.4f}  rec={r['recall_macro']:.4f}  "
              f"MACs={r['macs']:,}  s/epoch={r['sec_per_epoch']:.1f}")


if __name__ == "__main__":
    main()
