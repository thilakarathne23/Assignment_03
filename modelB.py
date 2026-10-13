import os, json, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,Subset
from torchvision import  datasets, transforms, models
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

SEED = 230646
EPOCHS = 20            # custom models (>= 20 required)
FT_EPOCHS = 10         # fine-tuning of pre-trained nets
BATCH = 128
OUT = "results"
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = r"C:\Users\THINAL THILAKARATHNE\Desktop\eurosat"  # folder that directly contains the class folders


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
def get_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=6),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),                  # valid for top-down satellite imagery
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    ds_train = datasets.ImageFolder(root=DATA_DIR, transform=train_tf)
    ds_eval = datasets.ImageFolder(root=DATA_DIR, transform=eval_tf)

    n = len(ds_train)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    n_te = n - n_tr - n_va

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()
    idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    tr = Subset(ds_train, idx_tr)   # augmented
    va = Subset(ds_eval,  idx_va)   # clean
    te = Subset(ds_eval,  idx_te)   # clean

    mk = lambda d, sh: DataLoader(d, batch_size=BATCH, shuffle=sh,
                                  num_workers=0, pin_memory=True)
    print(f"Split sizes  train={n_tr}  val={n_va}  test={n_te}  classes={ds_train.classes}")
    return mk(tr, True), mk(va, False), mk(te, False), ds_train.classes
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def show_batch(loader, classes, n=20, cols=5):
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:n], labels[:n]
    rows = (len(imgs) + cols - 1) // cols

    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i, (img, lab) in enumerate(zip(imgs, labels)):
        img = (img * STD + MEAN).clamp(0, 1)          # undo normalization
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.permute(1, 2, 0).numpy())      # CHW -> HWC
        plt.title(classes[lab], fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


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
            loss.backward(); optimizer.step(); #scheduler.step()
            run += loss.item() * len(y); n += len(y)
        if DEVICE == "cuda": torch.cuda.synchronize()
        et = time.time() - t0
        vl, vacc = evaluate_loss(model, va, crit)
        hist["train_loss"].append(run / n); hist["val_loss"].append(vl)
        hist["val_acc"].append(vacc); hist["epoch_time"].append(et)
        print(f"[{tag}] ep {ep+1:02d}/{epochs}  train {run/n:.4f}  val {vl:.4f}  acc {vacc:.4f}  {et:.1f}s")
    return hist
def evaluate_loss(model, loader, crit):
    model.eval(); tot, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            tot += crit(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot / n, correct / n
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
    plt.show()
    per_class = precision_recall_fscore_support(ys, ps, zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f,
            "precision_per_class": per_class[0].tolist(), "recall_per_class": per_class[1].tolist(),
            "confusion_matrix": cm.tolist()}
seed_all()
tr, va, te, classes = get_loaders()
# 4. Create optimizer


def make_opt(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)

    if name.startswith("sgd_m"):
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(name[5:])
        )

    raise ValueError(name)


# Optimizers to compare
opt_cfg = {
    "adam_lr1e-3": ("adam", 1e-3),
    "sgd_lr1e-2": ("sgd", 1e-2),
    "sgd_m0.9_lr1e-2": ("sgd_m0.9", 1e-2)
}


# Additional SGD momentum values
sweep = {
    f"sgd_m{b}_lr1e-2": (f"sgd_m{b}", 1e-2)
    for b in (0.5, 0.99)
}


# Combine all optimizers
all_opts = {**opt_cfg, **sweep}

opt_hists = {}
opt_res = {}


# Train ModelB separately with each optimizer
for tag, (opt_name, lr) in all_opts.items():

    print("\n======================================")
    print("Training:", tag)
    print("======================================")

    seed_all()

    # Create a NEW ModelB for each optimizer
    m = ModelB(nc=len(classes))

    # Create optimizer
    optimizer = make_opt(
        opt_name,
        m.parameters(),
        lr
    )

    # Train
    h = train(
        m,
        tr,
        va,
        optimizer,
        EPOCHS,
        f"ModelB/{tag}"
    )

    # Store training history
    opt_hists[tag] = h

    # Store important results
    opt_res[tag] = {
        "final_val_loss": h["val_loss"][-1],
        "best_val_acc": max(h["val_acc"]),
        "epochs_to_90pct_val":
            next(
                (
                    i + 1
                    for i, a in enumerate(h["val_acc"])
                    if a >= 0.90
                ),
                None
            )
    }


# 5. Plot training and validation loss
def plot_optimizer_curves(hists, fname, title):

    plt.figure(figsize=(10, 6))

    for tag, h in hists.items():

        # Training loss
        plt.plot(
            h["train_loss"],
            label=f"{tag} - train"
        )

        # Validation loss
        plt.plot(
            h["val_loss"],
            "--",
            label=f"{tag} - validation"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(title)

    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT, fname),
        dpi=200
    )

    plt.show()
    plt.close()


plot_optimizer_curves(
    opt_hists,
    "ModelB_optimizer_comparison.png",
    "Model B: Optimizer Comparison"
)


# 6. Print comparison results

print("\n======================================")
print("MODEL B OPTIMIZER COMPARISON")
print("======================================")

for tag, result in opt_res.items():

    print(
        f"{tag:25s} "
        f"Final Val Loss = {result['final_val_loss']:.4f}   "
        f"Best Val Acc = {result['best_val_acc']:.4f}   "
        f"Epochs to 90% = {result['epochs_to_90pct_val']}"
    )


# 7. Select the best optimizer

best_tag = max(
    all_opts,
    key=lambda tag: opt_res[tag]["best_val_acc"]
)

best_opt_name, best_lr = all_opts[best_tag]

print("\n======================================")
print("BEST OPTIMIZER FOR MODEL B")
print("======================================")

print("Selected:", best_tag)

print(
    "Best validation accuracy:",
    f"{opt_res[best_tag]['best_val_acc']:.4f}"
)


# 8. Train final ModelB using the best optimizer

seed_all()

model = ModelB(nc=len(classes))

optimizer = make_opt(
    best_opt_name,
    model.parameters(),
    best_lr
)

history = train(
    model,
    tr,
    va,
    optimizer,
    EPOCHS,
    "ModelB_Final"
)


# 9. Test the final trained ModelB

results = test_metrics(
    model,
    te,
    classes,
    "ModelB_Final"
)


# 10. Print final test results

print("\n==============================")
print("MODEL B - FINAL TEST RESULTS")
print("==============================")

print(f"Accuracy : {results['accuracy']:.4f}")
print(f"Precision: {results['precision_macro']:.4f}")
print(f"Recall   : {results['recall_macro']:.4f}")
print(f"F1 Score : {results['f1_macro']:.4f}")


# 11. Save final ModelB

torch.save(
    model.state_dict(),
    os.path.join(OUT, "ModelB.pth")
)

print("ModelB saved successfully.")