
import os, json, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# SETTINGS
# =========================================================

SEED = 230646

EPOCHS = 20            # custom models
FT_EPOCHS = 20         # fine-tuning of pretrained models

BATCH = 128

OUT = "results"
os.makedirs(OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = r"C:\Users\THINAL THILAKARATHNE\Desktop\eurosat"


# =========================================================
# RANDOM SEED
# =========================================================

def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# =========================================================
# NORMALIZATION VALUES
# =========================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# =========================================================
# DATA LOADERS
# =========================================================

def get_loaders():

    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=6),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        transforms.RandomApply(
            [transforms.RandomRotation((90, 90))],
            p=0.5
        ),

        transforms.ColorJitter(
            0.2, 0.2, 0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        ),
    ])


    eval_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        ),
    ])


    ds_train = datasets.ImageFolder(
        root=DATA_DIR,
        transform=train_tf
    )

    ds_eval = datasets.ImageFolder(
        root=DATA_DIR,
        transform=eval_tf
    )


    n = len(ds_train)

    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va


    # Same fixed split used for custom models
    g = torch.Generator().manual_seed(SEED)

    perm = torch.randperm(
        n,
        generator=g
    ).tolist()


    idx_tr = perm[:n_tr]

    idx_va = perm[
        n_tr:n_tr + n_va
    ]

    idx_te = perm[
        n_tr + n_va:
    ]


    # Training set uses augmentation
    tr = Subset(
        ds_train,
        idx_tr
    )

    # Validation and test use clean images
    va = Subset(
        ds_eval,
        idx_va
    )

    te = Subset(
        ds_eval,
        idx_te
    )


    mk = lambda d, sh: DataLoader(
        d,
        batch_size=BATCH,
        shuffle=sh,
        num_workers=0,
        pin_memory=True
    )


    print(
        f"Split sizes  train={n_tr}  "
        f"val={n_va}  "
        f"test={n_te}  "
        f"classes={ds_train.classes}"
    )


    return (
        mk(tr, True),
        mk(va, False),
        mk(te, False),
        ds_train.classes
    )





# =========================================================
# BUILD PRETRAINED MODELS
# =========================================================

def build_pretrained(
    name,
    nc=10
):

    # -----------------------------------------------------
    # MobileNetV2
    # -----------------------------------------------------

    if name == "mobilenet_v2":

        m = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )


        # Replace ImageNet classifier
        # with classifier for 10 EuroSAT classes

        m.classifier[1] = nn.Linear(
            m.last_channel,
            nc
        )


    # -----------------------------------------------------
    # SqueezeNet 1.1
    # -----------------------------------------------------

    elif name == "squeezenet1_1":

        m = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        )


        # Replace original 1000-class classifier
        # with 10-class classifier

        m.classifier[1] = nn.Conv2d(
            512,
            nc,
            kernel_size=1
        )


        m.num_classes = nc


    else:

        raise ValueError(name)


    return m


# =========================================================
# TRAINING FUNCTION
# =========================================================

def train(
    model,
    tr,
    va,
    optimizer,
    epochs,
    tag
):

    crit = nn.CrossEntropyLoss()

    model.to(DEVICE)


    hist = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }


    for ep in range(epochs):

        model.train()

        t0 = time.time()

        run = 0.0
        n = 0


        for x, y in tr:

            x = x.to(DEVICE)
            y = y.to(DEVICE)


            optimizer.zero_grad()


            loss = crit(
                model(x),
                y
            )


            loss.backward()

            optimizer.step()


            run += (
                loss.item() * len(y)
            )

            n += len(y)


        if DEVICE == "cuda":
            torch.cuda.synchronize()


        et = time.time() - t0


        vl, vacc = evaluate_loss(
            model,
            va,
            crit
        )


        hist["train_loss"].append(
            run / n
        )

        hist["val_loss"].append(
            vl
        )

        hist["val_acc"].append(
            vacc
        )

        hist["epoch_time"].append(
            et
        )


        print(
            f"[{tag}] "
            f"ep {ep+1:02d}/{epochs}  "
            f"train {run/n:.4f}  "
            f"val {vl:.4f}  "
            f"acc {vacc:.4f}  "
            f"{et:.1f}s"
        )


    return hist


# =========================================================
# VALIDATION
# =========================================================

def evaluate_loss(
    model,
    loader,
    crit
):

    model.eval()

    tot = 0.0
    n = 0
    correct = 0


    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)


            out = model(x)


            tot += (
                crit(out, y).item()
                * len(y)
            )


            correct += (
                out.argmax(1) == y
            ).sum().item()


            n += len(y)


    return (
        tot / n,
        correct / n
    )


# =========================================================
# TEST METRICS
# =========================================================

def test_metrics(
    model,
    te,
    classes,
    tag
):

    model.eval()

    ys = []
    ps = []


    with torch.no_grad():

        for x, y in te:

            prediction = model(
                x.to(DEVICE)
            ).argmax(1)


            ps += (
                prediction
                .cpu()
                .tolist()
            )

            ys += y.tolist()


    # Accuracy

    acc = accuracy_score(
        ys,
        ps
    )


    # Macro precision, recall and F1

    p, r, f, _ = (
        precision_recall_fscore_support(
            ys,
            ps,
            average="macro",
            zero_division=0
        )
    )


    # Confusion matrix

    cm = confusion_matrix(
        ys,
        ps
    )


    plt.figure(
        figsize=(7, 6)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )


    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.title(
        f"Confusion matrix - {tag}"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    plt.tight_layout()

    plt.show()


    # Per-class metrics

    per_class = (
        precision_recall_fscore_support(
            ys,
            ps,
            zero_division=0
        )
    )


    return {
        "accuracy": acc,

        "precision_macro": p,

        "recall_macro": r,

        "f1_macro": f,

        "precision_per_class":
            per_class[0].tolist(),

        "recall_per_class":
            per_class[1].tolist(),

        "confusion_matrix":
            cm.tolist()
    }


# =========================================================
# MODEL PARAMETER COUNT
# =========================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    return total, trainable


# =========================================================
# MODEL SIZE
# =========================================================

def model_size_mb(model):

    # Calculate size based on parameter tensors

    size_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )


    # Include buffers such as BatchNorm statistics

    size_bytes += sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )


    size_mb = (
        size_bytes / (1024 ** 2)
    )


    return size_mb


# =========================================================
# GET SAME DATA SPLITS
# =========================================================

seed_all()

tr, va, te, classes = get_loaders()


# =========================================================
# FINE-TUNE MOBILE-NET V2 AND SQUEEZE-NET 1.1
# =========================================================

ft_hists = {}

ft_results = {}

model_info = {}


for name in [
    "mobilenet_v2",
    "squeezenet1_1"
]:

    print("\n======================================")

    print(
        "Fine-tuning:",
        name
    )

    print("======================================")


    # -----------------------------------------------------
    # Reset random seed
    # -----------------------------------------------------

    seed_all()


    # -----------------------------------------------------
    # Create pretrained model
    # -----------------------------------------------------

    model = build_pretrained(
        name,
        nc=len(classes)
    )


    # -----------------------------------------------------
    # Count parameters BEFORE training
    # -----------------------------------------------------

    total_params, trainable_params = (
        count_parameters(model)
    )


    # -----------------------------------------------------
    # Calculate model size
    # -----------------------------------------------------

    size_mb = model_size_mb(
        model
    )


    model_info[name] = {
        "total_parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "model_size_MB":
            size_mb
    }


    print("\nModel information:")

    print(
        "Total parameters:",
        f"{total_params:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )

    print(
        "Model size:",
        f"{size_mb:.2f} MB"
    )


    # -----------------------------------------------------
    # Adam optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )


    # -----------------------------------------------------
    # Fine-tune model
    # -----------------------------------------------------

    history = train(
        model,
        tr,
        va,
        optimizer,
        FT_EPOCHS,
        f"{name}_FineTune"
    )


    ft_hists[name] = history


    # -----------------------------------------------------
    # Test model
    # -----------------------------------------------------

    results = test_metrics(
        model,
        te,
        classes,
        f"{name}_FineTune"
    )


    ft_results[name] = results


    # -----------------------------------------------------
    # Print final test results
    # -----------------------------------------------------

    print("\n==============================")

    print(
        f"{name} - FINAL TEST RESULTS"
    )

    print("==============================")


    print(
        f"Accuracy : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{results['precision_macro']:.4f}"
    )

    print(
        f"Recall   : "
        f"{results['recall_macro']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{results['f1_macro']:.4f}"
    )


    # -----------------------------------------------------
    # Save fine-tuned model
    # -----------------------------------------------------

    save_path = os.path.join(
        OUT,
        f"{name}_finetuned.pth"
    )


    torch.save(
        model.state_dict(),
        save_path
    )


    print(
        "Model saved successfully:"
    )

    print(save_path)


# =========================================================
# FINAL COMPARISON
# =========================================================

print("\n\n==============================================")

print(
    "SOTA FINE-TUNING FINAL COMPARISON"
)

print("==============================================")


print(
    f"{'Model':20s}"
    f"{'Parameters':>15s}"
    f"{'Size (MB)':>12s}"
    f"{'Accuracy':>12s}"
    f"{'Precision':>12s}"
    f"{'Recall':>12s}"
    f"{'F1':>12s}"
)

print("-" * 95)


for name in [
    "mobilenet_v2",
    "squeezenet1_1"
]:

    info = model_info[name]

    result = ft_results[name]


    print(
        f"{name:20s}"
        f"{info['total_parameters']:>15,}"
        f"{info['model_size_MB']:>12.2f}"
        f"{result['accuracy']:>12.4f}"
        f"{result['precision_macro']:>12.4f}"
        f"{result['recall_macro']:>12.4f}"
        f"{result['f1_macro']:>12.4f}"
    )


print("\nFine-tuning completed successfully.")
