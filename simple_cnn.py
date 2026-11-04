"""
TerraVision - Experiment 1: SimpleCNN baseline (trained from scratch)

Trains a small 4-block CNN on EuroSAT RGB, grouped into 3 classes:
    Forest   <- Forest
    Urban <- Residential, Industrial, Highway
    Other    <- AnnualCrop, HerbaceousVegetation, Pasture, PermanentCrop, River, SeaLake

"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Class grouping (10 EuroSAT classes to 3)
# ---------------------------------------------------------------------------
GROUPS = {
    "Forest": ["Forest"],
    "Built-up": ["Residential", "Industrial", "Highway"],
    "Other": ["AnnualCrop", "HerbaceousVegetation", "Pasture",
              "PermanentCrop", "River", "SeaLake"],
}
CLASS_NAMES = list(GROUPS.keys())
FINE_TO_COARSE = {fine: i for i, (_, fines) in enumerate(GROUPS.items()) for fine in fines}

# EuroSAT RGB channel stats (computed on the full dataset, 0-1 scale)
MEAN = [0.3444, 0.3803, 0.4078]
STD = [0.2037, 0.1366, 0.1148]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class Coarse(Dataset):
    """Wraps a 10-class EuroSAT dataset and remaps labels to 3 classes."""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform
        self.fine_names = base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, fine = self.base[idx]
        return self.transform(img), FINE_TO_COARSE[self.fine_names[fine]]


def load_base(args):
    if args.download:
        return datasets.EuroSAT(root=args.root, download=True)
    return datasets.ImageFolder(args.data_dir)


def make_split(labels, seed, path):
    """70/15/15 stratified split, saved so every model uses the same test set."""
    if os.path.exists(path):
        with open(path) as f:
            print(f"Reusing split from {path}")
            return json.load(f)
    idx = np.arange(len(labels))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=np.array(labels)[tmp], random_state=seed)
    split = {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}
    with open(path, "w") as f:
        json.dump(split, f)
    print(f"Saved new split to {path}")
    return split


def build_loaders(args):
    base = load_base(args)
    fine_labels = [y for _, y in base.samples] if hasattr(base, "samples") else base.targets
    labels = [FINE_TO_COARSE[base.classes[y]] for y in fine_labels]

    split_path = args.split or os.path.join(args.out, "split.json")
    split = make_split(labels, args.seed, split_path)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = Subset(Coarse(base, train_tf), split["train"])
    val_ds = Subset(Coarse(base, eval_tf), split["val"])
    test_ds = Subset(Coarse(base, eval_tf), split["test"])

    counts = np.bincount([labels[i] for i in split["train"]], minlength=3)
    print("Train class counts:", dict(zip(CLASS_NAMES, counts.tolist())))

    kw = dict(batch_size=args.batch_size, num_workers=args.workers)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
            counts)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """4 conv blocks: 64x64 -> 32 -> 16 -> 8 -> 4, then global pool + linear.
    in_channels is a parameter so the same net works for RGB (3) or RGB+NIR"""

    def __init__(self, in_channels=3, num_classes=3, width=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, width),
            conv_block(width, width * 2),
            conv_block(width * 2, width * 4),
            conv_block(width * 4, width * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(width * 8, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, trues = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds += logits.argmax(1).cpu().tolist()
            trues += y.cpu().tolist()
    n = len(trues)
    return (total_loss / n, accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro"), trues, preds)


def plot_curves(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_f1"], label="train")
    ax[1].plot(hist["val_f1"], label="val")
    ax[1].set_title("Macro-F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cm(trues, preds, path):
    cm = confusion_matrix(trues, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), CLASS_NAMES); ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("SimpleCNN - test (row-normalized)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/EuroSAT/2750",
                   help="EuroSAT RGB folder with one subfolder per class")
    p.add_argument("--download", action="store_true", help="download EuroSAT via torchvision")
    p.add_argument("--root", default="data", help="download location for --download")
    p.add_argument("--out", default="results/simple_cnn")
    p.add_argument("--split", default=None, help="path to an existing split.json to reuse")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weights", action="store_true",
                   help="weight the loss by inverse class frequency (experiment 7)")
    p.add_argument("--patience", type=int, default=7, help="early stopping on val macro-F1")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_dl, val_dl, test_dl, counts = build_loaders(args)

    model = SimpleCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleCNN parameters: {n_params:,}")

    weight = None
    if args.class_weights:
        weight = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32).to(device)
        print("Class weights:", weight.cpu().numpy().round(3))
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    hist = {k: [] for k in ["train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"]}
    best_f1, bad_epochs, start = -1, 0, time.time()
    ckpt = os.path.join(args.out, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        tl, ta, tf, _, _ = run_epoch(model, train_dl, device, criterion, optimizer)
        vl, va, vf, _, _ = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        for k, v in zip(hist, [tl, ta, tf, vl, va, vf]):
            hist[k].append(v)
        print(f"Epoch {epoch:02d} | train loss {tl:.3f} acc {ta:.3f} F1 {tf:.3f} "
              f"| val loss {vl:.3f} acc {va:.3f} F1 {vf:.3f}")

        if vf > best_f1:
            best_f1, bad_epochs = vf, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start

    # Final test with the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc, test_f1, trues, preds = run_epoch(model, test_dl, device, criterion)
    report = classification_report(trues, preds, target_names=CLASS_NAMES,
                                   output_dict=True, digits=4)
    print("\nTEST RESULTS")
    print(classification_report(trues, preds, target_names=CLASS_NAMES, digits=4))

    metrics = {
        "model": "SimpleCNN (from scratch)",
        "params": n_params,
        "epochs_run": len(hist["train_loss"]),
        "train_time_sec": round(train_time, 1),
        "best_val_macro_f1": best_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "per_class": {c: report[c] for c in CLASS_NAMES},
        "config": vars(args),
    }
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    np.save(os.path.join(args.out, "test_preds.npy"), np.array([trues, preds]))
    plot_curves(hist, os.path.join(args.out, "curves.png"))
    plot_cm(trues, preds, os.path.join(args.out, "confusion_matrix.png"))
    print(f"Saved everything to {args.out}/")


if __name__ == "__main__":
    main()
