"""
TerraVision - Experiment 2: Does ImageNet transfer learning help?
(also covers Exp 3 linear probe, Exp 4 architectures, Exp 5 input resolution)

Reuses the exact data pipeline from simple_cnn.py 
(same 3 classes, same split.json, same augmentation/normalization), so results are directly
comparable to the SimpleCNN baseline

Experiment 2 = run these two and compare against results/simple_cnn/:
    python transfer_learning.py --arch resnet50 --pretrained --out results/resnet50_pretrained
    python transfer_learning.py --arch resnet50 --scratch    --out results/resnet50_scratch

Other experiments with the same script:
    Exp 3 (linear probe):   --arch resnet50 --pretrained --freeze  --out results/resnet50_probe
    Exp 4 (architectures):  --arch resnet18 / efficientnet_b0      --out results/<name>
    Exp 5 (resolution):     --img-size 128  or  --img-size 224     --out results/<name>
    Exp 7 (class weights):  add --class-weights

Data flags are the same as simple_cnn.py:
    --download                 (torchvision EuroSAT under ./data)
    --data-dir data/EuroSAT/2750

Outputs (in --out):
    config.json, best_model.pt (best val macro-F1), history.json,
    test_metrics.json, confusion_matrix.png, curves.png
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same data pipeline as the baseline -> same split, same classes, fair comparison

from simple_cnn import CLASS_NAMES, build_loaders, get_device, set_seed 

NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Wrapped(nn.Module):
    """Optionally upsamples 64x64 EuroSAT patches before the backbone (Exp 5)."""

    def __init__(self, backbone, img_size):
        super().__init__()
        self.backbone = backbone
        self.img_size = img_size

    def forward(self, x):
        if self.img_size and x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        return self.backbone(x)


def build_model(arch, pretrained, freeze, img_size):
    if arch == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        head = m.fc
    elif arch == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        head = m.fc
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
        head = m.classifier[1]
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if freeze:  # linear probe: only the new head trains
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return Wrapped(m, img_size)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, criterion, optimizer=None, freeze=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze:
        model.eval()  # keep frozen BatchNorm stats fixed during linear probe
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.append(out.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return (total_loss / len(targets),
            accuracy_score(targets, preds),
            f1_score(targets, preds, average="macro"),
            preds, targets)


def save_plots(history, cm, out):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_f1"] for h in history], label="train")
    ax[1].plot(ep, [h["val_f1"] for h in history], label="val")
    ax[1].set_title("Macro-F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Test confusion matrix")
    fig.tight_layout(); fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet50",
                   choices=["resnet50", "resnet18", "efficientnet_b0"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pretrained", action="store_true", help="start from ImageNet weights")
    g.add_argument("--scratch", action="store_true", help="random init (Exp 2 control)")
    p.add_argument("--freeze", action="store_true", help="linear probe: train only the head")
    p.add_argument("--img-size", type=int, default=64, help="upsample input (Exp 5)")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-4 pretrained fine-tune, 1e-3 scratch/probe")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    # data flags (same as simple_cnn.py)
    p.add_argument("--download", action="store_true")
    p.add_argument("--root", default="data")
    p.add_argument("--data-dir", default="data/EuroSAT/2750")
    p.add_argument("--split", default="results/simple_cnn/split.json",
                   help="reuse the baseline's split so test sets match")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.freeze and args.scratch:
        p.error("--freeze only makes sense with --pretrained")
    if not os.path.exists(args.split):
        p.error(f"{args.split} not found. Run simple_cnn.py first, or pass --split.")

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    lr = args.lr or (1e-4 if args.pretrained and not args.freeze else 1e-3)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "lr_used": lr, "device": str(device)}, f, indent=2)

    data_args = SimpleNamespace(download=args.download, root=args.root,
                                data_dir=args.data_dir, split=args.split, out=args.out,
                                seed=args.seed, batch_size=args.batch_size,
                                workers=args.workers)
    train_loader, val_loader, test_loader, counts = build_loaders(data_args)

    model = build_model(args.arch, args.pretrained, args.freeze, args.img_size).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = "linear probe" if args.freeze else ("fine-tune" if args.pretrained else "from scratch")
    print(f"Device: {device} | {args.arch} ({mode}) | input {args.img_size}px | lr {lr}")
    print(f"Parameters: {n_total:,} total, {n_train:,} trainable")

    weight = None
    if args.class_weights:
        weight = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        print("Class weights:", [round(w, 2) for w in weight.tolist()])
        weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                                  lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt = os.path.join(args.out, "best_model.pt")
    history, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_loader, device, criterion,
                                                 optimizer, args.freeze)
        va_loss, va_acc, va_f1, _, _ = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1))
        flag = ""
        if va_f1 > best_f1:
            best_f1, bad_epochs = va_f1, 0
            torch.save(model.state_dict(), ckpt)
            flag = "  * best, saved"
        else:
            bad_epochs += 1
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} F1 {tr_f1:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} F1 {va_f1:.3f} | "
              f"{time.time() - t0:.0f}s{flag}")
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val F1 gain for {args.patience} epochs)")
            break

    # Test with the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, te_acc, te_f1, preds, targets = run_epoch(model, test_loader, device, criterion)
    report = classification_report(targets, preds, target_names=CLASS_NAMES,
                                   digits=4, output_dict=True)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    metrics = dict(model=args.arch, mode=mode, img_size=args.img_size,
                   class_weights=args.class_weights, best_val_f1=best_f1,
                   test_accuracy=te_acc, test_macro_f1=te_f1,
                   per_class=report, confusion_matrix=cm.tolist(),
                   trainable_params=n_train, total_params=n_total)
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_plots(history, cm, args.out)

    print("\n" + "=" * 60 + "\nTEST RESULTS (best checkpoint)\n" + "=" * 60)
    print(classification_report(targets, preds, target_names=CLASS_NAMES, digits=4))
    print(f"Accuracy {te_acc:.4f} | Macro-F1 {te_f1:.4f}")
    print(f"Everything saved to {args.out}/")


if __name__ == "__main__":
    main()