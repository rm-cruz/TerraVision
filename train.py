"""Train a land-cover classifier.

Examples
--------
python train.py                                            # defaults (ResNet-50, EuroSAT RGB, 3 classes)
python train.py --set model.name=simplecnn model.pretrained=false
python train.py --set data.limit_per_class=200 train.epochs=2 data.num_workers=0   # quick CPU sanity check
python train.py --config configs/eurosat_ms_rgbn.yaml     # RGB + NIR
"""
from __future__ import annotations

import argparse
import csv
import time

import torch
import torch.nn as nn

from satclass.data import build_data, class_weights
from satclass.engine import run_epoch
from satclass.metrics import (compute_metrics, plot_confusion_matrix, plot_history,
                              plot_misclassified, text_report)
from satclass.models import build_model, count_params, load_checkpoint, save_checkpoint
from satclass.utils import (get_device, load_config, make_run_dir, save_json, save_yaml,
                            seed_everything)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. train.lr=0.001 model.name=resnet18")
    return ap.parse_args()


def build_optimizer(cfg, params):
    t = cfg["train"]
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"])
    raise ValueError(f"Unknown optimizer {t['optimizer']!r}")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    device = get_device()

    # ---- data ------------------------------------------------------------ #
    data = build_data(cfg)
    print(f"[data] classes={data.class_names} channels={data.in_channels}")
    for split, counts in data.class_counts.items():
        print(f"[data] {split:5s}: {counts}")

    run_dir = make_run_dir(cfg)
    save_yaml(cfg, run_dir / "config.yaml")
    save_json({k: v for k, v in data.splits.items()}, run_dir / "splits.json")
    print(f"[run] saving to {run_dir}")

    # ---- model ----------------------------------------------------------- #
    m = cfg["model"]
    model = build_model(m["name"], len(data.class_names), data.in_channels,
                        m["pretrained"], m.get("freeze_backbone", False)).to(device)
    total, trainable = count_params(model)
    print(f"[model] {m['name']} on {device}: {total:,} params ({trainable:,} trainable)")

    t = cfg["train"]
    weight = class_weights(data.splits["train"], len(data.class_names)).to(device) \
        if t.get("class_weighted_loss") else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=t.get("label_smoothing", 0.0))
    optimizer = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"]) \
        if t.get("scheduler") == "cosine" else None
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ---- training loop --------------------------------------------------- #
    history, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, t["epochs"] + 1):
        start = time.time()
        tr = run_epoch(model, data.loaders["train"], criterion, device, optimizer, scaler,
                       desc=f"epoch {epoch} train")
        va = run_epoch(model, data.loaders["val"], criterion, device, desc=f"epoch {epoch} val")
        if scheduler:
            scheduler.step()

        tr_m = compute_metrics(tr["targets"], tr["preds"], data.class_names)
        va_m = compute_metrics(va["targets"], va["preds"], data.class_names)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": tr["loss"], "train_acc": tr["acc"], "train_macro_f1": tr_m["macro_f1"],
               "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va_m["macro_f1"],
               "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va_m['macro_f1']:.4f} | "
              f"{row['seconds']}s")

        # model selection on validation macro-F1 (better than accuracy under class imbalance)
        if va_m["macro_f1"] > best_f1:
            best_f1, bad_epochs = va_m["macro_f1"], 0
            save_checkpoint(run_dir / "best.pt", model, cfg, data, epoch, va_m)
        else:
            bad_epochs += 1
            if bad_epochs >= t.get("early_stopping_patience", 10**9):
                print(f"[train] early stopping (no val improvement for {bad_epochs} epochs)")
                break

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, run_dir / "curves.png")

    # ---- final test evaluation with the best checkpoint ------------------ #
    model, ckpt = load_checkpoint(run_dir / "best.pt", device)
    te = run_epoch(model, data.loaders["test"], criterion, device, desc="test")
    te_m = compute_metrics(te["targets"], te["preds"], data.class_names)
    te_m["best_epoch"] = ckpt["epoch"]
    save_json(te_m, run_dir / "test_metrics.json")
    report = text_report(te["targets"], te["preds"], data.class_names)
    (run_dir / "test_report.txt").write_text(report)
    plot_confusion_matrix(te["targets"], te["preds"], data.class_names, run_dir / "confusion_matrix.png")
    plot_misclassified(data.splits["test"], te["targets"], te["preds"], te["probs"],
                       data.class_names, data.bands, run_dir / "misclassified.png")

    print(f"\n[test] best epoch {ckpt['epoch']} | acc {te_m['accuracy']:.4f} | "
          f"macro-F1 {te_m['macro_f1']:.4f}\n{report}")
    print(f"[run] outputs in {run_dir}")


if __name__ == "__main__":
    main()
