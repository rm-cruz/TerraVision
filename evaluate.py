"""Evaluate a trained run in detail (per-image predictions for error analysis).

Examples
--------
python evaluate.py --run runs/20261001-120000_resnet50_eurosat_rgb            # its test split
python evaluate.py --run runs/... --split val
python evaluate.py --run runs/... --data-dir data/my_region                    # a new labeled folder
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from satclass.data import FileListDataset, apply_class_map, build_transforms, scan_folder
from satclass.engine import run_epoch
from satclass.metrics import (compute_metrics, plot_confusion_matrix, plot_misclassified,
                              text_report)
from satclass.models import load_checkpoint
from satclass.utils import get_device, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory created by train.py")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data-dir", help="evaluate on this folder (<class>/<images>) instead of a saved split")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    run = Path(args.run)
    device = get_device()
    model, ckpt = load_checkpoint(run / "best.pt", device)
    names = ckpt["class_names"]

    if args.data_dir:
        raw, folder_classes = scan_folder(args.data_dir)
        if set(folder_classes) <= set(names):     # folders already use our class names
            samples, _ = apply_class_map(raw, {c: [c] for c in names})
        else:                                      # original names -> reuse training class_map
            samples, _ = apply_class_map(raw, ckpt["config"]["data"].get("class_map"))
        tag = f"eval_{Path(args.data_dir).name}"
    else:
        samples = [tuple(s) for s in json.loads((run / "splits.json").read_text())[args.split]]
        tag = f"eval_{args.split}"

    tf = build_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"], train=False)
    loader = DataLoader(FileListDataset(samples, ckpt["bands"], tf), batch_size=args.batch_size)
    out = run_epoch(model, loader, nn.CrossEntropyLoss(), device, desc=tag)

    out_dir = run / tag
    out_dir.mkdir(exist_ok=True)
    metrics = compute_metrics(out["targets"], out["preds"], names)
    save_json(metrics, out_dir / "metrics.json")
    report = text_report(out["targets"], out["preds"], names)
    (out_dir / "report.txt").write_text(report)
    plot_confusion_matrix(out["targets"], out["preds"], names, out_dir / "confusion_matrix.png")
    plot_misclassified(samples, out["targets"], out["preds"], out["probs"], names,
                       ckpt["bands"], out_dir / "misclassified.png")

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", "correct"])
        for (path, _), t, p, pr in zip(samples, out["targets"], out["preds"], out["probs"]):
            w.writerow([path, names[t], names[p], f"{pr[p]:.4f}", int(t == p)])

    print(report)
    print(f"[eval] outputs in {out_dir}")


if __name__ == "__main__":
    main()
