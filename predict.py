"""Classify individual image tiles.

python predict.py --checkpoint runs/.../best.pt --input path/to/tile.jpg
python predict.py --checkpoint runs/.../best.pt --input path/to/folder --output preds.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from satclass.data import IMAGE_EXTS, build_transforms, load_image
from satclass.models import load_checkpoint
from satclass.utils import get_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="an image file or a folder of images")
    ap.add_argument("--output", help="optional CSV path")
    args = ap.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    names = ckpt["class_names"]
    tf = build_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"], train=False)

    inp = Path(args.input)
    paths = sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if inp.is_dir() else [inp]

    rows = []
    with torch.no_grad():
        for p in paths:
            x = tf(load_image(p, ckpt["bands"])).unsqueeze(0).to(device)
            probs = model(x).softmax(1)[0].cpu()
            k = int(probs.argmax())
            rows.append([str(p), names[k], f"{probs[k]:.4f}"] + [f"{v:.4f}" for v in probs.tolist()])
            print(f"{p.name:40s} -> {names[k]:10s} ({probs[k]:.1%})")

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "pred", "confidence"] + [f"p_{n}" for n in names])
            w.writerows(rows)
        print(f"[predict] wrote {args.output}")


if __name__ == "__main__":
    main()
