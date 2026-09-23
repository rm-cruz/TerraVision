"""Download the dataset (first run) and make figures for the proposal/report.

python explore_data.py                          # uses configs/default.yaml
python explore_data.py --set data.class_map=null    # look at the original 10 EuroSAT classes
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from satclass.data import build_data, load_image, to_display
from satclass.utils import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--per-class", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    data = build_data(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = data.class_names

    print(f"classes: {names} | channels: {data.in_channels}")
    for split, counts in data.class_counts.items():
        print(f"{split:5s} {sum(counts.values()):6d}  {counts}")

    # class distribution per split
    fig, ax = plt.subplots(figsize=(max(6, len(names)), 4))
    width = 0.8 / len(data.class_counts)
    for j, (split, counts) in enumerate(data.class_counts.items()):
        ax.bar([i + j * width for i in range(len(names))],
               [counts.get(n, 0) for n in names], width, label=split)
    ax.set_xticks([i + width for i in range(len(names))], names, rotation=30, ha="right")
    ax.set_ylabel("images")
    ax.set_title("Class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)

    # sample grid
    rng = random.Random(0)
    train = data.splits["train"]
    fig, axes = plt.subplots(len(names), args.per_class,
                             figsize=(args.per_class * 1.8, len(names) * 1.9), squeeze=False)
    for i, n in enumerate(names):
        picks = rng.sample([s for s in train if s[1] == i], args.per_class)
        for j, (path, _) in enumerate(picks):
            axes[i, j].imshow(to_display(load_image(path, data.bands)))
            axes[i, j].axis("off")
        axes[i, 0].set_title(n, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=150)
    plt.close(fig)
    print(f"figures saved to {out}")


if __name__ == "__main__":
    main()
