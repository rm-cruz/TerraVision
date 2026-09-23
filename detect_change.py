"""Before/after land-cover change detection on two co-registered scenes.

The scenes are cut into tiles, each tile is classified, and the two class maps
are compared. Outputs: class maps, a change map, a from->to transition matrix,
and area statistics (km^2 if you pass the pixel size).

Scenes should cover the SAME area at the SAME resolution (e.g. two Sentinel-2
exports of one bounding box from different years). For EuroSAT-trained models,
use 10 m/pixel imagery and --tile 64 so tiles match the training scale.

python detect_change.py --checkpoint runs/.../best.pt \
    --before data/scenes/amazon_2018.tif --after data/scenes/amazon_2024.tif \
    --out reports/amazon_change --pixel-size-m 10
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from satclass.data import load_image, to_display
from satclass.models import load_checkpoint
from satclass.scene import classify_scene, transition_matrix
from satclass.utils import get_device, save_json

KNOWN_COLORS = {"Forest": "#1b7837", "Urban": "#d73027", "Other": "#bdbdbd"}


def class_colors(names):
    fallback = plt.get_cmap("tab10").colors
    return [KNOWN_COLORS.get(n, fallback[i % 10]) for i, n in enumerate(names)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", default="reports/change")
    ap.add_argument("--tile", type=int, default=64, help="tile size in scene pixels")
    ap.add_argument("--pixel-size-m", type=float, default=None, help="e.g. 10 for Sentinel-2 RGB/NIR")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    names = ckpt["class_names"]
    k = len(names)

    before = load_image(args.before, ckpt["bands"])
    after = load_image(args.after, ckpt["bands"])
    if before.shape != after.shape:
        h, w = min(before.shape[1], after.shape[1]), min(before.shape[2], after.shape[2])
        print(f"[warn] scene sizes differ {tuple(before.shape)} vs {tuple(after.shape)}; "
              f"cropping both to {h}x{w}. Make sure they are co-registered!")
        before, after = before[:, :h, :w], after[:, :h, :w]

    map_b, prob_b = classify_scene(model, before, ckpt, device, tile=args.tile)
    map_a, prob_a = classify_scene(model, after, ckpt, device, tile=args.tile)
    trans = transition_matrix(map_b, map_a, k)
    n_tiles = map_b.size

    # ---- area statistics ------------------------------------------------- #
    tile_km2 = (args.tile * args.pixel_size_m / 1000) ** 2 if args.pixel_size_m else None
    summary = {"tiles": int(n_tiles), "grid": list(map_b.shape), "tile_px": args.tile,
               "tile_area_km2": tile_km2, "classes": {}}
    for i, n in enumerate(names):
        b, a = int((map_b == i).sum()), int((map_a == i).sum())
        entry = {"before_tiles": b, "after_tiles": a,
                 "before_pct": 100 * b / n_tiles, "after_pct": 100 * a / n_tiles,
                 "change_pct_points": 100 * (a - b) / n_tiles,
                 "relative_change_pct": (100 * (a - b) / b) if b else None}
        if tile_km2:
            entry.update(before_km2=b * tile_km2, after_km2=a * tile_km2, change_km2=(a - b) * tile_km2)
        summary["classes"][n] = entry
    summary["changed_tiles_pct"] = 100 * float((map_b != map_a).mean())
    summary["mean_confidence"] = {"before": float(prob_b.max(-1).mean()),
                                  "after": float(prob_a.max(-1).mean())}
    save_json(summary, out / "summary.json")

    with open(out / "transitions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from \\ to"] + names)
        for i, n in enumerate(names):
            w.writerow([n] + trans[i].tolist())

    # ---- figure ---------------------------------------------------------- #
    cmap = ListedColormap(class_colors(names))
    changed = (map_b != map_a).astype(float)
    if "Forest" in names:  # highlight forest loss specifically
        f_idx = names.index("Forest")
        changed[(map_b == f_idx) & (map_a != f_idx)] = 2.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(to_display(before)); axes[0, 0].set_title("Before")
    axes[0, 1].imshow(to_display(after)); axes[0, 1].set_title("After")
    axes[0, 2].imshow(changed, cmap=ListedColormap(["#f0f0f0", "#fdae61", "#7b3294"]),
                      vmin=0, vmax=2, interpolation="nearest")
    axes[0, 2].set_title("Change (orange = any change, purple = forest loss)")
    axes[1, 0].imshow(map_b, cmap=cmap, vmin=0, vmax=k - 1, interpolation="nearest")
    axes[1, 0].set_title("Predicted classes: before")
    axes[1, 1].imshow(map_a, cmap=cmap, vmin=0, vmax=k - 1, interpolation="nearest")
    axes[1, 1].set_title("Predicted classes: after")
    x = np.arange(k)
    axes[1, 2].bar(x - 0.2, [summary["classes"][n]["before_pct"] for n in names], 0.4, label="before")
    axes[1, 2].bar(x + 0.2, [summary["classes"][n]["after_pct"] for n in names], 0.4, label="after")
    axes[1, 2].set_xticks(x, names); axes[1, 2].set_ylabel("% of tiles"); axes[1, 2].legend()
    axes[1, 2].set_title("Land-cover share")
    for ax in axes.flat[:5]:
        ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=n) for n, c in zip(names, class_colors(names))],
               loc="lower center", ncol=k)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "change_report.png", dpi=150)
    plt.close(fig)

    print(f"[change] {summary['changed_tiles_pct']:.1f}% of {n_tiles} tiles changed class")
    for n, e in summary["classes"].items():
        print(f"  {n:10s} {e['before_pct']:5.1f}% -> {e['after_pct']:5.1f}% "
              f"({e['change_pct_points']:+.1f} pts)")
    print(f"[change] outputs in {out}")


if __name__ == "__main__":
    main()
