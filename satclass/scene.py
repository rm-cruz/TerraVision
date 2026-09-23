"""Scene-level inference: tile a large image, classify every tile, build a class map.

This is what powers the before/after change detection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def classify_scene(model, img: torch.Tensor, ckpt: dict, device, tile: int = 64,
                   batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Classify non-overlapping ``tile`` x ``tile`` patches of ``img`` (C, H, W in [0, 1]).

    Returns (class_map of shape (rows, cols), prob_map of shape (rows, cols, n_classes)).
    Pixels that don't fill a whole tile at the right/bottom edge are ignored.
    """
    c, h, w = img.shape
    if c != ckpt["in_channels"]:
        raise ValueError(f"Scene has {c} channels but the model expects {ckpt['in_channels']}. "
                         "Check the band selection.")
    rows, cols = h // tile, w // tile
    if rows == 0 or cols == 0:
        raise ValueError(f"Scene ({h}x{w}) is smaller than one tile ({tile}px)")
    img = img[:, : rows * tile, : cols * tile]
    tiles = (img.unfold(1, tile, tile).unfold(2, tile, tile)   # C, rows, cols, t, t
                .permute(1, 2, 0, 3, 4).reshape(-1, c, tile, tile))

    mean = torch.tensor(ckpt["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(ckpt["std"]).view(1, -1, 1, 1)
    size = ckpt["img_size"]
    all_probs = []
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start: start + batch_size].float()
        if size != tile:
            batch = F.interpolate(batch, size=(size, size), mode="bilinear",
                                  align_corners=False, antialias=True)
        batch = (batch - mean) / std
        all_probs.append(model(batch.to(device)).float().softmax(1).cpu())
    probs = torch.cat(all_probs).numpy().reshape(rows, cols, -1)
    return probs.argmax(-1), probs


def transition_matrix(before: np.ndarray, after: np.ndarray, n_classes: int) -> np.ndarray:
    """m[i, j] = number of tiles that went from class i (before) to class j (after)."""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (before.ravel(), after.ravel()), 1)
    return m
