"""Metrics and plots for the report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works on servers / Colab)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .data import load_image, to_display


def compute_metrics(targets, preds, class_names) -> dict:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(targets, preds, labels=labels, zero_division=0)
    return {
        "accuracy": float((np.asarray(targets) == np.asarray(preds)).mean()),
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(class_names)},
    }


def text_report(targets, preds, class_names) -> str:
    return classification_report(targets, preds, labels=list(range(len(class_names))),
                                 target_names=class_names, digits=4, zero_division=0)


def plot_confusion_matrix(targets, preds, class_names, path: str | Path):
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    size = max(5, 0.6 * len(class_names) + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (counts, row %)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: list[dict], path: str | Path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key, ax in zip(["loss", "acc", "macro_f1"], axes):
        if f"train_{key}" in history[0]:
            ax.plot(epochs, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="val")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(samples, targets, preds, probs, class_names, bands, path,
                       max_images: int = 24):
    """Grid of the most confident mistakes -- the fastest way to do error analysis."""
    wrong = np.where(targets != preds)[0]
    if len(wrong) == 0:
        return
    wrong = wrong[np.argsort(-probs[wrong, preds[wrong]])][:max_images]
    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(to_display(load_image(samples[i][0], bands)))
        ax.set_title(f"T:{class_names[targets[i]]}\nP:{class_names[preds[i]]} "
                     f"({probs[i, preds[i]]:.0%})", fontsize=8)
    fig.suptitle("Most confident misclassifications (T = true, P = predicted)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
