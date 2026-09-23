"""Model zoo: a from-scratch baseline CNN plus ImageNet-pretrained backbones.

Pretrained nets expect 3 input channels. For multispectral input (e.g. RGB+NIR)
we swap the first conv layer and initialise the extra channels from the mean
of the pretrained RGB filters, so transfer learning still works.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Baseline: 4 conv blocks -> global average pool -> linear."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_channels, 32), block(32, 64),
                                      block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Return a copy of ``conv`` that accepts ``in_channels`` inputs."""
    if in_channels == conv.in_channels:
        return conv
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight
        if in_channels < w.shape[1]:
            new.weight.copy_(w[:, :in_channels])
        else:
            new.weight[:, : w.shape[1]] = w
            new.weight[:, w.shape[1]:] = w.mean(dim=1, keepdim=True)
            new.weight.mul_(w.shape[1] / in_channels)  # keep activation scale similar
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_model(name: str, num_classes: int, in_channels: int = 3,
                pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(in_channels, num_classes)

    if name in {"resnet18", "resnet50"}:
        ctor, weights = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }[name]
        m = ctor(weights=weights if pretrained else None)
        m.conv1 = adapt_first_conv(m.conv1, in_channels)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head, stem = m.fc, m.conv1
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.features[0][0] = adapt_first_conv(m.features[0][0], in_channels)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head, stem = m.classifier, m.features[0][0]
    else:
        raise ValueError(f"Unknown model {name!r}")

    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        if in_channels != 3:  # the re-initialised stem must learn the new bands
            for p in stem.parameters():
                p.requires_grad = True
    return m


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# Checkpoints carry everything needed to rebuild the model and preprocess input
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, model: nn.Module, cfg: dict, bundle, epoch: int, val_metrics: dict):
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg,
        "class_names": bundle.class_names,
        "in_channels": bundle.in_channels,
        "bands": bundle.bands,
        "mean": bundle.mean,
        "std": bundle.std,
        "img_size": cfg["data"]["img_size"],
        "epoch": epoch,
        "val_metrics": val_metrics,
    }, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Return (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ckpt["config"]["model"]
    model = build_model(mcfg["name"], len(ckpt["class_names"]), ckpt["in_channels"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
