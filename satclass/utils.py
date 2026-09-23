"""Small shared helpers: config loading, seeding, devices, run folders."""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply dotted overrides like ``train.lr=0.001``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)  # parses numbers, bools, lists, null
    return cfg


def save_yaml(obj: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


def make_run_dir(cfg: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{cfg['model']['name']}_{cfg['data']['dataset']}"
    run_dir = Path(cfg.get("output_dir", "runs")) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
