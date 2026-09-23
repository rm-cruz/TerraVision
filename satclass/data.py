"""Datasets, preprocessing, and train/val/test splitting.

Every image (RGB JPEG or multispectral GeoTIFF) is loaded as a float tensor of
shape (C, H, W) scaled to [0, 1], so the rest of the pipeline doesn't care
where it came from.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Band order inside EuroSAT-MS GeoTIFFs (Sentinel-2 L1C, 13 bands).
EUROSAT_MS_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                    "B08", "B08A", "B09", "B10", "B11", "B12"]


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def resolve_bands(bands: list | None) -> list[int] | None:
    """Turn band names (``"B04"``) or indices into a list of channel indices."""
    if bands is None:
        return None
    out = []
    for b in bands:
        if isinstance(b, int):
            out.append(b)
        elif str(b).upper() in EUROSAT_MS_BANDS:
            out.append(EUROSAT_MS_BANDS.index(str(b).upper()))
        else:
            raise ValueError(f"Unknown band {b!r}; use an index or one of {EUROSAT_MS_BANDS}")
    return out


def load_image(path: str | Path, bands: list[int] | None = None) -> torch.Tensor:
    """Load an image as a float32 tensor (C, H, W) in [0, 1].

    GeoTIFFs with integer reflectance values (Sentinel-2 style, 0-10000) are
    divided by 10000; 8-bit images are divided by 255.
    """
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        raw = tifffile.imread(path)
        scale = 255.0 if raw.dtype == np.uint8 else 10000.0
        arr = raw.astype(np.float32) / scale
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.shape[-1] < arr.shape[0]:  # (H, W, C) -> (C, H, W)
            arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(np.clip(arr, 0.0, 1.0).copy())
    else:
        tensor = TF.to_tensor(Image.open(path).convert("RGB"))
    if bands is not None:
        tensor = tensor[bands]
    return tensor.contiguous()


def to_display(img: torch.Tensor) -> np.ndarray:
    """Make a (H, W, 3) array for plotting: first 3 channels, 2-98% contrast stretch."""
    x = img[:3].float()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    lo, hi = torch.quantile(x.flatten(), 0.02), torch.quantile(x.flatten(), 0.98)
    x = ((x - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


# --------------------------------------------------------------------------- #
# Augmentation (works for any number of channels, unlike ColorJitter)
# --------------------------------------------------------------------------- #
class RandomRot90(nn.Module):
    """Satellite tiles have no 'up', so all 90-degree rotations are valid."""

    def forward(self, x):
        return torch.rot90(x, random.randint(0, 3), dims=(-2, -1))


class RandomBrightness(nn.Module):
    def __init__(self, amount: float = 0.1):
        super().__init__()
        self.amount = amount

    def forward(self, x):
        return (x * random.uniform(1 - self.amount, 1 + self.amount)).clamp(0, 1)


def build_transforms(img_size: int, mean, std, train: bool, augment: bool = True):
    ops = [T.Resize((img_size, img_size), antialias=True)]
    if train and augment:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                RandomRot90(), RandomBrightness(0.1)]
    ops.append(T.Normalize(mean, std))
    return T.Compose(ops)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class FileListDataset(Dataset):
    """A list of (path, label) pairs -> (tensor, label)."""

    def __init__(self, samples, bands=None, transform=None):
        self.samples = samples
        self.bands = bands
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_image(path, self.bands)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def scan_folder(root: str | Path) -> tuple[list[tuple[str, str]], list[str]]:
    """ImageFolder-style scan: root/<class_name>/*.jpg|png|tif -> [(path, class_name)]."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    samples = [(str(p), c) for c in classes for p in sorted((root / c).iterdir())
               if p.suffix.lower() in IMAGE_EXTS]
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples, classes


def eurosat_rgb_dir(root: str | Path) -> Path:
    """Download EuroSAT RGB via torchvision (first run only) and return its class folder."""
    from torchvision.datasets import EuroSAT

    EuroSAT(root=str(root), download=True)
    for candidate in Path(root).rglob("Forest"):
        if candidate.is_dir():
            return candidate.parent
    raise RuntimeError("EuroSAT downloaded but the class folders could not be found.")


def apply_class_map(samples, class_map: dict | None):
    """Relabel samples using {new_class: [original classes]}. Unmapped classes are dropped."""
    if class_map is None:
        names = sorted({c for _, c in samples})
        idx = {c: i for i, c in enumerate(names)}
        return [(p, idx[c]) for p, c in samples], names
    names = list(class_map.keys())
    lookup = {orig: i for i, new in enumerate(names) for orig in class_map[new]}
    dropped = sorted({c for _, c in samples if c not in lookup})
    if dropped:
        print(f"[data] Dropping classes not in class_map: {dropped}")
    return [(p, lookup[c]) for p, c in samples if c in lookup], names


def compute_mean_std(samples, bands, n: int = 1000, seed: int = 0):
    """Per-channel mean/std from a random subset of training images."""
    rng = random.Random(seed)
    subset = rng.sample(samples, min(n, len(samples)))
    s, sq, count = None, None, 0
    for path, _ in subset:
        x = load_image(path, bands).double().flatten(1)
        s = x.sum(1) if s is None else s + x.sum(1)
        sq = (x ** 2).sum(1) if sq is None else sq + (x ** 2).sum(1)
        count += x.shape[1]
    mean = s / count
    std = (sq / count - mean ** 2).clamp_min(1e-12).sqrt()
    return mean.tolist(), std.tolist()


@dataclass
class DataBundle:
    loaders: dict
    splits: dict                      # split name -> [(path, label)]
    class_names: list[str]
    in_channels: int
    bands: list[int] | None
    mean: list[float]
    std: list[float]
    class_counts: dict = field(default_factory=dict)


def build_data(cfg: dict) -> DataBundle:
    d = cfg["data"]
    seed = cfg.get("seed", 42)

    # 1) Collect (path, original_class) pairs
    if d["dataset"] == "eurosat_rgb":
        folder = eurosat_rgb_dir(d["root"])
    elif d["dataset"] in {"eurosat_ms", "folder"}:
        if not d.get("folder_dir"):
            raise ValueError("data.folder_dir must be set for eurosat_ms / folder datasets")
        folder = d["folder_dir"]
    else:
        raise ValueError(f"Unknown dataset {d['dataset']!r}")
    raw_samples, _ = scan_folder(folder)

    # 2) Relabel (e.g. 10 EuroSAT classes -> Forest / Urban / Other)
    samples, class_names = apply_class_map(raw_samples, d.get("class_map"))

    # 3) Optional per-class cap for quick debugging runs
    if d.get("limit_per_class"):
        rng = random.Random(seed)
        rng.shuffle(samples)
        kept, seen = [], Counter()
        for s in samples:
            if seen[s[1]] < d["limit_per_class"]:
                kept.append(s)
                seen[s[1]] += 1
        samples = kept

    # 4) Stratified train / val / test split (deterministic given the seed)
    labels = [y for _, y in samples]
    train_val, test = train_test_split(samples, test_size=d["test_split"],
                                       stratify=labels, random_state=seed)
    rel_val = d["val_split"] / (1 - d["test_split"])
    train, val = train_test_split(train_val, test_size=rel_val,
                                  stratify=[y for _, y in train_val], random_state=seed)
    splits = {"train": train, "val": val, "test": test}

    # 5) Channels + normalization stats
    bands = resolve_bands(d.get("bands"))
    in_channels = load_image(train[0][0], bands).shape[0]
    norm = d.get("norm", "auto")
    if norm == "imagenet" or (norm == "auto" and in_channels == 3):
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = compute_mean_std(train, bands, seed=seed)

    # 6) DataLoaders
    loaders = {}
    for name, split in splits.items():
        tf = build_transforms(d["img_size"], mean, std, train=(name == "train"),
                              augment=d.get("augment", True))
        loaders[name] = DataLoader(
            FileListDataset(split, bands, tf),
            batch_size=d["batch_size"],
            shuffle=(name == "train"),
            num_workers=d.get("num_workers", 2),
            pin_memory=torch.cuda.is_available(),
        )

    counts = {name: {class_names[k]: v for k, v in sorted(Counter(y for _, y in split).items())}
              for name, split in splits.items()}
    return DataBundle(loaders, splits, class_names, in_channels, bands, mean, std, counts)


def class_weights(train_samples, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights: n_total / (n_classes * n_class)."""
    counts = Counter(y for _, y in train_samples)
    total = sum(counts.values())
    return torch.tensor([total / (num_classes * counts.get(c, 1)) for c in range(num_classes)],
                        dtype=torch.float32)
