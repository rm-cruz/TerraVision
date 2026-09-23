"""Fast checks that run without downloading anything: `pytest -q`."""
import numpy as np
import pytest
import torch
from PIL import Image

from satclass.data import apply_class_map, build_data, load_image, resolve_bands
from satclass.models import adapt_first_conv, build_model
from satclass.scene import classify_scene, transition_matrix


@pytest.mark.parametrize("name", ["simplecnn", "resnet18", "resnet50", "efficientnet_b0"])
@pytest.mark.parametrize("channels", [3, 4])
def test_models_forward(name, channels):
    model = build_model(name, num_classes=3, in_channels=channels, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, channels, 64, 64))
    assert out.shape == (2, 3)


def test_first_conv_keeps_rgb_filters():
    conv = torch.nn.Conv2d(3, 8, 3)
    new = adapt_first_conv(conv, 4)
    assert new.weight.shape == (8, 4, 3, 3)
    assert torch.allclose(new.weight[:, :3], conv.weight * 3 / 4)


def test_freeze_backbone():
    m = build_model("resnet18", 3, pretrained=False, freeze_backbone=True)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("fc.") for n in trainable)


def test_band_names():
    assert resolve_bands(["B04", "B03", "B02", "B08"]) == [3, 2, 1, 7]


def test_class_map_drops_unlisted():
    samples = [("a", "Forest"), ("b", "Residential"), ("c", "River"), ("d", "Cloud")]
    out, names = apply_class_map(samples, {"Forest": ["Forest"], "Urban": ["Residential"],
                                           "Other": ["River"]})
    assert names == ["Forest", "Urban", "Other"]
    assert [y for _, y in out] == [0, 1, 2]


def test_multispectral_tif_loading(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    arr = (np.random.rand(64, 64, 13) * 10000).astype(np.uint16)  # EuroSAT-MS layout (H, W, C)
    tifffile.imwrite(tmp_path / "x.tif", arr)
    x = load_image(tmp_path / "x.tif", resolve_bands(["B04", "B03", "B02", "B08"]))
    assert x.shape == (4, 64, 64) and 0 <= x.min() and x.max() <= 1


def test_build_data_from_folder(tmp_path):
    for cls in ["Forest", "Residential", "River"]:
        (tmp_path / cls).mkdir()
        for i in range(10):
            Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(tmp_path / cls / f"{i}.png")
    cfg = {"seed": 0, "data": {
        "dataset": "folder", "folder_dir": str(tmp_path), "bands": None, "norm": "auto",
        "img_size": 32, "val_split": 0.2, "test_split": 0.2, "batch_size": 4, "num_workers": 0,
        "class_map": {"Forest": ["Forest"], "Urban": ["Residential"], "Other": ["River"]}}}
    data = build_data(cfg)
    assert data.in_channels == 3 and data.class_names == ["Forest", "Urban", "Other"]
    assert sum(len(v) for v in data.splits.values()) == 30
    x, y = next(iter(data.loaders["train"]))
    assert x.shape == (4, 3, 32, 32)


def test_scene_classification():
    model = build_model("simplecnn", 3, pretrained=False).eval()
    ckpt = {"in_channels": 3, "mean": [0.5] * 3, "std": [0.25] * 3, "img_size": 64}
    cmap, probs = classify_scene(model, torch.rand(3, 200, 330), ckpt, "cpu", tile=64)
    assert cmap.shape == (3, 5) and probs.shape == (3, 5, 3)
    t = transition_matrix(cmap, cmap, 3)
    assert t.sum() == 15 and np.trace(t) == 15
