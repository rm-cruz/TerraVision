# Satellite Imagery Classification for Deforestation & Urban Sprawl

**COMP 569 – Artificial Intelligence, CSUN · Fall 2026 · Group 10**
Ava Harken · Roxanne Cruz · Quang Tran

We classify Sentinel-2 satellite image tiles into land-cover classes (**Forest / Urban / Other**) with CNNs, compare a from-scratch baseline against ImageNet-pretrained models, test whether adding the near-infrared band helps, and then apply the best model to before/after scenes of the same area to measure deforestation and urban expansion.

## Setup

```bash
git clone <repo-url> && cd satellite-landcover
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # ~10 s, no downloads needed
```

No GPU? Everything runs on CPU for small debug runs. For full training, use Google Colab (Runtime → GPU): clone the repo in a cell and run the same commands with `!python ...`.

## Data

| Dataset | What | How to get it |
|---|---|---|
| **EuroSAT RGB** (default) | 27,000 labeled 64×64 Sentinel-2 tiles, 10 classes, RGB JPEG | Downloads automatically on first run into `data/` |
| **EuroSAT MS** | Same tiles, all 13 spectral bands (GeoTIFF) | Download `EuroSAT_MS.zip` (~2 GB) from the EuroSAT GitHub/Zenodo page, unzip to `data/EuroSAT_MS/` |
| **Our own scenes** | Before/after images of a region for change detection | Export from [Copernicus Browser](https://browser.dataspace.copernicus.eu) (Sentinel-2 L2A, same bounding box, low cloud cover, two dates). Save to `data/scenes/` |

The 10 EuroSAT classes are grouped into our 3 target classes in `configs/default.yaml` (`data.class_map`):

- **Forest**: Forest
- **Urban**: Highway, Industrial, Residential
- **Other**: AnnualCrop, HerbaceousVegetation, Pasture, PermanentCrop, River, SeaLake

This grouping is imbalanced (Other ≫ Forest), so training uses class-weighted loss and picks the best epoch by **macro-F1**, not accuracy. Set `data.class_map=null` to train on all 10 original classes.

Data, checkpoints, and run outputs are git-ignored. Share large files through the team Drive folder, not GitHub.

## Quick start

```bash
# 1. Download EuroSAT + make figures (class distribution, sample tiles) -> reports/figures/
python explore_data.py

# 2. 2-minute sanity check on CPU
python train.py --set data.limit_per_class=100 train.epochs=2 model.name=simplecnn model.pretrained=false data.num_workers=0

# 3. Real training run (GPU recommended)
python train.py

# 4. Detailed evaluation: per-image predictions CSV, confusion matrix, worst mistakes
python evaluate.py --run runs/<run-folder>

# 5. Classify individual tiles
python predict.py --checkpoint runs/<run-folder>/best.pt --input path/to/images/

# 6. Change detection on two dates of the same area
python detect_change.py --checkpoint runs/<run-folder>/best.pt \
    --before data/scenes/region_2018.tif --after data/scenes/region_2024.tif \
    --out reports/region_change --pixel-size-m 10
```

Any config value can be overridden from the command line with `--set key=value` (dotted keys, YAML values), so experiments never require editing the default config.

## What each run saves (`runs/<timestamp>_<model>_<dataset>/`)

| File | Contents |
|---|---|
| `config.yaml` | exact config used (reproducibility) |
| `splits.json` | exact train/val/test file lists |
| `best.pt` | best checkpoint (weights + class names + normalization, self-contained) |
| `history.csv`, `curves.png` | loss / accuracy / macro-F1 per epoch |
| `test_metrics.json`, `test_report.txt` | precision, recall, F1 per class on the held-out test set |
| `confusion_matrix.png` | counts + row-normalized percentages |
| `misclassified.png` | the model's most confident mistakes (for error analysis) |

## Repository layout

```
satellite-landcover/
├── configs/
│   ├── default.yaml            # EuroSAT RGB, 3 classes, ResNet-50
│   └── eurosat_ms_rgbn.yaml    # RGB + NIR multispectral variant
├── satclass/                   # library code (import from here)
│   ├── data.py                 # loading RGB/GeoTIFF, band selection, class mapping, splits, augmentation
│   ├── models.py               # SimpleCNN, ResNet-18/50, EfficientNet-B0, N-channel input, checkpoints
│   ├── engine.py               # train / eval loop (mixed precision on GPU)
│   ├── metrics.py              # metrics + report figures
│   ├── scene.py                # tile a large scene and classify it; transition matrix
│   └── utils.py                # config overrides, seeding, device selection
├── tests/test_smoke.py         # fast tests on synthetic data
├── train.py  evaluate.py  predict.py  detect_change.py  explore_data.py
├── data/  runs/  reports/figures/   (git-ignored contents)
└── requirements.txt
```

## Planned experiments

Each row is one or more `train.py` runs; record results in the shared results sheet.

| # | Question | Command sketch |
|---|---|---|
| E1 | Baseline: how well does a small CNN from scratch do? | `--set model.name=simplecnn model.pretrained=false` |
| E2 | Does ImageNet transfer learning help? | `model.name=resnet50` vs E1 |
| E3 | Fine-tune everything vs. linear probe | `model.freeze_backbone=true` vs `false` |
| E4 | Architecture comparison | `resnet18`, `resnet50`, `efficientnet_b0` |
| E5 | Does input resolution matter for pretrained nets? | `data.img_size=64 / 128 / 224` |
| E6 | Does near-infrared help separate vegetation? | `--config configs/eurosat_ms_rgbn.yaml` vs RGB |
| E7 | Effect of augmentation / class weighting | `data.augment=false`, `train.class_weighted_loss=false` |
| E8 | Data efficiency | `data.limit_per_class=100 / 500 / 1000 / null` |
| E9 | Real-world change detection | `detect_change.py` on 2+ regions (e.g. Amazon, Inland Empire) |

For statistical significance, repeat the key comparisons with 3 seeds (`seed=1 2 3`) and report mean ± std.

## Known limitations to discuss in the report

- **Domain shift**: EuroSAT JPEGs use a fixed brightness scaling; scenes exported from other tools may look different, which can hurt predictions. Compare scene colors to EuroSAT samples, and consider training on your own labeled tiles.
- **Tile-level labels**: each 640 m × 640 m tile gets one label, so mixed tiles (forest edge, suburbs) are ambiguous.
- **Clouds and seasons** cause false "changes"; pick same-season, cloud-free dates.

## Team workflow

- `main` stays runnable. Work on branches (`data/...`, `model/...`, `eval/...`) and open a pull request; one teammate reviews before merge.
- Run `pytest -q` before pushing.
- Never commit data, `.pt` files, or `runs/`.

## References

- P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
- M. Tan, Q. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML, 2019.
- European Space Agency, Copernicus Sentinel-2 mission data.
