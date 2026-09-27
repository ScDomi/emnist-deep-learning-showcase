# EMNIST Deep Learning Showcase

[![CI](https://github.com/ScDomi/emnist-deep-learning-showcase/actions/workflows/ci.yml/badge.svg)](https://github.com/ScDomi/emnist-deep-learning-showcase/actions/workflows/ci.yml)

A portfolio-ready PyTorch project for handwritten character recognition on the official EMNIST Balanced dataset.

The repo is a cleaned-up, reproducible version of my ML2 deep-learning project: raw notebook/coursework structure out, clear package + CLI + reports in.

## Highlights

- Official dataset via `torchvision.datasets.EMNIST(split="balanced")`
- Transfer learning with ResNet18, ResNet50, EfficientNet-B0 and ConvNeXt-Tiny
- Optional modular classifier: character head + coarse digit/uppercase/lowercase routing head
- Balanced train/validation/test datasets built from official EMNIST splits
- Augmentation, AdamW, label smoothing, LR scheduling, early stopping and CUDA mixed precision
- CLI-based training instead of notebook-only execution
- Saved report artifacts: class-distribution plots, confusion matrices and model comparison notes

## Results from the original ML2 run

| Model | Accuracy | R² (historical) |
|---|---:|---:|
| EfficientNet-B0 | 0.8115 | 0.5015 |
| ResNet50 | 0.8359 | 0.5578 |
| ResNet18 | 0.8341 | 0.5550 |
| Modular classifier | 0.8289 | 0.5497 |

Note: this is a classification project, so Accuracy and Macro-F1 are the primary metrics. R² is kept only because it was part of the original university submission comparison.

## Example artifacts

The repo includes generated artifacts from the original experiment under `reports/figures/`:

- Class distribution before/after balancing
- ResNet18 confusion matrix
- Modular classifier confusion matrix

## Repository structure

```text
emnist-deep-learning-showcase/
  src/emnist_dl/          # Dataset, model, training and CLI code
  tests/                  # Lightweight model smoke tests
  reports/                # Result notes and figures
  configs/                # Reproducible experiment configs
  .github/workflows/      # CI checks
```

## Setup

Use Python 3.10-3.13. PyTorch wheels may not be available for newer Python versions immediately after release.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Quick smoke run

```bash
emnist-train --smoke --no-pretrained --batch-size 32 --output-dir outputs/smoke
```

## Full training examples

```bash
emnist-train --model resnet18 --epochs 30 --batch-size 128 --output-dir outputs/resnet18
```

```bash
emnist-train --model resnet18 --modular --epochs 30 --batch-size 128 --output-dir outputs/modular-resnet18
```

Each run writes:

- `metrics.json`
- `confusion_matrix.png`
- model weights as `.pt`

## Design choices

- EMNIST data is downloaded at runtime and excluded from Git.
- Large `.pth` checkpoints from the original run are not committed to keep the repo lightweight.
- The cleaned version favors reproducibility and readable project structure over dumping every experiment artifact.
- The modular model is kept because it shows experimentation beyond a plain pretrained backbone.

## Local verification

```bash
python -m py_compile src/emnist_dl/*.py tests/*.py
pytest -q
```

## License

MIT
