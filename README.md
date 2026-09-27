# EMNIST Deep Learning Showcase

Vorzeigerepo für ein Deep-Learning-Projekt auf dem offiziellen EMNIST-Datensatz. Das Projekt ist aus dem ursprünglichen ML2-Modularbeitscode bereinigt, reproduzierbarer strukturiert und GitHub-ready gemacht.

## Warum das Repo vorzeigewürdig ist

- Offizieller Datensatz: `torchvision.datasets.EMNIST(split="balanced")`
- Transfer Learning mit torchvision-Backbones: ResNet18, ResNet50, EfficientNet-B0, ConvNeXt-Tiny
- Optionaler modularer Ansatz: Zeichenklassifikation plus grobe Superklasse (Ziffern / Großbuchstaben / Kleinbuchstaben)
- Saubere Train/Validation/Test-Trennung auf Basis offizieller EMNIST-Splits
- Balancing + Augmentation für robuste Klassifikation
- Early Stopping, Label Smoothing, AdamW, LR-Scheduling, Mixed Precision auf CUDA
- Reproduzierbare CLI statt Notebook-only Projekt
- Reports und vorhandene Ergebnisartefakte liegen unter `reports/`

## Ergebnisstand aus der ursprünglichen ML2-Arbeit

| Modell | Accuracy | R² |
|---|---:|---:|
| EfficientNet-B0 | 0.8115 | 0.5015 |
| ResNet50 | 0.8359 | 0.5578 |
| ResNet18 | 0.8341 | 0.5550 |
| Modular classifier | 0.8289 | 0.5497 |

Hinweis: Für Klassifikation ist Accuracy / Macro-F1 aussagekräftiger als R². R² bleibt hier nur als historisches Vergleichsartefakt aus der ursprünglichen Abgabe dokumentiert.

## Repository-Struktur

```text
emnist-deep-learning-showcase/
  src/emnist_dl/          # Paket-Code: Dataset, Modelle, Training, CLI
  tests/                  # Smoke-Tests ohne Datensatzdownload
  reports/                # Ergebniszusammenfassung und Grafiken
  configs/                # Platz für reproduzierbare Experimente
  notebooks/              # Optional: explorative Notebooks
  pyproject.toml          # Installierbares Python-Projekt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Training

Kleiner Smoke-Run:

```bash
emnist-train --smoke --no-pretrained --batch-size 32 --output-dir outputs/smoke
```

Voller ResNet18-Lauf:

```bash
emnist-train --model resnet18 --epochs 30 --batch-size 128 --output-dir outputs/resnet18
```

Modularer Lauf:

```bash
emnist-train --model resnet18 --modular --epochs 30 --batch-size 128 --output-dir outputs/modular-resnet18
```

## Outputs

Jeder Lauf schreibt:

- `metrics.json` mit History und Testmetriken
- `confusion_matrix.png`
- Modellgewichte als `.pt` im Output-Ordner

## Methodische Notizen

- EMNIST wird erst beim Training heruntergeladen; große Daten- und Modelldateien bleiben aus Git draußen.
- `.gitignore` schließt `data/`, `outputs/`, `.venv/`, Caches und lokale Gewichte aus.
- Die alten `.pth`-Gewichte wurden bewusst nicht übernommen, weil sie GitHub unnötig aufblähen würden.
- Das Projekt ist jetzt eher Portfolio/Recruiter-tauglich: klare Story, reproduzierbare CLI, keine wilden Notebook-/Cache-Reste.
