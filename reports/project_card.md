# Project Card

## Goal
Classify handwritten characters from the official EMNIST Balanced dataset using modern deep-learning backbones and a modular classifier variant.

## Dataset
- Source: `torchvision.datasets.EMNIST`
- Split: `balanced`
- Default classes in this repo: digits `0-9` and uppercase letters `A-Z`
- Data is downloaded at runtime and is intentionally not committed.

## Models
- ResNet18 / ResNet50 adapted to 1-channel EMNIST input
- EfficientNet-B0 adapted to grayscale input
- ConvNeXt-Tiny option
- Modular classifier variant with auxiliary superclass routing

## Evaluation
Preferred metrics:
- Accuracy
- Macro-F1
- Confusion matrix

Historical metrics from the original ML2 submission are retained in `reports/model_results.txt`.
