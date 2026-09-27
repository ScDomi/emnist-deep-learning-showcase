from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights


class SuperclassCNN(nn.Module):
    """Small auxiliary CNN for digit/uppercase/lowercase routing."""

    def __init__(self, in_channels: int = 1, num_superclasses: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(32 * 7 * 7, 64), nn.ReLU(inplace=True), nn.Linear(64, num_superclasses))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ModularClassifier(nn.Module):
    """Combines a character classifier with a coarse superclass classifier.

    During normal training/evaluation the forward pass returns class logits so it
    works directly with CrossEntropyLoss. `forward_with_aux` exposes both heads.
    """

    def __init__(self, character_model: nn.Module, superclass_model: nn.Module, superclass_map: torch.Tensor):
        super().__init__()
        self.character_model = character_model
        self.superclass_model = superclass_model
        self.register_buffer("superclass_map", superclass_map.long())

    def forward_with_aux(self, x: torch.Tensor):
        char_logits = self.character_model(x)
        super_logits = self.superclass_model(x)
        char_log_probs = torch.log_softmax(char_logits, dim=1)
        super_log_probs = torch.log_softmax(super_logits, dim=1)[:, self.superclass_map]
        return char_log_probs + super_log_probs, char_logits, super_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined_log_probs, _, _ = self.forward_with_aux(x)
        return combined_log_probs


def _replace_first_conv(model: nn.Module, out_channels: int, kernel_size, stride, padding):
    return nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def build_backbone(name: str, num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.conv1 = _replace_first_conv(model, 64, 7, 2, 3)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.BatchNorm1d(in_features), nn.Dropout(dropout), nn.Linear(in_features, 256), nn.ReLU(inplace=True), nn.Dropout(dropout / 2), nn.Linear(256, num_classes))
    elif name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        model.conv1 = _replace_first_conv(model, 64, 7, 2, 3)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.BatchNorm1d(in_features), nn.Dropout(dropout), nn.Linear(in_features, 512), nn.ReLU(inplace=True), nn.Dropout(dropout / 2), nn.Linear(512, num_classes))
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        model.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model '{name}'. Use resnet18, resnet50, efficientnet_b0, or convnext_tiny.")
    return model


def build_superclass_map(idx_to_char: list[str]) -> torch.Tensor:
    mapping = []
    for char in idx_to_char:
        if char.isdigit():
            mapping.append(0)
        elif char.isupper():
            mapping.append(1)
        else:
            mapping.append(2)
    return torch.tensor(mapping)


def build_model(name: str, num_classes: int, dropout: float = 0.2, pretrained: bool = True, modular: bool = False, idx_to_char: list[str] | None = None) -> nn.Module:
    base = build_backbone(name, num_classes, dropout, pretrained)
    if not modular:
        return base
    if idx_to_char is None:
        raise ValueError("idx_to_char is required for modular=True")
    return ModularClassifier(base, SuperclassCNN(), build_superclass_map(idx_to_char))
