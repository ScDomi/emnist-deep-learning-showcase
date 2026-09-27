import torch

from emnist_dl.model import build_model
from emnist_dl.train import evaluate


def test_resnet18_forward_shape():
    model = build_model("resnet18", num_classes=36, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 36)


def test_modular_forward_shape():
    labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    model = build_model("resnet18", num_classes=len(labels), pretrained=False, modular=True, idx_to_char=labels)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, len(labels))
