from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy

import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_score = float("-inf")
        self.best_state = None
        self.bad_epochs = 0

    def step(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[Metrics, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = Metrics(total_loss / max(len(y_true), 1), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0))
    return metrics, y_true, y_pred


def train(model: nn.Module, train_loader, val_loader, *, epochs: int, lr: float, weight_decay: float, device: torch.device, patience: int = 5, amp: bool = True):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    stopper = EarlyStopping(patience=patience)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
        scheduler.step()
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": running_loss / max(seen, 1), "train_accuracy": correct / max(seen, 1), "val_loss": val_metrics.loss, "val_accuracy": val_metrics.accuracy, "val_macro_f1": val_metrics.macro_f1}
        history.append(row)
        print(row)
        if stopper.step(val_metrics.accuracy, model):
            print(f"Early stopping after epoch {epoch}; best validation accuracy={stopper.best_score:.4f}")
            break
    stopper.restore(model)
    return history


def save_confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation="vertical")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
