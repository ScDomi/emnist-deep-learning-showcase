from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import DatasetConfig, get_datasets
from .model import build_model
from .train import evaluate, save_confusion_matrix, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train an EMNIST deep-learning classifier")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.7378831750922145e-4)
    parser.add_argument("--dropout", type=float, default=0.24044341963473287)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--allowed-chars", default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--modular", action="store_true", help="Enable auxiliary digit/uppercase/lowercase routing head")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny run for CI/local verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = dict(train_samples_per_class=24, val_samples_per_class=4, test_samples_per_class=4) if args.smoke else {}
    config = DatasetConfig(data_root=args.data_root, allowed_chars=args.allowed_chars, **samples)
    data = get_datasets(config)

    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(data["val"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(data["test"], batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, len(data["idx_to_char"]), args.dropout, not args.no_pretrained, args.modular, data["idx_to_char"]).to(device)
    history = train(model, train_loader, val_loader, epochs=1 if args.smoke else args.epochs, lr=args.lr, weight_decay=args.weight_decay, device=device)

    criterion = torch.nn.CrossEntropyLoss()
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    save_confusion_matrix(y_true, y_pred, data["idx_to_char"], output_dir / "confusion_matrix.png")
    torch.save(model.state_dict(), output_dir / f"{args.model}_emnist.pt")

    summary = {"device": str(device), "model": args.model, "modular": args.modular, "classes": data["idx_to_char"], "history": history, "test": test_metrics.__dict__}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
