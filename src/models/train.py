"""Training utilities for character CNNs (MobileNetV2).

By default this script downloads EMNIST ByClass, canonicalizes the labels to
the repository's 36-class alphabet/digit set, and trains a MobileNetV2 head.
It also supports a local ImageFolder fallback for custom datasets.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from src.models.cnn import build_mobilenet_v2


CHAR_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _to_rgb(image):
    return image.convert("RGB")


def _correct_emnist(image):
    # EMNIST images are stored rotated and mirrored.
    image = TF.rotate(image, 90)
    image = TF.hflip(image)
    return image


def _build_base_transform(train: bool = True):
    augment = []
    if train:
        augment.extend([
            transforms.RandomApply([transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                shear=6,
            )], p=0.7),
            transforms.RandomAutocontrast(p=0.2),
        ])

    return transforms.Compose([
        transforms.Lambda(_correct_emnist),
        transforms.Lambda(_to_rgb),
        *augment,
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CanonicalEMNIST(torch.utils.data.Dataset):
    def __init__(self, base_dataset, allowed_classes: Sequence[str]):
        self.base_dataset = base_dataset
        self.allowed_classes = list(allowed_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.allowed_classes)}

        self.samples: list[tuple[int, int]] = []
        for idx, label_idx in enumerate(base_dataset.targets):
            raw_label = base_dataset.classes[int(label_idx)]
            canonical = raw_label.upper() if raw_label.isalpha() else raw_label
            if canonical in self.class_to_idx:
                self.samples.append((idx, self.class_to_idx[canonical]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        base_index, target = self.samples[index]
        image, _ = self.base_dataset[base_index]
        return image, target


class DatasetSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base_dataset[self.indices[index]]


def _build_emnist_datasets(data_dir: str, train: bool, download: bool = True):
    base = datasets.EMNIST(
        root=data_dir,
        split="byclass",
        train=train,
        download=download,
        transform=_build_base_transform(train=train),
    )
    return CanonicalEMNIST(base, CHAR_CLASSES)


def _build_imagefolder_dataset(data_dir: str):
    return datasets.ImageFolder(data_dir, transform=_build_base_transform(train=True))


def _make_subset(dataset, limit: int | None):
    if limit is None or limit >= len(dataset):
        return dataset
    return DatasetSubset(dataset, range(limit))


def _make_weighted_sampler(dataset) -> WeightedRandomSampler | None:
    if not hasattr(dataset, "samples"):
        return None
    labels = []
    for _, target in dataset.samples:
        labels.append(int(target))
    counts = torch.bincount(torch.tensor(labels), minlength=max(labels) + 1)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train(
    data_dir: str,
    out_path: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    dataset: str = "emnist",
    val_split: float = 0.1,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    unfreeze_last_blocks: int = 3,
    label_smoothing: float = 0.1,
    patience: int = 3,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if dataset == "emnist":
        train_ds = _build_emnist_datasets(data_dir, train=True, download=True)
        val_ds = _build_emnist_datasets(data_dir, train=False, download=True)
        classes: Sequence[str] = CHAR_CLASSES
    elif dataset == "imagefolder":
        full_ds = _build_imagefolder_dataset(data_dir)
        classes = full_ds.classes
        val_size = max(1, int(len(full_ds) * val_split))
        train_size = max(1, len(full_ds) - val_size)
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")

    train_ds = _make_subset(train_ds, max_train_samples)
    val_ds = _make_subset(val_ds, max_val_samples)

    sampler = _make_weighted_sampler(train_ds) if dataset == "emnist" else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_mobilenet_v2(
        num_classes=len(classes),
        pretrained=True,
        freeze_features=True,
        unfreeze_last_blocks=unfreeze_last_blocks,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_acc = -math.inf
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += xb.size(0)

        epoch_loss /= total
        acc = correct / total if total > 0 else 0.0
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{epochs} — loss={epoch_loss:.4f} acc={acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s"
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {epoch} epochs (best val_acc={best_val_acc:.4f})")
                break

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(classes),
            "dataset": dataset,
            "num_classes": len(classes),
            "best_val_acc": None if best_val_acc == -math.inf else float(best_val_acc),
            "unfreeze_last_blocks": unfreeze_last_blocks,
            "label_smoothing": label_smoothing,
        },
        str(out_path),
    )
    print(f"Saved checkpoint to {out_path}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += float(loss.item()) * xb.size(0)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += xb.size(0)
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="EMNIST root or local ImageFolder root")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--dataset", choices=["emnist", "imagefolder"], default="emnist")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--unfreeze-last-blocks", type=int, default=3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_dir=args.data_dir,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        dataset=args.dataset,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
    )
