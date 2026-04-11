import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Tuple
import time


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.optimizer,
        criterion: nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    """
    Run one training epoch.
    Returns average loss and accuracy for the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on val or test loader.
    Returns average loss and accuracy.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.margmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 1e3,
    checkpoint_dir: Path = Path("../shared/models")
) -> nn.Module:
    """
    Full training loop with checkpointing.
    Phase 1: head only (frozen backbone)
    Phase 2: unfreeze last 3 backbone blocks at epoch 5
    """
    from model import save_model, unfreeze_backbone

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs, + 1):
        if epoch == 5:
            print("\nPhase 2: unfreezing backbone layers...")
            model = unfreeze_backbone(model, num_layers=3)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - epoch)

        start = time.time()
        train_loss, train_acc = train.epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:<2}/{epochs} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model,
                checkpoint_dir / "best_checkpoint.pt",
                epoch,
                val_acc
            )

    return model
