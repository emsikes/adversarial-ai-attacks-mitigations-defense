import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from pathlib import Path
from typing import Tuple


def build_model(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    """
    EfficientNet-B3 with frozen ImageNet backbone and custom classification head.
    Backbone weights from frozen or initial training, only the head trains first.
    """
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model

def unfreeze_backbone(model: nn.Module, num_layers: int = 3) -> nn.Module:
    """
    Selectively unfreeze the last num_layers blocks of the EfficientNet backbone
    for fine-tuning after classification head has stabilized.
    """
    blocks = list(model.features.children())
    layers_to_unfreeze = blocks[-num_layers:]

    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")
    
    return model


def save_model(
    model: nn.Module,
    path: Path,
    epoch: int,
    val_accuracy: float
) -> None:
    """
    Save model checkpoint with metadata.
    """
    checkpoint = {
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "model_state_dict": model.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} | epoch {epoch} | val_acc {val_accuracy:.4f}")

def load_model(path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    """
    Load model checkpoint from disk.
    Returns model and checkpoint metadata.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']} |"
          f"val_acc {checkpoint['val_accuracy']:.4f}")
    
    return model, checkpoint

def get_device() -> torch.device:
    """
    Return best available device.
    CUDA > CPU.  MPS intentionally excluded.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
