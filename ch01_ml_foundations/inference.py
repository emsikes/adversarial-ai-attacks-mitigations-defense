import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Union
import numpy as np


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load a single image from disk and convert to RGB.
    Handles grayscale X-rays that ship as a single-channel.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileExistsError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image: Image.image) -> torch.Tensor:
    """
    Apply eval transforms and add batch dimension.
    Returns tensor shapre of [1, 3, 300, 300] ready for model input.
    Unsqueeze adds the batch dimensions expected by the model.
    """
    from transforms import get_eval_transforms
    transform = get_eval_transforms()
    tensor = transform(image)
    return tensor.unsqueeze(0)

def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Run inference on a preprocessed image tensor.
    Returns class label, confidence, and full probability distribution.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class = probs.max(dim=1)

    class_names = ["NORMAL", "PNEUMONIA"]
    return {
        "class": class_names[pred_class.item()],
        "class_idx": pred_class.item(),
        "confidence": confidence.item(),
        "prbabilities": {
            "NORMAL": probs[0, 0].item(),
            "PNEUMONIA": probs[0, 1].item()
        }
    }

def predict_batch(
    model: torch.nn.Module,
    image_tensors: torch.Tensor,
    device: torch.device
) -> list:
    """
    Run inference on a batch of preprocessed image tensors.
    Returns a list of prediction dicts, one per image.
    """
    model.eval()
    image_tensors = image_tensors.to(device)

    with torch.no_grad():
        outputs = model(image_tensors)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_classes = probs.max(dim=1)

    class_names = ["NORMAL", "PNEUMONIA"]
    results = []
    for i in range(len(pred_classes)):
        results.append({
            "class": class_names[pred_classes[i].item()],
            "class_idx": pred_classes[i].item(),
            "confidence": confidence[i].item(),
            "probabilities": {
                "NORMAL": probs[i, 0].item(),
                "PNEUMONIA": probs[i, 1].item()
            }
        })

    return results