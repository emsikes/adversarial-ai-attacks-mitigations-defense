import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from torch.utils.data import DataLoader
from pathlib import Path


def get_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Run inference over a dataloader.
    Returns all true labels, predicted labels, and prediction probabilities.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

def print_classification_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray
) -> None:
    """
    Print precision, recall, F1, and support per class.
    """
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=["NORMAL", "PNEUMONIA"]
    )
    print("\nClassification Report:")
    print("-" * 50)
    print(report)

def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path: Path = None
) -> None:
    """
    Plot and optionally save confusion matrix.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)

    classes = ["NORMAL", "PNEUMONIA"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
            
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Labels")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved: {save_path}")
    plt.show()

def plot_roc_curve(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path: Path = None
) -> None:
    """
    Plot ROC curve and print AUC curve.
    """
    fpr, tpr, _ = roc_curve(true_labels, pred_labels)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()

    print(f"AUC: {roc_auc:.4f}")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved: {save_path}")
    plt.show()

def run_full_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path = None
) -> dict:
    """
    Run complete evaluation pipeline.
    Returns metrics dict for programmatic use in attack chapters.
    """
    print("Running evaluation...")
    true_labels, pred_labels, pred_probs = get_predictions(
        model, loader, device
    )

    print_classification_report(true_labels, pred_labels)

    cm_path = save_dir / "confusion_matrix.png" if save_dir else None
    roc_path = save_dir / "roc_curve.png" if save_dir else None

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(true_labels, pred_labels, save_path=cm_path)
    plot_roc_curve(true_labels, pred_probs, save_path=roc_path)

    accuracy = (true_labels == pred_labels).mean()
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy": float(accuracy),
        "auc": float(roc_auc)
    }

    print(f"\nAccuracy: {accuracy:.4f} | AUC {roc_auc:.4f}")
    return metrics