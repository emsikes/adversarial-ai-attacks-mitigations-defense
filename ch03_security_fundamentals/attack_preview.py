import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "ch01_ml_foundations"))

from model import build_model, load_model, get_device
from transforms import get_eval_transforms


def load_target_model(device: torch.device):
    """
    Load the ch01 target model from shared/models/.
    Returns model in eval mode ready for attack.
    """
    checkpoint_path = Path(__file__).parent.parent / "shared" / "models" / "best_checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Target model not found: {checkpoint_path}\n"
            f"Run ch01 training or pull from HuggingFace."
        )
    
    model, checkpoint = load_model(checkpoint_path, device)
    print(f"Target model loaded - epoch {checkpoint['epoch']} | "
          f"val_acc {checkpoint['val_accuracy']:.4f}")
    
    return model


def wrap_for_art(model: torch.nn.Module, device: torch.device) -> PyTorchClassifier:
    """
    Wrap the PyTorch model in ART's PyTorchClassifier.
    Required for all ART attack and defense methods.
    """
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 300, 300),
        nb_classes=2,
        device_type="gpu" if device.type == "cuda" else "cpu"
    )
    print("Model wrapped for ART.")
    return classifier


def load_sample_image(image_path: Path) -> tuple:
    """
    Load a single image and prepare for attack.
    Returns original PIL image, preprocessed tensor, and numpy array for ART.
    """
    transform = get_eval_transforms()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    numpy_array = tensor.numpy()

    return image, tensor, numpy_array


def run_fgsm_attack(
        classifier: PyTorchClassifier,
        image_array: np.ndarray,
        epsilon: float = 0.01
) -> np.ndarray:
    """
    Run Fast Gradient Sign Method attack against the classifier.
    Epsilon controls perturbation magnitude, higher = more visible distoration.
    Returns adversarial image as numpy array.
    """
    attack = FastGradientMethod(
        estimator=classifier,
        eps=epsilon,
        eps_step=epsilon / 2,
        targeted=False,
        num_random_init=0
    )

    adversarial_array = attack.generate(x=image_array)
    return adversarial_array


def compare_predictions(
        classifier: PyTorchClassifier,
        original_array: np.ndarray,
        adversarial_array: np.ndarray
) -> dict:
    """
    Compare model predictions on original vs adversarial image.
    Returns dict with before / after prediction and confidence scores.
    """
    class_names = ["NORMAL", "PNEUMONIA"]

    orig_logits = classifier.predict(original_array)
    adv_logits = classifier.predict(adversarial_array)

    # Apply softmax to convert logits to probabilities
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    orig_probs = softmax(orig_logits)
    adv_probs = softmax(adv_logits)

    orig_class = np.argmax(orig_probs, axis=1)[0]
    adv_class = np.argmax(adv_probs, axis=1)[0]

    orig_confidence = orig_probs[0][orig_class]
    adv_confidence = adv_probs[0][adv_class]

    result = {
        "original": {
            "class": class_names[orig_class],
            "confidence": float(orig_confidence),
            "probabilities": {
                "NORMAL": float(orig_probs[0][0]),
                "PNEUMONIA": float(orig_probs[0][1])
            }
        },
        "adversarial": {
            "class": class_names[adv_class],
            "confidence": float(adv_confidence),
            "probabilities": {
                "NORMAL": float(adv_probs[0][0]),
                "PNEUMONIA": float(adv_probs[0][1])
            }
        },
        "attack_succeeded": orig_class != adv_class
    }

    print(f"\nOriginal   : {result['original']['class']} "
          f"({result['original']['confidence']:.4f})")
    print(f"Adversarial: {result['adversarial']['class']} "
          f"({result['adversarial']['confidence']:.4f})")
    print(f"Attack succeeded: {result['attack_succeeded']}")

    return result


def visualize_attack(
        original_image: Image.Image,
        adversarial_array: np.ndarray,
        result: dict,
        epsilon: float,
        save_path: Path = None
) -> None:
    """
    Visualize original vs adversarial image side by side.
    Shows perturbation magnitude in a third panel.
    """
    original_array = np.array(original_image.resize((300, 300))) / 255.0
    adversarial_display = np.transpose(adversarial_array[0], (1, 2, 0))
    adversarial_display = np.clip(adversarial_display, 0, 1)

    perturbation = np.abs(original_array - adversarial_display)
    perturbation_amplified = np.clip(perturbation * 10, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_array)
    axes[0].set_title(
        f"Original\n{result['original']['class']} "
        f"({result['original']['confidence']:.4f})",
        fontsize=12
    )
    axes[0].axis("off")

    axes[1].imshow(adversarial_display)
    axes[1].set_title(
        f"Adversarial (eps{epsilon})\n{result['adversarial']['class']} "
        f"({result['adversarial']['confidence']:.4f})",
        fontsize=12
    )
    axes[1].axis("off")

    plt.suptitle(
        f"FGSM Attack - Attack Succeeded: {result['attack_succeeded']}",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    import sys
    from datasets import load_dataset

    device = get_device()
    model = load_target_model(device)
    classifier = wrap_for_art(model, device)

    data_dir = Path(__file__).parent.parent / "shared" / "datasets"
    pneumonia_dir = data_dir / "chest_xray" / "test" / "PNEUMONIA"

    if pneumonia_dir.exists():
        sample_images = list(pneumonia_dir.glob("*.jpeg"))[:1]
        if sample_images:
            image_path = sample_images[0]
            print(f"\nAttacking: {image_path.name}")
            original_image, tensor, image_array = load_sample_image(image_path)
        else:
            print("No test images found in local dataset.")
            sys.exit(1)
    else:
        print("Local dataset not found — streaming sample from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset(
            "hf-vision/chest-xray-pneumonia",
            split="test",
            streaming=True
        )
        sample = next(iter(ds.filter(lambda x: x["label"] == 1)))
        original_image = sample["image"].convert("RGB")
        print("Streaming sample loaded — PNEUMONIA class.")

        transform = get_eval_transforms()
        tensor = transform(original_image).unsqueeze(0)
        image_array = tensor.numpy()

    output_dir = Path(__file__).parent / "attack_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epsilon in [0.01, 0.05, 0.1]:
        print(f"\n--- FGSM eps={epsilon} ---")
        adversarial_array = run_fgsm_attack(classifier, image_array, epsilon)
        result = compare_predictions(classifier, image_array, adversarial_array)
        visualize_attack(
            original_image,
            adversarial_array,
            result,
            epsilon,
            save_path=output_dir / f"fgsm_eps{epsilon}.png"
        )

    print(f"\nAttack outputs saved to: {output_dir}")