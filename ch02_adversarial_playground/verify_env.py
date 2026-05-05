import sys
import subprocess
import importlib
import torch
from pathlib import Path
from typing import Tuple


def check_python_version(requried: Tuple[int, int] = (3, 11)) -> bool:
    """
    Verify Python version meets minimum requirement.
    """
    current = sys.version_info[:2]
    status = current >= requried
    print(f"Python {'.'.join(map(str, current))} "
          f"{'✅' if status else '❌'} (required: {'.'.join(map(str, requried))}+)")
    return status


def check_package(package: str, min_version: str = None) -> bool:
    """
    Verify a package is importable and optionally check version.
    """
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "unknown")
        print(f" {package} {version} ✅")
        return True
    except ImportError:
        print(f" {package} ❌ NOT FOUND")
        return False
    

def check_cuda() -> bool:
    """
    Verify CUDA is available and report GPU details.
    """
    print("\nCUDA / GPU:")
    if torch.cuda.is_available():
        print(f" CUDA AVAILABLE ✅")
        print(f" Device: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f" CUDA version: {torch.version.cuda}")
        return True
    else:
        print(f" CUDA not available ❌")
        print(" Running on CPU - attack chapters will be slower")
        return False
    

def check_target_model(repo_path: Path) -> bool:
    """
    Verify target model checkpoint exists and is loadable.
    """
    print(f"\nTarget Model:")
    checkpoint_path = repo_path / "shared" / "models" / "best_checkpoint.pt"

    if not checkpoint_path.exists():
        print(f" Checkpoint not found ❌")
        print(f" Expected: {checkpoint_path}")
        print(f" Run ch01 training or pull from HuggingFace.")
        return False

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True # load from checkpoint - prevents arbitrary execution during serialization
        )
        val_acc = checkpoint.get("val_accuracy", "unknown")
        epoch = checkpoint.get("epoch", "unknown")
        print(f" Checkpoint found ✅")
        print(f" Epoch: {epoch} | Val accuracy: {val_acc:.4f}")
        return True
    except Exception as e:
        print(f" Checkpoint load failed x: {e}")
        return False
    

def run_full_verification(repo_path: Path) -> bool:
    """
    Run complete environment verification.
    Returns True if all critical checks pass.
    """
    print("\n" + "=" * 50)
    print("ADVERSARIAL AI LAB - ENVIRONMENT VERIFICATION")
    print("=" * 50)

    results = {}

    print("\nPython:")
    results["python"] = check_python_version()

    print("\nCore Packages:")
    core_packages = [
        "torch", "torchvision", "numpy", "pandas",
        "sklearn", "matplotlib", "PIL", "tqdm"
    ]
    results["core"] = all(check_package(p) for p in core_packages)

    print("\nSecurity Packages:")
    security_packages = [
        "modelscan", "bandit", "detect_secrets"
    ]
    results["security"] = all(check_package(p) for p in security_packages)

    print("\nAI/ML Packages:")
    ml_packages = [
        "art", "fastapi", "uvicorn", "huggingface_hub"
    ]
    results["ml"] = all(check_package(p) for p in ml_packages)

    results["cuda"] = check_cuda()
    results["model"] = check_target_model(repo_path)

    print("\n" + "=" * 50)
    critical = ["python", "core", "cuda", "model"]
    all_critical = all(results[k] for k in critical)

    if all_critical:
        print("ENVIRONMENT READY ✅")
    else:
        failed = [k for k in critical if not results[k]]
        print(f"ENVIRONMENT NOT READY ❌ - failed: {failed}")

    print(f"=" * 50)
    return all_critical


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    success = run_full_verification(repo_path)
    sys.exit(0 if success else 1)