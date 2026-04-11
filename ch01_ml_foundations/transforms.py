from torchvision import transforms


def get_train_transforms() -> transforms.Compose:
    """
    Training transforms with augmentation.
    EfficientNet-B3 expects 300x300 input.
    """
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
 
def get_eval_transforms() -> transforms.Compose:
    """
    Evaluation transforms, no augmentation.
    Deterministic pipeline for val, test, and attack chapters.
    """
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])