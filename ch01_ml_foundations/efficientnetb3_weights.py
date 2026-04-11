from torchvision.models import EfficientNet_B3_Weights

weights = EfficientNet_B3_Weights.IMAGENET1K_V1
print(weights.transforms())