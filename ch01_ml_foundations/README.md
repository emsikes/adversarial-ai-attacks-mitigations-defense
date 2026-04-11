# Chapter 1 — ML Foundations: Building the Target Model

## Model
EfficientNet-B3 fine-tuned for binary chest X-ray classification
- Classes: NORMAL vs PNEUMONIA
- Dataset: Kaggle chest-xray-pneumonia (5,863 images)
- HuggingFace: theinferenceloop/adversarial-ai-target

## Files
- dataset.py      — download, split, class imbalance handling
- transforms.py   — preprocessing pipeline
- model.py        — EfficientNet-B3 with custom classification head
- train.py        — training loop with checkpointing
- evaluate.py     — metrics, confusion matrix, ROC curve
- inference.py    — local inference wrapper (attack surface for ch04-09)