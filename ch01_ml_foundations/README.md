# Chapter 01 — Building the Target Model

## Model
EfficientNet-B3 fine-tuned for binary chest X-ray classification
- Classes: NORMAL (0) vs PNEUMONIA (1)
- Dataset: Kaggle chest-xray-pneumonia (5,863 images)
- HuggingFace: theinferenceloop/adversarial-ai-target

## Files
- dataset.py     — download verification, class imbalance handling, DataLoaders
- transforms.py  — train augmentation and deterministic eval pipeline
- model.py       — EfficientNet-B3 with frozen backbone, custom head, save/load
- train.py       — two-phase training loop with AdamW and CosineAnnealingLR
- evaluate.py    — classification report, confusion matrix, ROC/AUC
- inference.py   — single and batch inference wrappers
- app.py         — FastAPI inference service, /health and /predict endpoints

## Training Strategy
- Phase 1 epochs 1-4: backbone frozen, head only, lr=1e-3
- Phase 2 epoch 5+: unfreeze last 3 backbone blocks, lr=1e-4
- WeightedRandomSampler corrects 3:1 pneumonia/normal class imbalance
- Checkpoint saved to shared/models/best_checkpoint.pt

## Running the Service
```bash
uvicorn app:app --reload --port 8000
```

## Next Steps
- Download Kaggle dataset to shared/datasets/
- Train on Colab Pro A100
- Push checkpoint to HuggingFace