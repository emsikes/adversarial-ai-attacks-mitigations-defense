# Adversarial AI Attacks — Hands-On Lab Series

A chapter-by-chapter attack, defense, and MLSecOps implementation series based on
**Adversarial AI Attacks, Mitigations, and Defense Strategies** (Packt). Each chapter is paired with a focused code-along,
lab session, or both — depending on what the material demands.

## Target Model

EfficientNet-B3 fine-tuned on chest X-ray pneumonia detection (NORMAL vs PNEUMONIA).
Hosted on HuggingFace: theinferenceloop/adversarial-ai-target

The target model is the attack surface for chapters 4–9 and 12.

## Environments

| Environment | Purpose |
|---|---|
| Windows 11 RTX 4070 WSL2 | Primary dev, ART attacks, inference |
| Colab Pro A100/H100 | Model training, GAN work, LLM fine-tuning |
| Proxmox security VMs | Isolated exploit execution |
| AWS sandbox | Deployed API target, RAG/LLM chapters |

## Structure

Each chapter folder contains a README describing session type,
target environment, and key concepts covered.

## Framework

- PyTorch 2.6 + CUDA 12.4
- IBM Adversarial Robustness Toolbox (ART)
- TextAttack (NLP chapters)
- TensorFlow Privacy (ch10)
- Python 3.11.9