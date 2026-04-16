<div align="center">

# Adversarial AI Attacks — Hands-On Lab Series

**A chapter-by-chapter attack, defense, and MLSecOps implementation series**

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![ART](https://img.shields.io/badge/ART-1.18.2-0052CC?style=for-the-badge&logo=ibm&logoColor=white)](https://adversarial-robustness-toolbox.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-theinferenceloop-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/theinferenceloop)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![TextAttack](https://img.shields.io/badge/TextAttack-NLP_Attacks-purple?style=flat-square)](https://textattack.readthedocs.io)
[![TF Privacy](https://img.shields.io/badge/TensorFlow_Privacy-Differential_Privacy-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/tensorflow/privacy)
[![Book](https://img.shields.io/badge/Book-Packt_2024-blueviolet?style=flat-square)](https://www.packtpub.com)

[Target Model](#target-model) • [Structure](#structure) • [Environments](#environments) • [Framework](#framework) • [Chapters](#chapters)

---

</div>

## Overview

A chapter-by-chapter hands-on implementation series based on
**Adversarial AI Attacks, Mitigations, and Defense Strategies** (Packt, 2024)
by John Sotiropoulos. Each chapter is paired with a focused code-along,
lab session, or both — depending on what the material demands.

Built function by function. Every component is understood before moving forward.

---

## Target Model

EfficientNet-B3 fine-tuned on chest X-ray pneumonia detection.

| Property | Value |
|---|---|
| Architecture | EfficientNet-B3 (ImageNet pretrained) |
| Task | Binary classification — NORMAL vs PNEUMONIA |
| Dataset | Kaggle chest-xray-pneumonia (5,863 images) |
| Input size | 300 × 300 RGB |
| HuggingFace | [theinferenceloop/adversarial-ai-target](https://huggingface.co/theinferenceloop) |
| Attack surface | Chapters 4–9 and 12 |

---

## Environments

| Environment | Purpose |
|---|---|
| Windows 11 RTX 4070 WSL2 | Primary dev, ART attacks, inference |
| Colab Pro A100 / H100 | Model training, GAN work, LLM fine-tuning |
| Proxmox security VMs | Isolated exploit execution |
| AWS sandbox | Deployed API target, RAG/LLM chapters |

---

## Structure

```
adversarial-ai-attacks-mitigations/
├── shared/
│   ├── models/       # Checkpoints (gitignored — hosted on HuggingFace)
│   ├── datasets/     # Training data (gitignored)
│   └── utils/        # Shared utilities
├── ch01_ml_foundations/
├── ch02_adversarial_playground/
├── ch03_security_fundamentals/
├── ch04_poisoning_attacks/
├── ch05_trojan_horses/
├── ch06_supply_chain/
├── ch07_evasion_attacks/
├── ch08_model_stealing/
├── ch09_data_stealing/
├── ch10_privacy_preserving/
├── ch11_generative_ai/
├── ch12_deepfakes/
├── ch13_llm_foundations/
├── ch14_prompt_attacks/
├── ch15_llm_poisoning/
├── ch16_advanced_genai/
├── ch17_secure_by_design/
├── ch18_mlsecops/
└── ch19_maturing_security/
```

---

## Framework

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.6.0 | Model development and training |
| torchvision | 0.21.0 | EfficientNet-B3 pretrained weights |
| IBM ART | 1.18.2 | Adversarial attack and defense toolkit |
| TextAttack | 0.3.10 | NLP evasion attacks (ch07, ch14) |
| TensorFlow Privacy | 0.9.0 | Differential privacy (ch10) |
| MLflow | latest | Model registry and MLSecOps (ch18) |
| FastAPI | 0.115.12 | Inference service and attack surface |
| LangChain | latest | RAG pipeline (ch13–ch16) |

---

## Chapters

| # | Chapter | Session Type |
|---|---|---|
| 01 | ML Foundations: Building the Target Model | Code-along |
| 02 | Building Our Adversarial Playground | Lab |
| 03 | Security and Adversarial AI | Code-along + Lab |
| 04 | Poisoning Attacks | Code-along + Lab |
| 05 | Trojan Horses and Model Reprogramming | Lab |
| 06 | Supply Chain Attacks | Code-along + Lab |
| 07 | Evasion Attacks against Deployed AI | Code-along + Lab |
| 08 | Privacy Attacks — Stealing Models | Lab |
| 09 | Privacy Attacks — Stealing Data | Code-along + Lab |
| 10 | Privacy-Preserving AI | Code-along |
| 11 | Generative AI — A New Frontier | Code-along + Lab |
| 12 | Weaponizing GANs for Deepfakes | Code-along + Lab |
| 13 | LLM Foundations for Adversarial AI | Code-along |
| 14 | Adversarial Attacks with Prompts | Code-along + Lab |
| 15 | Poisoning Attacks and LLMs | Code-along + Lab |
| 16 | Advanced Generative AI Scenarios | Lab |
| 17 | Secure by Design and Trustworthy AI | Code-along |
| 18 | AI Security with MLSecOps | Code-along |
| 19 | Maturing AI Security | Code-along |

---

## Getting Started

```bash
git clone https://github.com/emsikes/adversarial-ai-attacks-mitigations
cd adversarial-ai-attacks-mitigations
pyenv local 3.11.9
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install torch==2.6.0 torchvision==0.21.0
```

Full dependency installation is done chapter by chapter as needed.
See each chapter's README for specific requirements.

---

## Disclaimer

All attack techniques in this repository are implemented strictly for
educational and research purposes in controlled lab environments.
Do not use any techniques here against systems you do not own or have
explicit written authorization to test.

---

<div align="center">
Part of the <a href="https://theinferenceloop.substack.com">The Inference Loop</a> research series.
</div>
