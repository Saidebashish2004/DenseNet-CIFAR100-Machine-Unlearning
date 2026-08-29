# Cybernetic Unlearning Lab: DenseNet-201 CIFAR-100 Selective Forgetting Engine

An advanced, full-stack machine unlearning research platform designed to audit, visualize, and execute selective data forgetting on deep neural networks. Built using a high-performance **FastAPI** PyTorch backend and an obsidian-themed **React** telemetry dashboard.

---

## System Architecture

The platform compares three distinct model states to audit the effectiveness of machine unlearning algorithms:
1. **Base Model (Original)**: Standard model trained on the complete CIFAR-100 dataset.
2. **Unlearned Model (Targeted)**: Fine-tuned model designed to scrub the influence of target classes (Classes 0–9) while preserving baseline performance.
3. **Retrained Model (Gold Standard)**: A baseline model trained exclusively on the remaining dataset (retain set) from scratch.

---

## Project Structure

```text
Machine-Unlearning-CIFAR/
├── backend/                  # FastAPI inference server & training scripts
│   ├── app.py                # FastAPI app serving DenseNet-201 checkpoints
│   └── train_densenet.py     # Training and unlearning pipeline generator
├── frontend/                 # React + Vite telemetry dashboard
│   ├── src/
│   │   ├── App.jsx           # Obsidian-themed dashboard interface
│   │   └── ...
│   └── package.json
├── models/                   # Saved PyTorch checkpoint weights (.pth)
│   ├── base_densenet201.pth
│   ├── unlearned_densenet201.pth
│   └── retrained_densenet201.pth
└── README.md
