# 🤖 Agent Guidelines (AGENTS.md)

Welcome, agent! This document outlines the codebase conventions, design decisions, and running instructions for this project. Please read this before modifying files or running experiments.

---

## 🎯 Project Overview

This repository is designed to fine-tune and benchmark encoder-only models (`jhu-clsp/mmBERT-base`, `Qualcomm-AI-Research/BamiBERT`, and `microsoft/mdeberta-v3-base`) for bilingual English-Vietnamese prompt injection detection.

- **Objective:** Binary classification (`INJECTION` vs `BENIGN`).
- **Input Text:** Multi-turn/complex bilingual prompts containing potential adversarial actions.
- **Key Metric:** F1-score is prioritized due to class-balance requirements, followed by accuracy and inference latency.

---

## 📁 Repository Structure

- `dataset.jsonl`: **Static Golden Dataset.** Contains pre-generated synthetic examples. Do NOT regenerate this file by default to preserve the reproducibility of benchmarks.
- `train_mmbert.py`: Fine-tuning script for `jhu-clsp/mmBERT-base`.
- `train_bamibert.py`: Fine-tuning script for `Qualcomm-AI-Research/BamiBERT`.
- `train_mdeberta.py`: Fine-tuning script for `microsoft/mdeberta-v3-base`.
- `evaluate_models.py`: Loads checkpoints from `./results/`, runs latency benchmarking, and compiles the report.
- `project_plan.md`: Architecture outline of the comparison workflow.
- `comparison_report.md`: Benchmark output showing the performance tables for each model.
- `pyproject.toml`: Dependency configuration managed by `uv`.

---

## 🛠️ Code Conventions & Guardrails

1. **Deterministic Dataset:** The dataset (`dataset.jsonl`) is considered static to ensure reproducible runs. If you need to generate a new dataset, version control the existing one or name the new file differently (e.g., `dataset_v2.jsonl`).
2. **Pure Python Metrics & Split:** Due to potential environment differences and NumPy 1.x vs 2.x conflicts with `scipy`/`scikit-learn`, all scripts use custom pure-Python/NumPy split and metric logic. Do not import `sklearn` modules for splitting or metrics unless environment compatibility is guaranteed.
3. **Smoke Tests:** When modifying training logic or hyperparameters, always run a quick validation check using the `--smoke-test` flag to verify the tokenizer and data formatting on CPU/GPU before starting a full training run.
4. **Device Auto-Detection:** Training scripts must auto-detect CUDA availability (`torch.cuda.is_available()`) and print the selected execution hardware upon starting.
5. **Modern HF Trainer syntax:** Ensure training arguments use `eval_strategy="epoch"` instead of the deprecated `evaluation_strategy="epoch"`.

---

## 🚀 Commands Reference

### Verify training script logic locally (Smoke Test):
```bash
uv run train_mmbert.py --smoke-test --dataset dataset.jsonl
uv run train_bamibert.py --smoke-test --dataset dataset.jsonl
uv run train_mdeberta.py --smoke-test --dataset dataset.jsonl
```

### Run full benchmark comparison on GPU:
```bash
# Fine-tune models
uv run train_mmbert.py --dataset dataset.jsonl --epochs 3 --batch-size 16
uv run train_bamibert.py --dataset dataset.jsonl --epochs 3 --batch-size 16
uv run train_mdeberta.py --dataset dataset.jsonl --epochs 3 --batch-size 16

# Evaluate
uv run evaluate_models.py --dataset dataset.jsonl --report-out comparison_report.md
```
