# Bilingual Prompt Injection Detection (Vi-En)

This repository contains fine-tuning and evaluation scripts to compare three encoder-only models (`mmBERT`, `BamiBERT`, and `mDeBERTa-v3-base`) for detecting prompt injection attacks in bilingual English-Vietnamese environments.

## 📁 File Structure

- `dataset.jsonl`: The static golden dataset containing 50 high-quality, balanced, bilingual (English-Vietnamese) examples of prompt injections and benign prompts.
- `train_mmbert.py`: Fine-tuning script for `jhu-clsp/mmBERT-base`.
- `train_bamibert.py`: Fine-tuning script for `Qualcomm-AI-Research/BamiBERT`.
- `train_mdeberta.py`: Fine-tuning script for `microsoft/mdeberta-v3-base`.
- `evaluate_models.py`: Evaluation script to load the trained checkpoints, benchmark inference speed, calculate metrics, and compile the final Markdown report.
- `project_plan.md`: Details the architectural design and workflow of the project.
- `AGENTS.md`: Guidelines and conventions for AI agents working in this repository.
- `pyproject.toml`: The `uv` package manager configuration file.

---

## 🚀 Setup & Execution on GPU Server

When you transfer this codebase to your GPU server, you can use **`uv`** to manage dependencies:

### 1. Install Dependencies
```bash
uv sync
```
*(This will install PyTorch, Transformers, Datasets, and critical tokenizer dependencies like `sentencepiece`, `protobuf`, and `tiktoken` automatically from `pyproject.toml`)*

### 2. Log in to Hugging Face
To authenticate for model downloads (required for restricted or gated weights):
```bash
uv run huggingface-cli login
```

### 3. Run Fine-Tuning
You can fine-tune each model individually. The best checkpoints will be saved inside `./results/`:

```bash
# Train mmBERT
uv run train_mmbert.py --dataset dataset.jsonl --epochs 3 --batch-size 16

# Train BamiBERT
uv run train_bamibert.py --dataset dataset.jsonl --epochs 3 --batch-size 16

# Train mDeBERTa-v3-base
uv run train_mdeberta.py --dataset dataset.jsonl --epochs 3 --batch-size 16
```
*(For quick validations on CPU, add the `--smoke-test` flag to any training command)*

### 4. Evaluate Checkpoints
Once the models are trained, run the evaluation script to compare their performance:
```bash
uv run evaluate_models.py --dataset dataset.jsonl --report-out comparison_report.md
```

---

## 📊 Evaluation & Output
The evaluation script compiles the benchmarks into `comparison_report.md` comparing:
- F1-Score
- Accuracy
- Precision
- Recall
- Inference Latency (ms/sample)
