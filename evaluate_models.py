#!/usr/bin/env python3
"""
evaluate_models.py
Evaluates and benchmarks trained model checkpoints located in `./results/`.
Computes Precision, Recall, F1, Accuracy, and Latency, then writes `comparison_report.md`.
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# Define models and their source hubs (for tokenizer loading if local tokenizer files aren't saved)
MODEL_PATHS = {
    "mmBERT-base": ("./results/mmBERT-base", "jhu-clsp/mmBERT-base"),
    "BamiBERT": ("./results/BamiBERT", "Qualcomm-AI-Research/BamiBERT"),
    "mDeBERTa-v3-base": ("./results/mDeBERTa-v3-base", "microsoft/mdeberta-v3-base")
}

def load_data(file_path: str):
    if not os.path.exists(file_path):
        print(f"Dataset file {file_path} not found.")
        sys.exit(1)
        
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def stratified_split(texts, labels, test_size=0.2, random_seed=42):
    random.seed(random_seed)
    label_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)
        
    train_indices = []
    val_indices = []
    
    for label, indices in label_indices.items():
        shuffled = list(indices)
        random.shuffle(shuffled)
        split_point = int(len(shuffled) * (1 - test_size))
        train_indices.extend(shuffled[:split_point])
        val_indices.extend(shuffled[split_point:])
        
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    
    return (
        [texts[i] for i in train_indices],
        [texts[i] for i in val_indices],
        [labels[i] for i in train_indices],
        [labels[i] for i in val_indices]
    )

def compute_metrics(predictions, labels):
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    total = tp + tn + fp + fn
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def measure_latency(model, val_dataset, device: str) -> float:
    """Measures the average inference latency per sample in milliseconds."""
    model.eval()
    latencies = []
    
    samples_to_test = min(len(val_dataset), 50)
    
    with torch.no_grad():
        for i in range(samples_to_test):
            item = val_dataset[i]
            inputs = {
                "input_ids": torch.tensor([item["input_ids"]]).to(device),
                "attention_mask": torch.tensor([item["attention_mask"]]).to(device)
            }
            if "token_type_ids" in item:
                inputs["token_type_ids"] = torch.tensor([item["token_type_ids"]]).to(device)
                
            start_time = time.time()
            model(**inputs)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            
    return float(np.mean(latencies))

def write_report(results, output_path: str):
    """Writes a markdown report summarizing the model comparison."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Model Comparison Report: Bilingual Prompt Injection Detection\n\n")
        f.write("This report presents the fine-tuning results and benchmark comparison across the target encoder-only models.\n\n")
        
        f.write("| Model Name | F1-Score | Accuracy | Precision | Recall | Latency (ms/sample) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for res in results:
            f.write(
                f"| **{res['model_name']}** | {res['f1']:.4f} | {res['accuracy']:.4f} | {res['precision']:.4f} | "
                f"{res['recall']:.4f} | {res['inference_latency_ms']:.2f} ms |\n"
            )
            
        f.write("\n\n## 💡 Key Findings & Recommendations\n\n")
        best_model = max(results, key=lambda x: x["f1"])
        fastest_model = min(results, key=lambda x: x["inference_latency_ms"])
        
        f.write(f"- **Top Performer (F1-score):** `{best_model['model_name']}` (F1: `{best_model['f1']:.4f}`)\n")
        f.write(f"- **Fastest Model (Latency):** `{fastest_model['model_name']}` (Avg: `{fastest_model['inference_latency_ms']:.2f} ms`)\n\n")
        
        f.write("### Architectures Analyzed:\n")
        f.write("1. **mmBERT-base** (`jhu-clsp/mmBERT-base`): Multilingual Multimodal BERT useful for understanding mixed language contexts.\n")
        f.write("2. **BamiBERT** (`Qualcomm-AI-Research/BamiBERT`): Explicitly pre-trained on English-Vietnamese code-switched text, offering great domain alignment.\n")
        f.write("3. **mDeBERTa-v3-base** (`microsoft/mdeberta-v3-base`): Employs a disentangled attention mechanism and ELECTRA-style pre-training, usually yielding the highest classification quality.\n")
        
    print(f"\nSaved comparison report to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned checkpoints")
    parser.add_argument("--dataset", type=str, default="dataset.jsonl", help="Path to the JSONL dataset")
    parser.add_argument("--report-out", type=str, default="comparison_report.md", help="Output comparison report filename")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset validation split
    data = load_data(args.dataset)
    texts = [item["text"] for item in data]
    labels = [1 if item["label"] == "INJECTION" else 0 for item in data]
    
    _, val_texts, _, val_labels = stratified_split(texts, labels, test_size=0.2, random_seed=42)
    print(f"Evaluation dataset size: {len(val_texts)} samples")
    
    all_results = []
    
    for model_name, (local_path, hub_path) in MODEL_PATHS.items():
        # Find best checkpoint folder inside the local path directory
        if not os.path.exists(local_path):
            print(f"⚠️ Skipped {model_name}: No training folder found at '{local_path}'. Run its training script first.")
            continue
            
        checkpoint_dirs = [os.path.join(local_path, d) for d in os.listdir(local_path) if d.startswith("checkpoint-")]
        if not checkpoint_dirs:
            # Maybe the model was saved directly in the directory
            model_dir = local_path
        else:
            # Load the latest checkpoint
            model_dir = sorted(checkpoint_dirs, key=os.path.getmtime)[-1]
            
        print(f"\nEvaluating Model: {model_name} from checkpoint: {model_dir}")
        
        try:
            # Load tokenizer
            use_fast = "deberta" not in hub_path.lower()
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(hub_path, use_fast=use_fast)
                
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.to(device)
            model.eval()
            
            # Tokenize
            encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
            
            val_dataset = Dataset.from_dict({
                **{k: v.tolist() for k, v in encodings.items()},
                "label": val_labels
            })
            
            # Predict
            predictions = []
            with torch.no_grad():
                for i in range(len(val_texts)):
                    inputs = {k: torch.tensor([v]).to(device) for k, v in encodings.items()}
                    logits = model(**inputs).logits
                    pred = int(torch.argmax(logits, dim=-1).cpu().item())
                    predictions.append(pred)
                    
            metrics = compute_metrics(np.array(predictions), np.array(val_labels))
            latency = measure_latency(model, val_dataset, device)
            
            res = {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "inference_latency_ms": latency
            }
            all_results.append(res)
            print(f"  Accuracy: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | Latency: {res['inference_latency_ms']:.2f} ms/sample")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
    if all_results:
        write_report(all_results, args.report_out)
    else:
        print("No models evaluated successfully.")

if __name__ == "__main__":
    main()
