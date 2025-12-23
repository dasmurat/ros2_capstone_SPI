#!/bin/bash
set -e  # stop on first error



MODELS_FULL="./models.yaml"
MODELS_MEDIUM="./models_medium.yaml"

# === Full Run (20 commands × 10 trials, all models) ===
python3 benchmark_openrouter_forLLM.py \
  --models $MODELS_FULL \
  --warmup 2 \
  --trials 10 \
  --commands \
    "Go to the charging station" \
    "Move to the docking station" \
    "Navigate to the table" \
    "Head to the shelf near entry" \
    "Approach the person standing" \
    "Go to the foldable chair" \
    "I'm thirsty, go to the table for water" \
    "I'm thirsty, go to the bottle of water" \
    "I need pain relief, go to the foldable chair" \
    "Go to paracetamol" \
    "Go to Panadol" \
    "Navigate to the table for supplies" \
    "Go to the foldable chair for medicine" \
    "Guide me to the docking station" \
    "Return to the charging station" \
    "Approach the entry shelf" \
    "Go stand by the person" \
    "Find the chair and stop there" \
    "Take me to the table" \
    "Navigate to the shelf for supplies" \
  --collect-usage \
  --output-csv results/results_full.csv \
  --raw-jsonl results_full.jsonl

# === Medium Run (10 commands × 5 trials, medium models) ===
python3 benchmark_openrouter_forLLM.py \
  --models $MODELS_MEDIUM \
  --warmup 1 \
  --trials 5 \
  --commands \
    "Go to the charging station" \
    "Move to the docking station" \
    "Navigate to the table" \
    "Head to the shelf near entry" \
    "Approach the person standing" \
    "Go to the foldable chair" \
    "I'm thirsty, go to the table for water" \
    "Go to paracetamol" \
    "Go to Panadol" \
    "Go to the foldable chair for medicine" \
  --collect-usage \
  --output-csv results/results_medium.csv \
  --raw-jsonl results_medium.jsonl

# === Short Run (5 commands × 3 trials, medium models) ===
python3 benchmark_openrouter_forLLM.py \
  --models $MODELS_MEDIUM \
  --warmup 1 \
  --trials 3 \
  --commands \
    "Go to the charging station" \
    "Move to the docking station" \
    "Go to the table" \
    "Go to the foldable chair" \
    "Head to the shelf near entry" \
  --collect-usage \
  --output-csv results/results_short.csv \
  --raw-jsonl results_short.jsonl
