#!/bin/bash
set -e  # stop on first error

MODELS_FULL="./models.yaml"
MODELS_MEDIUM="./models_medium.yaml"

# === Super-Full Run (40 commands × 10 trials, all models) ===
python3 benchmark_openrouter_forLLM.py \
  --models $MODELS_FULL \
  --warmup 2 \
  --trials 10 \
  --commands \
    "Go to the charging station" \
    "Return to the charging station" \
    "Head over to the charger" \
    "Take me back to where the robot charges" \
    "Move to the docking station" \
    "Guide me to the docking station" \
    "Navigate towards the dock" \
    "Go park at the docking station" \
    "Go to the table" \
    "Take me to the table" \
    "Navigate to the table in the room" \
    "Approach the table directly" \
    "I'm thirsty, go to the table for water" \
    "I'm thirsty, go to the bottle of water" \
    "Bring me to where water is kept" \
    "Go to the table for supplies" \
    "Head to the shelf near entry" \
    "Navigate to the entry shelf" \
    "Approach the shelf by the entrance" \
    "Go stand next to the shelf near the entry" \
    "Approach the person standing" \
    "Go stand by the person" \
    "Move closer to the person nearby" \
    "Navigate to the standing person" \
    "Go to the foldable chair" \
    "Find the chair and stop there" \
    "Take me to the foldable chair" \
    "Navigate to the foldable chair for medicine" \
    "I need pain relief, go to the foldable chair" \
    "Go to paracetamol" \
    "Go to Panadol" \
    "Take me to the pain relief" \
    "Go to the chair for paracetamol" \
    "Guide me to where the medicine is" \
    "Navigate to the shelf for supplies" \
    "Move to the docking station to recharge" \
    "Return to the charger after visiting the table" \
    "First go to the person, then stop near the chair" \
    "Go to the table where water bottles are located" \
    "Take me near the foldable chair where medicine is stored" \
  --collect-usage \
  --output-csv results/results_superfull.csv \
  --raw-jsonl results_superfull.jsonl

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
