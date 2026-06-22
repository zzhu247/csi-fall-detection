#!/bin/bash

# Run Human Activity Recognition task over 7 models
# Using cross-device, cross-environment, and cross-user evaluation

TASK="HumanActivityRecognition"
MODELS=("mlp" "lstm" "transformer" "vit_paper" "patchtst" "timesformer1d" "resnet18")
CHECKPOINT_DIR="checkpoints/har_models"
mkdir -p "$CHECKPOINT_DIR"

echo "=========================================="
echo "HAR Task: Training 7 Models"
echo "=========================================="

# Step 1: Train all 7 models and save checkpoints
for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> Training $model on $TASK..."
    checkpoint="${CHECKPOINT_DIR}/${model}_har.pt"
    
    python eval_benchmark.py \
        --task "$TASK" \
        --model "$model" \
        --mode supervised \
        --epochs 50 \
        --batch_size 32 \
        --save "$checkpoint"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to train $model"
        exit 1
    fi
    echo "✓ $model training complete. Checkpoint saved: $checkpoint"
done

echo ""
echo "=========================================="
echo "HAR Task: OOD Evaluation (Cross-Device, Cross-Env, Cross-User)"
echo "=========================================="

# Step 2: Evaluate each model on OOD splits
for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> Evaluating $model on OOD splits..."
    checkpoint="${CHECKPOINT_DIR}/${model}_har.pt"
    
    if [ ! -f "$checkpoint" ]; then
        echo "❌ Checkpoint not found: $checkpoint"
        exit 1
    fi
    
    python eval_benchmark.py \
        --task "$TASK" \
        --model "$model" \
        --mode ood \
        --checkpoint "$checkpoint" \
        --batch_size 32
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to evaluate $model"
        exit 1
    fi
    echo "✓ $model OOD evaluation complete"
done

echo ""
echo "=========================================="
echo "✓ All evaluations complete!"
echo "Results saved in: results/benchmark_${TASK}_*_ood.json"
echo "=========================================="
