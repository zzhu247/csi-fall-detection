#!/bin/bash
cd ~/csi-project/csi-fall-detection

tasks=(
    "FallDetection"
    "MotionSourceRecognition"
    "Localization"
    "HumanActivityRecognition"
    "HumanIdentification"
    "ProximityRecognition"
)

for task in "${tasks[@]}"; do
    echo "$(date) Starting ViT: $task"
    python -u eval_benchmark.py --task $task --model vit --mode supervised \
        --epochs 100 --lr 3e-4 \
        --save checkpoints/vit_${task}.pt \
        > logs/vit_${task}_fixed.log 2>&1
    echo "$(date) Done: $task"
done
