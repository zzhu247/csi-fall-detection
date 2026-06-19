#!/bin/bash
# run_all_paper_baselines.sh
#
# Runs all 6 CSI-Bench paper baseline models across all 7 tasks.
# Skips if checkpoint already exists.
# Serial execution — no GPU contention.
#
# Usage: nohup bash run_all_paper_baselines.sh > logs/paper_baselines.log 2>&1 &

cd ~/csi-project/csi-fall-detection

MODELS=(mlp lstm transformer vit_paper patchtst timesformer1d resnet18)

# Single-task datasets
SINGLE_TASKS=(FallDetection MotionSourceRecognition Localization)

# Multi-task datasets
MULTI_TASKS=(HumanActivityRecognition HumanIdentification ProximityRecognition)

ALL_TASKS=("${SINGLE_TASKS[@]}" "${MULTI_TASKS[@]}")

for model in "${MODELS[@]}"; do
    for task in "${ALL_TASKS[@]}"; do
        ckpt="checkpoints/paper_${model}_${task}.pt"
        log="logs/paper_${model}_${task}.log"

        if [ -f "$ckpt" ]; then
            echo "$(date '+%H:%M:%S') SKIP  $model / $task — checkpoint exists"
            continue
        fi

        echo "$(date '+%H:%M:%S') START $model / $task"
        python -u eval_benchmark.py \
            --task  "$task" \
            --model "$model" \
            --mode  supervised \
            --epochs 100 \
            --lr    1e-3 \
            --batch_size 128 \
            --save  "$ckpt" \
            > "$log" 2>&1

        if [ $? -eq 0 ]; then
            echo "$(date '+%H:%M:%S') DONE  $model / $task"
        else
            echo "$(date '+%H:%M:%S') FAIL  $model / $task — check $log"
        fi
    done
done

echo ""
echo "$(date) All paper baselines complete."
echo "Results in results/benchmark_*_supervised.json"
