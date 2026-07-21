#!/bin/bash
# Patch-size ablation -- encoder_depth=12 version
#
# encoder_depth=12 DOUBLES the cumulative attention memory vs depth=6 (all layers'
# attention scores are retained simultaneously during backward, no gradient checkpointing).
# Re-running the safety check at depth=12 moves the safe floor up: patch=11 (23.4 GB
# cumulative) is UNSAFE at batch_size=128, encoder_depth=12 -- it was fine at depth=6
# (11.7 GB) but is not here. Patch sizes: 13, 15, 17, 19, 21 -- all comfortably safe
# (cumulative 1.9-11.3 GB) with a uniform batch_size=128 and epochs=300.
#
# check_attention_memory() in train_mae_har.py needed no code changes for this --
# it already gates on the cumulative estimate parameterized by --encoder_depth.

mkdir -p logs

for patch in 13 15 17 19 21; do
    for seed in 42 43; do
        echo "=== Starting patch=${patch}x${patch} seed=${seed} (enc12) ==="
        python train_mae_har.py \
            --epochs 300 \
            --mask_ratio 0.75 \
            --mask_strategy random \
            --encoder_depth 12 \
            --batch_size 128 \
            --patch_h ${patch} --patch_w ${patch} \
            --seed ${seed} \
            > logs/mae_har_enc12_patch${patch}x${patch}_seed${seed}_ep300.log 2>&1
        echo "=== Finished patch=${patch}x${patch} seed=${seed} (enc12) ==="
    done
done
echo "ALL DONE"