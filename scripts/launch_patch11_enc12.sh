#!/bin/bash
# Patch-size ablation -- encoder_depth=12, patch=11x11 only
# batch_size=96 (not the standard 128) -- patch=11 at encoder_depth=12 is UNSAFE at
# batch_size=128 (23.4 GB cumulative attention memory vs the 20GB budget). 96 gives
# 17.6 GB cumulative, safe with margin. See launch_patch_size_ablation_enc12.sh for
# the full 5-patch-size sweep this belongs to.

mkdir -p logs

for seed in 42 43; do
    echo "=== Starting patch=11x11 seed=${seed} (enc12, batch_size=96) ==="
    python train_mae_har.py \
        --epochs 300 \
        --mask_ratio 0.75 \
        --mask_strategy random \
        --encoder_depth 12 \
        --batch_size 96 \
        --patch_h 11 --patch_w 11 \
        --seed ${seed} \
        > logs/mae_har_enc12_patch11x11_seed${seed}_bs96_ep300.log 2>&1
    echo "=== Finished patch=11x11 seed=${seed} (enc12) ==="
done
echo "ALL DONE"
