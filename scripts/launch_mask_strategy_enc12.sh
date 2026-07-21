#!/bin/bash
# Mask Strategy Ablation -- enc12 rerun
# random/freq/mixed/time/2d x seed 42/43, mask_ratio=0.75, patch_h=29/patch_w=25, encoder_depth=12
#
# patch_h=29, patch_w=25 evenly divide 232x500, so num_patches=160 regardless of
# encoder_depth -- batch_size=128 is safe (no O(N^2) attention memory concern here).

mkdir -p logs

for strategy in random freq mixed time 2d; do
    for seed in 42 43; do
        echo "=== Starting strategy=${strategy} seed=${seed} (enc12) ==="
        python train_mae_har.py \
            --epochs 300 \
            --mask_ratio 0.75 \
            --mask_strategy ${strategy} \
            --encoder_depth 12 \
            --batch_size 128 \
            --patch_h 29 --patch_w 25 \
            --seed ${seed} \
            > logs/mae_har_enc12_strategy${strategy}_seed${seed}_ep300.log 2>&1
        echo "=== Finished strategy=${strategy} seed=${seed} (enc12) ==="
    done
done
echo "ALL DONE"
