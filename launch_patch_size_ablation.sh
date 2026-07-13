#!/bin/bash
# Patch-size ablation launch script
# Uses enc6, mask_ratio=0.75, mask_strategy=random -- consistent with your other ablations
# (mask ratio ablation, mask strategy ablation).
#
# NOTE: epochs=100 here, vs epochs=300 for the mask-ratio/mask-strategy ablations.
# Final-epoch numbers from this ablation are NOT directly comparable to those --
# keep conclusions scoped to within-ablation comparisons (patch=3 vs 5 vs 7 vs 11 vs 13,
# all at epoch 100). eval_every defaults to 50, so you'll only get eval_50 and eval_100
# checkpoints (vs 6 checkpoints for the epochs=300 runs) -- the epoch x layer heatmap
# in the notebook will be sparser accordingly.
#
# batch_size is set PER patch size, not a fixed 128, because the naive (non-flash)
# attention in models/vit.py is O(B * heads * N^2) memory. patch=3 and patch=5 need a
# much smaller batch_size to fit under a 32GB GPU -- see check_attention_memory() in
# train_mae_har.py for the exact math. Using --batch_size 128 for patch=3/5 will hard-stop
# at the pre-flight check (by design) instead of wasting GPU time on a mid-run OOM.

declare -A BATCH_SIZE
BATCH_SIZE[3]=7      # num_patches=13026 -- attention memory is the bottleneck, not compute
BATCH_SIZE[5]=60     # num_patches=4700
BATCH_SIZE[7]=128    # num_patches=2448  -- fits at full batch_size, borderline (~11.4GB/layer)
BATCH_SIZE[11]=128   # num_patches=1012  -- comfortably fits
BATCH_SIZE[13]=128   # num_patches=702   -- comfortably fits

mkdir -p logs

for patch in 3 5 7 11 13; do
    bs=${BATCH_SIZE[$patch]}
    for seed in 42 43; do
        echo "=== Starting patch=${patch}x${patch} seed=${seed} (batch_size=${bs}) ==="
        python train_mae_har.py \
            --epochs 100 \
            --mask_ratio 0.75 \
            --mask_strategy random \
            --encoder_depth 6 \
            --batch_size ${bs} \
            --patch_h ${patch} --patch_w ${patch} \
            --seed ${seed} \
            > logs/mae_har_patch${patch}x${patch}_seed${seed}_bs${bs}_ep100.log 2>&1
        echo "=== Finished patch=${patch}x${patch} seed=${seed} ==="
    done
done
echo "ALL DONE"
