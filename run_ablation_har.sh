#!/bin/bash
mkdir -p logs/new_split/har
for MASK in 0.5 0.75 0.875 0.95; do
    nohup python train/train_mae_har.py \
        --mask_ratio $MASK \
        --mask_strategy random \
        --patch_h 29 --patch_w 25 \
        --seed 42 \
        --encoder_depth 12 \
        --encoder_dim 128 \
        --batch_size 128 \
        --epochs 300 \
        --train_split train_hp \
        --eval_splits val_hp,test_cross_device,test_cross_env,test_cross_user \
        > logs/new_split/har/mae_har_mask${MASK}_hp.log 2>&1 &
    sleep 15
done
