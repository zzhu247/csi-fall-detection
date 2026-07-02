
#!/bin/bash

cd ~/csi-project/csi-fall-detection

    for seed in 42 43; do

        echo "=== Starting mask_ratio=${ratio} seed=${seed} ==="

        python train_mae_har.py \

            --epochs 300 \

            --mask_ratio 0.95 \

            --mask_strategy random \

            --encoder_depth 6 \

            --batch_size 128 \

            --seed ${seed} \

            > logs/mae_har_mask${ratio}_seed${seed}_ep300.log 2>&1

        echo "=== Finished mask_ratio=${ratio} seed=${seed} ==="

    done

done

echo "ALL DONE"

