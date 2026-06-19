#!/bin/bash
cd ~/csi-project/csi-fall-detection

tasks=(
    "MotionSourceRecognition resnet18 checkpoints/resnet18_MotionSource.pt logs/resnet18_MotionSource.log"
    "Localization resnet18 checkpoints/resnet18_Localization.pt logs/resnet18_Localization.log"
    "ProximityRecognition resnet18 checkpoints/resnet18_Proximity.pt logs/resnet18_Proximity.log"
    "HumanIdentification resnet18 checkpoints/resnet18_HumanID.pt logs/resnet18_HumanID.log"
    "HumanActivityRecognition vit checkpoints/vit_HAR.pt logs/vit_HAR.log"
)

for entry in "${tasks[@]}"; do
    task=$(echo $entry | awk '{print $1}')
    model=$(echo $entry | awk '{print $2}')
    ckpt=$(echo $entry | awk '{print $3}')
    log=$(echo $entry | awk '{print $4}')

    echo "$(date) Starting: $task $model"
    python -u eval_benchmark.py --task $task --model $model --mode supervised \
        --epochs 50 --save $ckpt > $log 2>&1
    echo "$(date) Done: $task $model"
done

echo "All baselines complete!"
