#!/bin/bash
cd ~/csi-project/csi-fall-detection

tasks=(
    "FallDetection resnet18 checkpoints/resnet18_FallDetection.pt logs/resnet18_FallDetection.log"
    "FallDetection vit checkpoints/vit_FallDetection.pt logs/vit_FallDetection.log"
    "MotionSourceRecognition vit checkpoints/vit_MotionSource.pt logs/vit_MotionSource.log"
    "Localization vit checkpoints/vit_Localization.pt logs/vit_Localization.log"
    "ProximityRecognition vit checkpoints/vit_Proximity.pt logs/vit_Proximity.log"
    "HumanIdentification vit checkpoints/vit_HumanID.pt logs/vit_HumanID.log"
    "HumanActivityRecognition resnet18 checkpoints/resnet18_HAR.pt logs/resnet18_HAR.log"
    "HumanActivityRecognition vit checkpoints/vit_HAR.pt logs/vit_HAR.log"
)

for entry in "${tasks[@]}"; do
    task=$(echo $entry | awk '{print $1}')
    model=$(echo $entry | awk '{print $2}')
    ckpt=$(echo $entry | awk '{print $3}')
    log=$(echo $entry | awk '{print $4}')

    # Skip if checkpoint already exists
    if [ -f "$ckpt" ]; then
        echo "$(date) Skipping $task $model — checkpoint exists"
        continue
    fi

    echo "$(date) Starting: $task $model"
    python -u eval_benchmark.py --task $task --model $model --mode supervised \
        --epochs 50 --save $ckpt > $log 2>&1
    echo "$(date) Done: $task $model"
done

echo "All done!"
