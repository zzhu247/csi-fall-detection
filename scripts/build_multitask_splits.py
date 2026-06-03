# scripts/build_multitask_splits.py
# Build train/val/test splits for all tasks

import os, json, pandas as pd
import config

TASKS = {
    "FallDetection": {
        "splits": {
            "train":     "train_id.json",
            "val":       "val_id.json",
            "test_easy": "test_easy.json",
            "test_hard": "test_hard.json",
        },
        "id_col": "id",
    },
    "MotionSourceRecognition": {
        "splits": {
            "train":     "train_id.json",
            "val":       "val_id.json",
            "test_easy": "test_easy.json",
            "test_hard": "test_hard.json",
        },
        "id_col": "id",
    },
    "BreathingDetection": {
        "splits": {
            "train":     "train_id.json",
            "val":       "val_id.json",
            "test_easy": "test_easy_id.json",
            "test_hard": "test_hard_id.json",
        },
        "id_col": "id",
    },
    "Localization": {
        "splits": {
            "train":     "train_id.json",
            "val":       "val_id.json",
            "test_easy": "test_easy_id.json",
            "test_hard": "test_hard_id.json",
        },
        "id_col": "id",
    },
}

save_dir = os.path.expanduser("~/data/splits")
os.makedirs(save_dir, exist_ok=True)

for task, info in TASKS.items():
    print(f"\n── {task} ──")

    meta_path = os.path.join(config.DATA_ROOT, task, "metadata", "sample_metadata.csv")
    if not os.path.exists(meta_path):
        print(f"  No metadata found, skipping.")
        continue

    meta = pd.read_csv(meta_path)

    # Filter missing files
    def file_exists(row):
        p = os.path.join(config.DATA_ROOT, task, row["file_path"].lstrip("./"))
        return os.path.exists(p)

    meta = meta[meta.apply(file_exists, axis=1)].reset_index(drop=True)
    meta["task"] = task

    splits_dir = os.path.join(config.DATA_ROOT, task, "splits")

    for split_name, split_file in info["splits"].items():
        split_path = os.path.join(splits_dir, split_file)
        if not os.path.exists(split_path):
            print(f"  {split_name}: file not found ({split_file})")
            continue

        with open(split_path) as f:
            ids = set(json.load(f))

        split_df = meta[meta[info["id_col"]].isin(ids)].reset_index(drop=True)

        # Save
        save_path = os.path.join(save_dir, f"{task}_{split_name}.csv")
        split_df.to_csv(save_path, index=False)

        print(f"  {split_name:10s}: {len(split_df):5d} samples  "
              f"labels={split_df['label'].value_counts().to_dict()}")