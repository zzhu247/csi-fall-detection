# scripts/build_combined_dataset.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import config

# ── Task definitions ──────────────────────────────────────
# format: (meta_path, h5_base_path, file_path_prefix_to_strip)
TASKS = {
    # Direct tasks (file_path is relative to task root)
    "FallDetection": {
        "meta":    "FallDetection/metadata/sample_metadata.csv",
        "h5_base": "FallDetection",
        "path_type": "relative",
    },
    "MotionSourceRecognition": {
        "meta":    "MotionSourceRecognition/metadata/sample_metadata.csv",
        "h5_base": "MotionSourceRecognition",
        "path_type": "relative",
    },
    "BreathingDetection": {
        "meta":    "BreathingDetection/metadata/sample_metadata.csv",
        "h5_base": "BreathingDetection",
        "path_type": "relative",
        "known_issue": True,   # data reshaping issue per README
    },
    "Localization": {
        "meta":    "Localization/metadata/sample_metadata.csv",
        "h5_base": "Localization",
        "path_type": "relative",
        "known_issue": True,
    },
    # Multitask tasks (file_path uses ../../sub_Human_h5/...)
    "HumanActivityRecognition": {
        "meta":    "Multitask/HumanActivityRecognition/metadata/sample_metadata.csv",
        "h5_base": "Multitask",
        "path_type": "multitask",
    },
    "HumanIdentification": {
        "meta":    "Multitask/HumanIdentification/metadata/sample_metadata.csv",
        "h5_base": "Multitask",
        "path_type": "multitask",
    },
    "ProximityRecognition": {
        "meta":    "Multitask/ProximityRecognition/metadata/sample_metadata.csv",
        "h5_base": "Multitask",
        "path_type": "multitask",
    },
}

def resolve_h5_path(data_root, task_name, file_path, path_type, h5_base):
    """Resolve h5 file path based on task type."""
    if path_type == "relative":
        return os.path.join(data_root, h5_base, file_path.lstrip("./"))
    elif path_type == "multitask":
        # file_path = ../../sub_Human_h5/user_U01/...
        # resolve from Multitask/TaskName/ → goes up 2 levels to data_root/Multitask/
        rel = file_path.replace("../../", "")
        return os.path.join(data_root, "Multitask", rel)

# ── Build combined dataset ────────────────────────────────
save_dir = os.path.expanduser("~/data")
os.makedirs(save_dir, exist_ok=True)

all_dfs = []
summary = []

for task, info in TASKS.items():
    meta_path = os.path.join(config.DATA_ROOT, info["meta"])
    if not os.path.exists(meta_path):
        print(f"SKIP {task}: no metadata")
        continue

    df = pd.read_csv(meta_path)
    df["task"]      = task
    df["path_type"] = info["path_type"]
    df["h5_base"]   = info["h5_base"]
    df["known_issue"] = info.get("known_issue", False)

    # Resolve full h5 path and filter missing files
    df["h5_path"] = df["file_path"].apply(
        lambda p: resolve_h5_path(
            config.DATA_ROOT, task, p, info["path_type"], info["h5_base"]
        )
    )
    before = len(df)
    df = df[df["h5_path"].apply(os.path.exists)].reset_index(drop=True)
    after = len(df)

    summary.append({
        "task":        task,
        "total":       before,
        "found":       after,
        "missing":     before - after,
        "known_issue": info.get("known_issue", False),
    })
    print(f"{task}: {before} → {after} found  (missing: {before-after})")

    if after > 0:
        all_dfs.append(df)

# Summary
print("\n── Summary ──────────────────────────────────")
for s in summary:
    flag = " ⚠ known issue" if s["known_issue"] else ""
    print(f"  {s['task']:<30} {s['found']:>6} samples{flag}")

# Combine and shuffle
combined = pd.concat(all_dfs, ignore_index=True).sample(
    frac=1, random_state=42
).reset_index(drop=True)

total = len(combined)
print(f"\nTotal combined: {total}")

# ── 70/20/10 split ────────────────────────────────────────
n_train = int(total * 0.70)
n_val   = int(total * 0.20)
n_test  = total - n_train - n_val

train_df = combined.iloc[:n_train].reset_index(drop=True)
val_df   = combined.iloc[n_train:n_train+n_val].reset_index(drop=True)
test_df  = combined.iloc[n_train+n_val:].reset_index(drop=True)

print(f"\n70/20/10 Split:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")

print(f"\nTask distribution in train:")
print(train_df["task"].value_counts().to_string())

# Save
train_df.to_csv(f"{save_dir}/combined_train.csv", index=False)
val_df.to_csv(  f"{save_dir}/combined_val.csv",   index=False)
test_df.to_csv( f"{save_dir}/combined_test.csv",  index=False)
combined.to_csv(f"{save_dir}/combined_all.csv",   index=False)

print(f"\nSaved to {save_dir}/")
print("  combined_train.csv")
print("  combined_val.csv")
print("  combined_test.csv")
print("  combined_all.csv")
