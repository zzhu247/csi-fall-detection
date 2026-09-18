"""
Confirms which dataset (and which CSI-Bench task subset) a given result JSON was
produced from. The result JSON's 'args' dict does NOT store the dataset path directly
(META_PATH/SPLITS_DIR are hardcoded module-level constants in train_mae_har.py /
train_mae_hid.py, not CLI arguments), so this checks the two things that DO reveal it:
  1. The 'exp' name prefix ('mae_har_...' vs 'mae_hid_...') identifies the task.
  2. config.DATA_ROOT identifies the CSI-Bench dataset root actually used.
"""
import json
import sys

sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config

RESULT_JSON = "results/mae_har/mae_har_ep150_mask0.75_strategyrandom_ph29pw25_seed42_enc12_dim128_bs128.json"

with open(RESULT_JSON) as f:
    result = json.load(f)

exp_name = result['exp']
task = "HumanActivityRecognition" if exp_name.startswith("mae_har") else \
       "HumanIdentification" if exp_name.startswith("mae_hid") else "UNKNOWN"

print(f"exp name:    {exp_name}")
print(f"task subset: CSI-Bench Multitask / {task}")
print(f"DATA_ROOT:   {config.DATA_ROOT}")
print(f"META_PATH:   {config.DATA_ROOT}/Multitask/{task}/metadata/sample_metadata.csv")
print(f"SPLITS_DIR:  {config.DATA_ROOT}/Multitask/{task}/splits")

exclude_devices = result['args'].get('exclude_devices', '')
if exclude_devices:
    print(f"NOTE: this run excluded device(s): {exclude_devices}")
