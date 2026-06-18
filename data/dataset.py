# data/dataset.py

import h5py
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import config


# ── Normalization utilities ────────────────────────────────

def normalize_subcarriers(csi, target_n=232):
    """
    Normalize CSI to have exactly target_n subcarriers.
    csi: numpy array [n_subcarriers, time]
    """
    n = csi.shape[0]
    if n == target_n:
        return csi
    elif n < target_n:
        pad_total = target_n - n
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(csi, ((pad_left, pad_right), (0, 0)), mode="constant")
    else:
        start = (n - target_n) // 2
        return csi[start:start + target_n, :]


def normalize_time(csi, target_t=500):
    """
    Normalize CSI to have exactly target_t time steps via crop or zero-pad.
    csi: numpy array [n_subcarriers, time]
    """
    T = csi.shape[1]
    if T == target_t:
        return csi
    elif T > target_t:
        start = (T - target_t) // 2
        return csi[:, start:start + target_t]
    else:
        pad = np.zeros((csi.shape[0], target_t - T), dtype=csi.dtype)
        return np.concatenate([csi, pad], axis=1)


def load_and_normalize_csi(h5_path, target_n=232, target_t=500):
    """
    Load CSI from h5 file and normalize to (target_n, target_t).
    Handles different key names across tasks.
    """
    with h5py.File(h5_path, "r") as hf:
        # Try common key names
        if "CSI_amps" in hf:
            csi = hf["CSI_amps"][:].squeeze(-1)
        elif "csi_amps" in hf:
            csi = hf["csi_amps"][:].squeeze(-1)
        elif "amplitude" in hf:
            csi = hf["amplitude"][:].squeeze(-1)
        else:
            # Use first available key
            key = list(hf.keys())[0]
            csi = hf[key][:].squeeze(-1)

    csi = normalize_subcarriers(csi, target_n=target_n)
    csi = normalize_time(csi, target_t=target_t)
    return csi


# ── Metadata utilities ─────────────────────────────────────

def load_metadata(data_root):
    meta = pd.read_csv(data_root + "/FallDetection/metadata/sample_metadata.csv")
    meta_hp = meta[meta["device"] == "HP"].copy().reset_index(drop=True)
    return meta_hp


def get_splits(data_root, meta_hp):
    with open(data_root + "/FallDetection/splits/train_id.json") as f:
        train_ids = set(json.load(f))
    with open(data_root + "/FallDetection/splits/test_easy.json") as f:
        test_ids = set(json.load(f))

    train_df = meta_hp[meta_hp["id"].isin(train_ids)].reset_index(drop=True)
    test_df  = meta_hp[meta_hp["id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


# ── Datasets ───────────────────────────────────────────────

class CSIFallDataset(Dataset):
    """Labeled FallDetection dataset."""

    def __init__(self, meta_df, data_root):
        self.meta      = meta_df.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row     = self.meta.iloc[idx]
        h5_path = self.data_root + "/FallDetection/" + row["file_path"].lstrip("./")

        csi = load_and_normalize_csi(h5_path)          # (232, 500)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)  # [1, 232, 500]

        label = torch.tensor(
            0 if row["label"] == "Fall" else 1, dtype=torch.long
        )
        return csi, label


class CSIPretrainDataset(Dataset):
    """Unlabeled pretraining dataset (single task)."""

    def __init__(self, meta_df, data_root, task="FallDetection"):
        self.meta      = meta_df.reset_index(drop=True)
        self.data_root = data_root
        self.task      = task

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row     = self.meta.iloc[idx]
        h5_path = os.path.join(
            self.data_root, self.task, row["file_path"].lstrip("./")
        )
        csi = load_and_normalize_csi(h5_path)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        return torch.tensor(csi, dtype=torch.float32).unsqueeze(0)


class CSIPretrainDatasetV2(Dataset):
    """
    Unlabeled multi-task pretraining dataset.
    Supports both direct tasks and Multitask folder structure.
    """
    def __init__(self, meta_df, data_root):
        self.meta      = meta_df.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        # Use pre-resolved h5_path if available, else resolve on the fly
        if "h5_path" in row and pd.notna(row["h5_path"]):
            h5_path = row["h5_path"]
        else:
            task    = row["task"]
            h5_path = os.path.join(
                self.data_root, task, row["file_path"].lstrip("./")
            )

        csi = load_and_normalize_csi(h5_path)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        return torch.tensor(csi, dtype=torch.float32).unsqueeze(0)


class MultiTaskDataset(Dataset):
    """
    Labeled multi-task dataset supporting all CSI-Bench tasks.
    Handles different subcarrier counts and time lengths automatically.
    Converts string labels to integers.
    """

    def __init__(self, meta_df, data_root, task, label_map=None):
        self.meta      = meta_df.reset_index(drop=True)
        self.data_root = data_root
        self.task      = task

        if label_map is None:
            unique_labels = sorted(meta_df["label"].unique(), key=str)
            self.label_map = {l: i for i, l in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row     = self.meta.iloc[idx]
        h5_path = os.path.join(
            self.data_root, self.task, row["file_path"].lstrip("./")
        )

        csi = load_and_normalize_csi(h5_path)          # (232, 500)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(
            self.label_map[row["label"]], dtype=torch.long
        )
        return csi, label


# ── DataLoader factories ───────────────────────────────────

def get_dataloaders(data_root=config.DATA_ROOT):
    meta_hp           = load_metadata(data_root)
    train_df, test_df = get_splits(data_root, meta_hp)

    train_loader = DataLoader(
        CSIFallDataset(train_df, data_root),
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        CSIFallDataset(test_df, data_root),
        batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    return train_loader, test_loader


def get_pretrain_dataloader(data_root=config.DATA_ROOT, sample_frac=0.3):
    """Pretrain loader: FallDetection + MotionSourceRecognition (HP device only)."""
    tasks = {
        "FallDetection":           "FallDetection/metadata/sample_metadata.csv",
        "MotionSourceRecognition": "MotionSourceRecognition/metadata/sample_metadata.csv",
    }

    dfs = []
    for task, meta_rel in tasks.items():
        df = pd.read_csv(os.path.join(data_root, meta_rel))
        df = df[df["device"] == "HP"].copy()
        df["task"] = task

        if task == "FallDetection":
            df["h5_path"] = df["file_path"].apply(
                lambda p: os.path.join(data_root, task, p.lstrip("./"))
            )
            df = df[df["h5_path"].apply(os.path.exists)].reset_index(drop=True)

        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"{task}: {len(df)} samples")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"Total pretrain: {len(combined)}")
    return DataLoader(
        CSIPretrainDatasetV2(combined, data_root),
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS
    )


def get_pretrain_dataloader_all(data_root=config.DATA_ROOT, sample_frac=0.3):
    """
    Pretrain loader: all 6 tasks, stratified by difficulty,
    subcarrier and time normalized to (232, 500).
    """
    tasks_with_difficulty = {
        "FallDetection":           ("FallDetection/metadata/sample_metadata.csv",           "Difficulty"),
        "MotionSourceRecognition": ("MotionSourceRecognition/metadata/sample_metadata.csv", "difficulty"),
    }

    # (meta_csv_path, task_root_dir) — both relative to data_root
    tasks_no_difficulty = {
        "Multitask/HumanActivityRecognition": (
            "Multitask/HumanActivityRecognition/metadata/sample_metadata.csv",
            "Multitask",
        ),
        "Multitask/HumanIdentification": (
            "Multitask/HumanIdentification/metadata/sample_metadata.csv",
            "Multitask",
        ),
        "Multitask/ProximityRecognition": (
            "Multitask/ProximityRecognition/metadata/sample_metadata.csv",
            "Multitask",
        ),
        "Localization": (
            "Localization/metadata/sample_metadata.csv",
            "Localization",
        ),
    }

    dfs = []

    for task, (meta_rel, diff_col) in tasks_with_difficulty.items():
        df = pd.read_csv(os.path.join(data_root, meta_rel))
        if "device" in df.columns:
            df = df[df["device"] == "HP"].copy()
        df["task"] = task

        if task == "FallDetection":
            df["h5_path"] = df["file_path"].apply(
                lambda p: os.path.join(data_root, task, p.lstrip("./"))
            )
            df = df[df["h5_path"].apply(os.path.exists)].reset_index(drop=True)

        sampled = df.groupby(diff_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=42)
        ).reset_index(drop=True)

        print(f"{task}: {len(sampled)} samples (stratified {int(sample_frac*100)}%)")
        dfs.append(sampled)

    for task, (meta_rel, task_root) in tasks_no_difficulty.items():
        df = pd.read_csv(os.path.join(data_root, meta_rel))
        df["task"] = task

        meta_dir = os.path.dirname(os.path.join(data_root, meta_rel))
        task_dir = os.path.join(data_root, task_root)

        def resolve(p, md=meta_dir, td=task_dir):
            # Multitask uses ../../sub_Human_h5/... (relative to metadata dir)
            candidate = os.path.normpath(os.path.join(md, p))
            if os.path.exists(candidate):
                return candidate
            # Localization uses ./sub_Human/... (relative to task root)
            return os.path.normpath(os.path.join(td, p.lstrip("./")))

        df["h5_path"] = df["file_path"].apply(resolve)
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"{task}: {len(df)} samples (uniform {int(sample_frac*100)}%)")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"\nTotal pretrain: {len(combined)}")
    return DataLoader(
        CSIPretrainDatasetV2(combined, data_root),
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS
    )


class CombinedDataset(Dataset):
    """
    Labeled dataset for combined multi-task evaluation.
    Uses pre-resolved h5_path and global_label columns.
    """
    def __init__(self, meta_df, data_root):
        self.meta      = meta_df.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row     = self.meta.iloc[idx]
        h5_path = row["h5_path"]

        csi = load_and_normalize_csi(h5_path)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(int(row["global_label"]), dtype=torch.long)
        return csi, label