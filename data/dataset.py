# data/dataset.py

import h5py
import json
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader

import config


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


class CSIFallDataset(Dataset):
    def __init__(self, meta_df, data_root):
        self.meta      = meta_df
        self.data_root = data_root

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row    = self.meta.iloc[idx]
        h5_path = self.data_root + "/FallDetection/" + row["file_path"].lstrip("./")

        with h5py.File(h5_path, "r") as hf:
            csi = hf["CSI_amps"][:].squeeze(-1)        # (232, 500)

        csi   = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi   = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)  # [1, 232, 500]
        label = torch.tensor(
            0 if row["label"] == "Fall" else 1,
            dtype=torch.long
        )
        return csi, label
    
class CSIPretrainDataset(Dataset):
    # Dataset for self-supervised pretraining.
    # Loads CSI from multiple tasks, returns signal only (no label).
    
    def __init__(self, meta_df, data_root):
        self.meta      = meta_df
        self.data_root = data_root

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row     = self.meta.iloc[idx]
        task    = row["task"]
        h5_path = os.path.join(
            self.data_root, task, row["file_path"].lstrip("./")
        )

        with h5py.File(h5_path, "r") as hf:
            csi = hf["CSI_amps"][:].squeeze(-1)   # (232, 500)

        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)  # [1, 232, 500]
        return csi


def get_dataloaders(data_root=config.DATA_ROOT):
    meta_hp             = load_metadata(data_root)
    train_df, test_df   = get_splits(data_root, meta_hp)

    train_loader = DataLoader(
        CSIFallDataset(train_df, data_root),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        CSIFallDataset(test_df, data_root),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    return train_loader, test_loader

def get_pretrain_dataloader(data_root=config.DATA_ROOT, sample_frac=0.3):
    tasks = {
        "FallDetection":           "FallDetection/metadata/sample_metadata.csv",
        "MotionSourceRecognition": "MotionSourceRecognition/metadata/sample_metadata.csv",
    }

    dfs = []
    for task, meta_rel_path in tasks.items():
        df = pd.read_csv(os.path.join(data_root, meta_rel_path))
        df = df[df["device"] == "HP"].copy()
        df["task"] = task

        # Filter missing files
        df["h5_path"] = df["file_path"].apply(
            lambda p: os.path.join(data_root, task, p.lstrip("./"))
        )
        df = df[df["h5_path"].apply(os.path.exists)].reset_index(drop=True)

        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"{task}: {len(df)} samples after filter")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"\nTotal pretrain samples: {len(combined)}")
    return DataLoader(
        CSIPretrainDataset(combined, data_root),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )