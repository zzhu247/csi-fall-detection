import math
# eval_benchmark.py
#
# Replicates CSI-Bench paper:
#   Table 3 — supervised baseline accuracy + weighted F1 on standard splits
#   Table 5 — OOD evaluation (cross-device, cross-env, cross-user)
#
# Usage:
#   # Train + eval ResNet-18 on FallDetection (Table 3)
#   python eval_benchmark.py --task FallDetection --model resnet18 --mode supervised
#
#   # Train + eval ViT on HAR (Table 3)
#   python eval_benchmark.py --task HumanActivityRecognition --model vit --mode supervised
#
#   # OOD eval on HAR with trained checkpoint (Table 5)
#   python eval_benchmark.py --task HumanActivityRecognition --model resnet18 \
#       --mode ood --checkpoint checkpoints/resnet18_HAR.pt

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

import config
from data.dataset import load_and_normalize_csi


# ── Dataset ───────────────────────────────────────────────────────────────────

TASK_CONFIG = {
    "FallDetection": {
        "meta":       "FallDetection/metadata/sample_metadata.csv",
        "splits_dir": "FallDetection/splits",
        "task_root":  "FallDetection",
        "path_mode":  "relative_to_task",
        "ood_splits": None,
        "split_suffix": "",          # test_easy.json
    },
    "MotionSourceRecognition": {
        "meta":       "MotionSourceRecognition/metadata/sample_metadata.csv",
        "splits_dir": "MotionSourceRecognition/splits",
        "task_root":  "MotionSourceRecognition",
        "path_mode":  "relative_to_task",
        "ood_splits": None,
        "split_suffix": "",
    },
    "Localization": {
        "meta":       "Localization/metadata/sample_metadata.csv",
        "splits_dir": "Localization/splits",
        "task_root":  "Localization",
        "path_mode":  "relative_to_task",
        "ood_splits": None,
        "split_suffix": "_id",       # test_easy_id.json
    },
    "HumanActivityRecognition": {
        "meta":       "Multitask/HumanActivityRecognition/metadata/sample_metadata.csv",
        "splits_dir": "Multitask/HumanActivityRecognition/splits",
        "task_root":  "Multitask",
        "path_mode":  "relative_to_meta",
        "ood_splits": ["test_cross_device", "test_cross_env", "test_cross_user"],
        "split_suffix": "",
    },
    "HumanIdentification": {
        "meta":       "Multitask/HumanIdentification/metadata/sample_metadata.csv",
        "splits_dir": "Multitask/HumanIdentification/splits",
        "task_root":  "Multitask",
        "path_mode":  "relative_to_meta",
        "ood_splits": ["test_cross_device"],
        "split_suffix": "",
    },
    "ProximityRecognition": {
        "meta":       "Multitask/ProximityRecognition/metadata/sample_metadata.csv",
        "splits_dir": "Multitask/ProximityRecognition/splits",
        "task_root":  "Multitask",
        "path_mode":  "relative_to_meta",
        "ood_splits": ["test_cross_device", "test_cross_env", "test_cross_user"],
        "split_suffix": "",
    },
}


def resolve_path(file_path, path_mode, data_root, task_root, meta_csv):
    if path_mode == "relative_to_task":
        return os.path.normpath(
            os.path.join(data_root, task_root, file_path.lstrip("./"))
        )
    else:  # relative_to_meta
        meta_dir = os.path.dirname(os.path.join(data_root, meta_csv))
        candidate = os.path.normpath(os.path.join(meta_dir, file_path))
        if os.path.exists(candidate):
            return candidate
        return os.path.normpath(
            os.path.join(data_root, task_root, file_path.lstrip("./"))
        )


class BenchmarkDataset(Dataset):
    def __init__(self, meta_df, data_root, task_cfg):
        self.meta      = meta_df.reset_index(drop=True)
        self.data_root = data_root
        self.task_cfg  = task_cfg

        # Build label map
        unique = sorted(meta_df["label"].unique(), key=str)
        self.label_map  = {l: i for i, l in enumerate(unique)}
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        if "h5_path" in row and pd.notna(row.get("h5_path")):
            h5_path = row["h5_path"]
        else:
            h5_path = resolve_path(
                row["file_path"],
                self.task_cfg["path_mode"],
                self.data_root,
                self.task_cfg["task_root"],
                self.task_cfg["meta"],
            )

        csi = load_and_normalize_csi(h5_path)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.label_map[row["label"]], dtype=torch.long)
        return csi, label


def load_split(data_root, splits_dir, split_name, meta_df, split_suffix=""):
    # Handle both test_easy.json and test_easy_id.json naming conventions
    split_path = os.path.join(data_root, splits_dir, f"{split_name}{split_suffix}.json")
    if not os.path.exists(split_path):
        # Try without suffix as fallback
        split_path = os.path.join(data_root, splits_dir, f"{split_name}.json")
    with open(split_path) as f:
        ids = set(json.load(f))
    return meta_df[meta_df["id"].isin(ids)].reset_index(drop=True)


# ── Models ────────────────────────────────────────────────────────────────────

def build_model(model_name, num_classes, device):
    from models.csibench_models import (
        MLPClassifier, LSTMClassifier, ResNet18Classifier,
        TransformerClassifier, ViTClassifier, PatchTST, TimesFormer1D
    )
    kwargs = dict(win_len=500, feature_size=232, num_classes=num_classes)
    if model_name == "mlp":
        return MLPClassifier(**kwargs).to(device)
    elif model_name == "lstm":
        return LSTMClassifier(feature_size=232, num_classes=num_classes).to(device)
    elif model_name == "resnet18":
        return ResNet18Classifier(**kwargs).to(device)
    elif model_name == "transformer":
        return TransformerClassifier(feature_size=232, num_classes=num_classes).to(device)
    elif model_name in ("vit", "vit_paper"):
        return ViTClassifier(**kwargs).to(device)
    elif model_name == "patchtst":
        return PatchTST(**kwargs).to(device)
    elif model_name == "timesformer1d":
        return TimesFormer1D(**kwargs).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_supervised(model, train_loader, val_loader, device,
                     epochs=100, lr=1e-3, patience=15, warmup_epochs=5,
                     weight_decay=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(epochs):
        model.train()
        for csi, labels in train_loader:
            csi, labels = csi.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(csi), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_acc, _ = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d} | val_acc={val_acc:.4f} | best={best_val_acc:.4f}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for csi, labels in loader:
        csi = csi.to(device)
        preds = model(csi).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = (all_preds == all_labels).mean()
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return float(acc), float(f1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  required=True, choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--model", required=True,
                        choices=["mlp","lstm","transformer","vit_paper",
                                 "patchtst","timesformer1d","resnet18","vit"])
    parser.add_argument("--mode",  default="supervised",
                        choices=["supervised", "ood"])
    parser.add_argument("--checkpoint", default=None,
                        help="Load pretrained checkpoint for OOD eval")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--save",      default=None,
                        help="Path to save trained checkpoint")
    args = parser.parse_args()

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = config.DATA_ROOT
    task_cfg  = TASK_CONFIG[args.task]

    print(f"\nTask:  {args.task}")
    print(f"Model: {args.model}")
    print(f"Mode:  {args.mode}")
    print(f"Device:{device}\n")

    # Load metadata
    meta = pd.read_csv(os.path.join(data_root, task_cfg["meta"]))

    # Pre-resolve and filter missing files for relative_to_task tasks
    if task_cfg["path_mode"] == "relative_to_task":
        meta["h5_path"] = meta["file_path"].apply(
            lambda p: os.path.normpath(
                os.path.join(data_root, task_cfg["task_root"], p.lstrip("./"))
            )
        )
        before = len(meta)
        meta = meta[meta["h5_path"].apply(os.path.exists)].reset_index(drop=True)
        print(f"Filtered missing files: {before} -> {len(meta)}")

    # Pre-resolve and filter missing files for FallDetection
    if task_cfg["path_mode"] == "relative_to_task":
        meta["h5_path"] = meta["file_path"].apply(
            lambda p: os.path.normpath(
                os.path.join(data_root, task_cfg["task_root"], p.lstrip("./"))
            )
        )
        before = len(meta)
        meta = meta[meta["h5_path"].apply(os.path.exists)].reset_index(drop=True)
        print(f"Filtered missing files: {before} -> {len(meta)}")

    # Pre-resolve h5_path for Multitask (relative paths)
    if task_cfg["path_mode"] == "relative_to_meta":
        meta_dir = os.path.dirname(os.path.join(data_root, task_cfg["meta"]))
        task_dir = os.path.join(data_root, task_cfg["task_root"])
        def resolve(p):
            c = os.path.normpath(os.path.join(meta_dir, p))
            return c if os.path.exists(c) else os.path.normpath(
                os.path.join(task_dir, p.lstrip("./"))
            )
        meta["h5_path"] = meta["file_path"].apply(resolve)

    # Load train/val splits
    sfx = task_cfg.get("split_suffix", "")
    train_df = load_split(data_root, task_cfg["splits_dir"], "train_id", meta, sfx)
    val_df   = load_split(data_root, task_cfg["splits_dir"], "val_id",   meta, sfx)
    test_df  = load_split(data_root, task_cfg["splits_dir"], "test_id",  meta, sfx)

    train_ds = BenchmarkDataset(train_df, data_root, task_cfg)
    val_ds   = BenchmarkDataset(val_df,   data_root, task_cfg)
    test_ds  = BenchmarkDataset(test_df,  data_root, task_cfg)

    num_classes = train_ds.num_classes
    print(f"Classes: {num_classes}  |  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build model
    model = build_model(args.model, num_classes, device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.2f}M\n")

    # Load checkpoint or train
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Training supervised baseline...")
        model = train_supervised(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
        )
        if args.save:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save({"model_state": model.state_dict()}, args.save)
            print(f"Saved: {args.save}")

    # ── Standard evaluation ────────────────────────────────────
    results = {}

    test_acc, test_f1 = evaluate(model, test_loader, device)
    results["test"] = {"acc": test_acc, "f1": test_f1}
    print(f"\n{'='*50}")
    print(f"Standard Test  |  Acc={test_acc*100:.2f}%  F1={test_f1*100:.2f}%")

    # Difficulty splits (Easy / Medium / Hard) — single-task only
    for split in ["test_easy", "test_medium", "test_hard"]:
        split_path = os.path.join(data_root, task_cfg["splits_dir"], f"{split}.json")
        if os.path.exists(split_path):
            df_split = load_split(data_root, task_cfg["splits_dir"], split, meta)
            if len(df_split) > 0:
                ds = BenchmarkDataset(df_split, data_root, task_cfg)
                loader = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)
                acc, f1 = evaluate(model, loader, device)
                results[split] = {"acc": acc, "f1": f1}
                print(f"{split:20s}  |  Acc={acc*100:.2f}%  F1={f1*100:.2f}%")

    # ── OOD evaluation (Table 5) ───────────────────────────────
    if args.mode == "ood" and task_cfg["ood_splits"]:
        print(f"\n{'='*50}")
        print("OOD Evaluation (Table 5):")
        for ood_split in task_cfg["ood_splits"]:
            split_path = os.path.join(
                data_root, task_cfg["splits_dir"], f"{ood_split}.json"
            )
            if not os.path.exists(split_path):
                print(f"  {ood_split}: split file not found, skipping")
                continue
            df_ood = load_split(data_root, task_cfg["splits_dir"], ood_split, meta)
            if len(df_ood) == 0:
                print(f"  {ood_split}: 0 samples, skipping")
                continue
            ds_ood = BenchmarkDataset(df_ood, data_root, task_cfg)
            loader_ood = DataLoader(ds_ood, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)
            acc, f1 = evaluate(model, loader_ood, device)
            results[ood_split] = {"acc": acc, "f1": f1}
            print(f"  {ood_split:25s}  |  Acc={acc*100:.2f}%  F1={f1*100:.2f}%")

    print(f"\n{'='*50}")

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/benchmark_{args.task}_{args.model}_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump({"task": args.task, "model": args.model,
                   "mode": args.mode, "results": results}, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
