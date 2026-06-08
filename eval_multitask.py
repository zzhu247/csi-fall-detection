# eval_multitask.py
# Evaluate MAE encoder on all tasks using KNN + Linear Probe

import os, sys, json, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from models.mae import MAE
from data.dataset import MultiTaskDataset
from eval.knn_probe import knn_eval
import config

config.N_LAYERS = 12
device = torch.device("cuda")

SPLITS_DIR  = os.path.expanduser("~/data/splits")
LAYERS      = [1, 4, 8, 12]
K_VALUES    = [5, 10]

TASKS = {
    "FallDetection":          {"num_classes": 2,  "decoder_dim": 128},
    "MotionSourceRecognition":{"num_classes": 4,  "decoder_dim": 128},
    # "BreathingDetection":     {"num_classes": 2,  "decoder_dim": 128},
    # "Localization":           {"num_classes": 6,  "decoder_dim": 128},
}

CHECKPOINTS = {
    "mae_200": ("checkpoints/mae_ep200_mask0.75_dec64_bs32_best.pth",  64),
    "mae_300": ("checkpoints/mae_ep300_mask0.75_dec64_bs32_best.pth",  64),
    "mae_500": ("checkpoints/mae_ep500_mask0.75_dec128_bs32_best.pth", 128),
}

def get_feats(model, loader, layer, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model.extract_layer_embeddings(x, [layer])
            feats.append(emb[layer].cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def linear_probe(model, train_loader, eval_loader, layer, num_classes, device, epochs=100):
    train_feats, train_labels = get_feats(model, train_loader, layer, device)
    eval_feats,  eval_labels  = get_feats(model, eval_loader,  layer, device)

    # Normalize
    mean = train_feats.mean(0, keepdim=True)
    std  = train_feats.std(0, keepdim=True) + 1e-8
    train_feats = (train_feats - mean) / std
    eval_feats  = (eval_feats  - mean) / std

    # Weighted loss
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    weights = 1.0 / (class_counts + 1e-8)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    head      = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optim     = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    best_acc = 0.0
    for ep in range(epochs):
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(head(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()
        scheduler.step()

        if (ep+1) % 20 == 0:
            head.eval()
            with torch.no_grad():
                preds = head(eval_feats.to(device)).argmax(dim=1).cpu()
            acc = (preds == eval_labels).float().mean().item()
            if acc > best_acc:
                best_acc = acc

    return best_acc


# ── Main evaluation loop ──────────────────────────────────
all_results = {}

for ckpt_name, (ckpt_path, dec_dim) in CHECKPOINTS.items():
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_name}")

    model = MAE(
        in_channels=1, img_h=232, img_w=500, patch_h=8, patch_w=25,
        encoder_dim=128, encoder_ff_dim=512, encoder_heads=4, encoder_depth=12,
        decoder_dim=dec_dim, decoder_heads=2, decoder_depth=2,
        mask_ratio=0.75
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    all_results[ckpt_name] = {}

    for task, task_info in TASKS.items():
        print(f"\n  ── {task} ──")

        train_path = os.path.join(SPLITS_DIR, f"{task}_train.csv")
        if not os.path.exists(train_path):
            print(f"  No splits found, skipping.")
            continue

        train_df = pd.read_csv(train_path)
        label_map = {l: i for i, l in enumerate(sorted(train_df["label"].unique(), key=str))}

        loaders = {}
        for split in ["val", "test_easy", "test_hard"]:
            split_path = os.path.join(SPLITS_DIR, f"{task}_{split}.csv")
            if not os.path.exists(split_path):
                continue
            df = pd.read_csv(split_path)
            if len(df) == 0:
                continue
            loaders[split] = DataLoader(
                MultiTaskDataset(df, config.DATA_ROOT, task, label_map),
                batch_size=64, shuffle=False, num_workers=4, pin_memory=True
            )

        train_loader = DataLoader(
            MultiTaskDataset(train_df, config.DATA_ROOT, task, label_map),
            batch_size=64, shuffle=True, num_workers=4, pin_memory=True
        )

        task_results = {}
        for split_name, eval_loader in loaders.items():
            print(f"\n    {split_name}:")
            task_results[split_name] = {"knn": {}, "linear_probe": {}}

            # KNN
            knn_res = knn_eval(model, train_loader, eval_loader,
                               layers=LAYERS, k_values=K_VALUES, device=device)
            task_results[split_name]["knn"] = {
                str(k): float(v) for k, v in knn_res.items()
            }

            # Linear Probe
            for layer in LAYERS:
                acc = linear_probe(model, train_loader, eval_loader,
                                   layer, task_info["num_classes"], device)
                task_results[split_name]["linear_probe"][f"layer_{layer}"] = round(acc, 4)
                print(f"    Linear Probe L{layer}: {acc:.3f}")

        all_results[ckpt_name][task] = task_results

# Save
with open("results/multitask_eval.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print summary table
print("\n\n" + "="*70)
print("MULTI-TASK EVALUATION SUMMARY (Best across layers)")
print("="*70)
print(f"{'Checkpoint':<12} {'Task':<25} {'Split':<12} {'KNN':>7} {'LP':>7}")
print("-"*70)

for ckpt, tasks in all_results.items():
    for task, splits in tasks.items():
        for split, metrics in splits.items():
            best_knn = max(float(v) for v in metrics["knn"].values()) if metrics["knn"] else 0
            best_lp  = max(metrics["linear_probe"].values()) if metrics["linear_probe"] else 0
            print(f"{ckpt:<12} {task:<25} {split:<12} {best_knn:>7.3f} {best_lp:>7.3f}")

print("\nSaved: results/multitask_eval.json")
