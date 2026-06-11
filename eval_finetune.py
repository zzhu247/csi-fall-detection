# eval_finetune.py
import os, json, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from models.mae import MAE
from data.dataset import MultiTaskDataset
import config

config.N_LAYERS = 12
device = torch.device("cuda")

SPLITS_DIR = os.path.expanduser("~/data/splits")

TASKS = {
    "FallDetection":           {"num_classes": 2,  "decoder_dim": 128},
    "MotionSourceRecognition": {"num_classes": 4,  "decoder_dim": 128},
}

CHECKPOINTS = {
    "mae_341k_200": ("checkpoints/mae_ep200_mask0.75_dec128_bs64_best.pth", 128),
    "mae_341k_300": ("checkpoints/mae_ep300_mask0.75_dec128_bs64_best.pth", 128),  
    "mae_341k_500": ("checkpoints/mae_ep500_mask0.75_dec128_bs64_best.pth", 128),
}

FINETUNE_EPOCHS = 50
ENCODER_LR      = 1e-5   # small lr for pretrained encoder
HEAD_LR         = 1e-3   # larger lr for new head


def finetune_eval(model, train_loader, eval_loader,
                  num_classes, device,
                  epochs=FINETUNE_EPOCHS):
    """
    Full fine-tuning: unfreeze encoder + train classification head.
    Uses layer-wise learning rate decay (encoder gets smaller lr).
    """
    # Add classification head on top of encoder
    head = nn.Linear(config.D_MODEL, num_classes).to(device)

    # Layer-wise lr: encoder gets small lr, head gets large lr
    optimizer = torch.optim.AdamW([
    {"params": model.patch_embedding.parameters(), "lr": ENCODER_LR},
    {"params": [model.encoder_pos_embed],          "lr": ENCODER_LR},
    {"params": model.encoder_blocks.parameters(),  "lr": ENCODER_LR},
    {"params": model.encoder_norm.parameters(),    "lr": ENCODER_LR},
    {"params": head.parameters(),                  "lr": HEAD_LR},
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Weighted loss for class imbalance
    all_labels = []
    for _, y in train_loader:
        all_labels.append(y)
    all_labels = torch.cat(all_labels)
    class_counts = torch.bincount(all_labels, minlength=num_classes).float()
    weights  = 1.0 / (class_counts + 1e-8)
    weights  = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_acc  = 0.0
    best_state = None

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────
        model.train()
        head.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward through encoder (all layers unfrozen)
            tokens  = model.patch_embedding(x) + model.encoder_pos_embed
            encoded = model.encoder_blocks(tokens)
            encoded = model.encoder_norm(encoded)
            feat    = encoded.mean(dim=1)          # global average pool

            logits = head(feat)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step()
        train_acc = correct / total

        # ── Eval ─────────────────────────────────────────
        model.eval()
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                tokens  = model.patch_embedding(x) + model.encoder_pos_embed
                encoded = model.encoder_blocks(tokens)
                encoded = model.encoder_norm(encoded)
                feat    = encoded.mean(dim=1)
                preds   = head(feat).argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)

        eval_acc = correct / total

        if eval_acc > best_acc:
            best_acc   = eval_acc
            best_state = {
                "encoder": {k: v.clone() for k, v in model.state_dict().items()},
                "head":    {k: v.clone() for k, v in head.state_dict().items()},
            }

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:02d}/{epochs} | "
                  f"Train: {train_acc:.3f}  Eval: {eval_acc:.3f}  "
                  f"Best: {best_acc:.3f}")

    return best_acc


# ── Main ──────────────────────────────────────────────────
all_results = {}

for ckpt_name, (ckpt_path, dec_dim) in CHECKPOINTS.items():
    if not os.path.exists(ckpt_path):
        print(f"Skip {ckpt_name}: checkpoint not found")
        continue

    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_name}")

    all_results[ckpt_name] = {}

    for task, task_info in TASKS.items():
        print(f"\n  ── {task} ──")

        train_path = os.path.join(SPLITS_DIR, f"{task}_train.csv")
        if not os.path.exists(train_path):
            print(f"  No splits found, skipping")
            continue

        train_df  = pd.read_csv(train_path)
        label_map = {l: i for i, l in
                     enumerate(sorted(train_df["label"].unique(), key=str))}

        train_loader = DataLoader(
            MultiTaskDataset(train_df, config.DATA_ROOT, task, label_map),
            batch_size=64, shuffle=True, num_workers=4, pin_memory=True
        )

        # Build val + merged test loaders
        loaders = {}

        val_path = os.path.join(SPLITS_DIR, f"{task}_val.csv")
        if os.path.exists(val_path):
            val_df = pd.read_csv(val_path)
            if len(val_df) > 0:
                loaders["val"] = DataLoader(
                    MultiTaskDataset(val_df, config.DATA_ROOT, task, label_map),
                    batch_size=64, shuffle=False, num_workers=4, pin_memory=True
                )

        test_dfs = []
        for split in ["test_easy", "test_hard"]:
            path = os.path.join(SPLITS_DIR, f"{task}_{split}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                if len(df) > 0:
                    test_dfs.append(df)
        if test_dfs:
            test_merged = pd.concat(test_dfs, ignore_index=True).reset_index(drop=True)
            loaders["test"] = DataLoader(
                MultiTaskDataset(test_merged, config.DATA_ROOT, task, label_map),
                batch_size=64, shuffle=False, num_workers=4, pin_memory=True
            )
            print(f"  test merged: {len(test_merged)} samples")

        task_results = {}

        for split_name, eval_loader in loaders.items():
            print(f"\n    -- Fine-tune eval on {split_name} --")

            # Reload fresh pretrained weights for each eval split
            model = MAE(
                in_channels=1, img_h=232, img_w=500, patch_h=8, patch_w=25,
                encoder_dim=128, encoder_ff_dim=512, encoder_heads=4,
                encoder_depth=12, decoder_dim=dec_dim, decoder_heads=2,
                decoder_depth=2, mask_ratio=0.75
            ).to(device)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device)
            )

            acc = finetune_eval(
                model, train_loader, eval_loader,
                task_info["num_classes"], device
            )
            task_results[split_name] = {"finetune_acc": round(acc, 4)}
            print(f"    → Fine-tune best acc: {acc:.3f}")

        all_results[ckpt_name][task] = task_results

# ── Save ──────────────────────────────────────────────────
with open("results/finetune_eval.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ── Print summary ─────────────────────────────────────────
print("\n\n" + "="*65)
print("FINE-TUNE EVALUATION SUMMARY")
print("="*65)
print(f"{'Checkpoint':<18} {'Task':<25} {'Split':<8} {'FT Acc':>8}")
print("-"*65)

# Compare with KNN and LP from pertask eval
try:
    with open("results/pertask_eval.json") as f:
        pertask = json.load(f)
    print(f"\n{'Checkpoint':<18} {'Task':<25} {'Split':<8} "
          f"{'KNN':>6} {'LP':>6} {'FT':>6}")
    print("-"*65)
    for ckpt, tasks in all_results.items():
        for task, splits in tasks.items():
            for split, m in splits.items():
                knn = pertask.get(ckpt, {}).get(task, {}).get(
                    split, {}).get("best_knn", 0)
                lp  = pertask.get(ckpt, {}).get(task, {}).get(
                    split, {}).get("best_lp", 0)
                ft  = m["finetune_acc"]
                print(f"{ckpt:<18} {task:<25} {split:<8} "
                      f"{knn:>6.3f} {lp:>6.3f} {ft:>6.3f}")
except FileNotFoundError:
    for ckpt, tasks in all_results.items():
        for task, splits in tasks.items():
            for split, m in splits.items():
                print(f"{ckpt:<18} {task:<25} {split:<8} "
                      f"{m['finetune_acc']:>8.3f}")

print("\nSaved: results/finetune_eval.json")
