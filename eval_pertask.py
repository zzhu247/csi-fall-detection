# eval_pertask.py
import os, json, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from models.mae import MAE
from data.dataset import MultiTaskDataset
from eval.knn_probe import knn_eval
import config

config.N_LAYERS = 12
device = torch.device("cuda")

SPLITS_DIR = os.path.expanduser("~/data/splits")

TASKS = {
    "FallDetection":           {"num_classes": 2,  "decoder_dim": 128},
    "MotionSourceRecognition": {"num_classes": 4,  "decoder_dim": 128},
}

CHECKPOINTS = {
    "mae_341k_200":  ("checkpoints/mae_ep200_mask0.75_dec128_bs64_best.pth",  128),
    "mae_341k_300":  ("checkpoints/mae_ep300_mask0.75_dec128_bs64_best.pth",  128),
    "mae_341k_500":  ("checkpoints/mae_ep500_mask0.75_dec128_bs64_best.pth",  128),
    "mae_341k_1000": ("checkpoints/mae_ep1000_mask0.75_dec128_bs64_best.pth", 128),
}

LAYERS   = [1, 4, 8, 12]
K_VALUES = [5, 10, 20]


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


def run_linear_probe(model, train_loader, eval_loader,
                     layer, num_classes, device, epochs=100):
    train_feats, train_labels = get_feats(model, train_loader, layer, device)
    eval_feats,  eval_labels  = get_feats(model, eval_loader,  layer, device)

    # Normalize features
    mean = train_feats.mean(0, keepdim=True)
    std  = train_feats.std(0,  keepdim=True) + 1e-8
    train_feats = (train_feats - mean) / std
    eval_feats  = (eval_feats  - mean) / std

    # Weighted loss for class imbalance
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    weights  = 1.0 / (class_counts + 1e-8)
    weights  = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    head    = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optim   = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    best_acc   = 0.0
    no_improve = 0

    for ep in range(epochs):
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(head(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

        if (ep + 1) % 10 == 0:
            head.eval()
            with torch.no_grad():
                preds = head(eval_feats.to(device)).argmax(dim=1).cpu()
            acc = (preds == eval_labels).float().mean().item()
            if acc > best_acc:
                best_acc   = acc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 3:
                break

    return best_acc


# ── Main evaluation loop ──────────────────────────────────
all_results = {}

for ckpt_name, (ckpt_path, dec_dim) in CHECKPOINTS.items():
    if not os.path.exists(ckpt_path):
        print(f"Skip {ckpt_name}: checkpoint not found")
        continue

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
            print(f"  No splits found, skipping")
            continue

        train_df  = pd.read_csv(train_path)
        label_map = {l: i for i, l in
                     enumerate(sorted(train_df["label"].unique(), key=str))}

        train_loader = DataLoader(
            MultiTaskDataset(train_df, config.DATA_ROOT, task, label_map),
            batch_size=64, shuffle=True, num_workers=4, pin_memory=True
        )

        # ── Build val + merged test loaders ───────────────
        loaders = {}

        # Val
        val_path = os.path.join(SPLITS_DIR, f"{task}_val.csv")
        if os.path.exists(val_path):
            val_df = pd.read_csv(val_path)
            if len(val_df) > 0:
                loaders["val"] = DataLoader(
                    MultiTaskDataset(val_df, config.DATA_ROOT, task, label_map),
                    batch_size=64, shuffle=False, num_workers=4, pin_memory=True
                )

        # Test: merge easy + hard
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
            print(f"  test merged: {len(test_merged)} samples (easy+hard)")

        # ── Evaluate each split ────────────────────────────
        task_results = {}

        for split_name, eval_loader in loaders.items():
            print(f"\n    -- {split_name} --")
            task_results[split_name] = {}

            # KNN
            knn_res      = knn_eval(model, train_loader, eval_loader,
                                    layers=LAYERS, k_values=K_VALUES,
                                    device=device)
            best_knn     = max(float(v) for v in knn_res.values())
            best_knn_key = max(knn_res, key=lambda k: float(knn_res[k]))

            task_results[split_name]["knn"] = {
                str(k): float(v) for k, v in knn_res.items()
            }
            task_results[split_name]["best_knn"]    = round(best_knn, 4)
            task_results[split_name]["best_knn_at"] = str(best_knn_key)

            # Linear Probe per layer
            lp_results    = {}
            best_lp       = 0.0
            best_lp_layer = None

            for layer in LAYERS:
                acc = run_linear_probe(
                    model, train_loader, eval_loader,
                    layer, task_info["num_classes"], device
                )
                lp_results[f"layer_{layer}"] = round(acc, 4)
                print(f"    LP L{layer}: {acc:.3f}")
                if acc > best_lp:
                    best_lp       = acc
                    best_lp_layer = layer

            task_results[split_name]["linear_probe"]  = lp_results
            task_results[split_name]["best_lp"]       = round(best_lp, 4)
            task_results[split_name]["best_lp_layer"] = best_lp_layer

            print(f"    → KNN={best_knn:.3f} (at {best_knn_key})  "
                  f"LP={best_lp:.3f} (L{best_lp_layer})")

        all_results[ckpt_name][task] = task_results

# ── Save ──────────────────────────────────────────────────
with open("results/pertask_eval.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ── Print summary ─────────────────────────────────────────
print("\n\n" + "="*65)
print("PER-TASK EVALUATION SUMMARY")
print("="*65)
print(f"{'Checkpoint':<18} {'Task':<25} {'Split':<8} {'KNN':>6} {'LP':>6}")
print("-"*65)
for ckpt, tasks in all_results.items():
    for task, splits in tasks.items():
        for split, m in splits.items():
            print(f"{ckpt:<18} {task:<25} {split:<8} "
                  f"{m['best_knn']:>6.3f} {m['best_lp']:>6.3f}")

print("\nSaved: results/pertask_eval.json")