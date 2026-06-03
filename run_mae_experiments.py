# run_mae_experiments.py
# Main script to run MAE pretraining + KNN evaluation
# across different hyperparameter combinations

import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from itertools import product

import config
from models.mae import MAE
from data.dataset import (CSIFallDataset, CSIPretrainDatasetV2,
                           load_metadata, get_splits)
from train_mae import train_one_epoch_mae
from eval.knn_probe import knn_eval, build_table


# ── Data setup ────────────────────────────────────────────
def get_dataloaders(batch_size):
    pretrain_df  = pd.read_csv('/home/zhuzih19/data/pretrain_combined.csv')
    train_df     = pd.read_csv('/home/zhuzih19/data/train.csv')
    val_df       = pd.read_csv('/home/zhuzih19/data/val.csv')
    test_easy_df = pd.read_csv('/home/zhuzih19/data/test_easy.csv')
    test_hard_df = pd.read_csv('/home/zhuzih19/data/test_hard.csv')

    pretrain_loader = DataLoader(
        CSIPretrainDatasetV2(pretrain_df, config.DATA_ROOT),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    train_loader = DataLoader(
        CSIFallDataset(train_df, config.DATA_ROOT),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        CSIFallDataset(val_df, config.DATA_ROOT),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_easy_loader = DataLoader(
        CSIFallDataset(test_easy_df, config.DATA_ROOT),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_hard_loader = DataLoader(
        CSIFallDataset(test_hard_df, config.DATA_ROOT),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Pretrain: {len(pretrain_df)} | Train: {len(train_df)} | "
          f"Val: {len(val_df)} | Test Easy: {len(test_easy_df)} | "
          f"Test Hard: {len(test_hard_df)}")

    return pretrain_loader, train_loader, val_loader, test_easy_loader, test_hard_loader

    def file_exists(row):
        return os.path.exists(
            config.DATA_ROOT + "/FallDetection/" + row["file_path"].lstrip("./")
        )

    train_df = train_df[train_df.apply(file_exists, axis=1)].reset_index(drop=True)
    test_df  = test_df[test_df.apply(file_exists, axis=1)].reset_index(drop=True)

    train_loader = DataLoader(
        CSIFallDataset(train_df, config.DATA_ROOT),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        CSIFallDataset(test_df, config.DATA_ROOT),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Pretrain: {len(combined_df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    return pretrain_loader, train_loader, test_loader


# ── Hyperparameter grid ────────────────────────────────────
HPARAM_GRID = {
    "mask_ratio":    [0.5, 0.75],
    "decoder_depth": [2, 4],
    "decoder_dim":   [64, 128],
    "epochs":        [200],       # start with 200, scale to 500 if promising
    "batch_size":    [32, 64],
}

LAYERS   = [1, 4, 8, 12]
K_VALUES = [5, 10, 20]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Main experiment loop ───────────────────────────────────
all_experiment_results = []

for mask_ratio, dec_depth, dec_dim, epochs, batch_size in product(
    HPARAM_GRID["mask_ratio"],
    HPARAM_GRID["decoder_depth"],
    HPARAM_GRID["decoder_dim"],
    HPARAM_GRID["epochs"],
    HPARAM_GRID["batch_size"],
):
    exp_name = (f"mae_mask{mask_ratio}_dec{dec_depth}x{dec_dim}"
                f"_ep{epochs}_bs{batch_size}")
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    # ── Build dataloaders ──────────────────────────────────
    pretrain_loader, train_loader, test_loader = get_dataloaders(batch_size)

    # ── Init model ────────────────────────────────────────
    model = MAE(
        in_channels    = config.IN_CHANNELS,
        img_h          = config.IMG_H,
        img_w          = config.IMG_W,
        patch_h        = config.PATCH_H,
        patch_w        = config.PATCH_W,
        encoder_dim    = config.D_MODEL,
        encoder_ff_dim = config.D_FF,
        encoder_heads  = config.N_HEADS,
        encoder_depth  = config.N_LAYERS,
        decoder_dim    = dec_dim,
        decoder_heads  = 2,
        decoder_depth  = dec_depth,
        mask_ratio     = mask_ratio,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-4, weight_decay=0.05
    )

    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # ── Pretraining loop ──────────────────────────────────
    best_loss = float("inf")
    for epoch in range(epochs):
        loss = train_one_epoch_mae(model, pretrain_loader, optimizer, device)
        scheduler.step()

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(),
                       f"checkpoints/{exp_name}_best.pth")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:03d}/{epochs} | "
                  f"Loss: {loss:.4f} | LR: {lr:.2e}")

    # ── KNN evaluation ────────────────────────────────────
    # In the experiment loop, replace knn_eval call with:
    print("\nRunning KNN evaluation...")
    for eval_name, eval_loader in [
        ("val",       val_loader),
        ("test_easy", test_easy_loader),
        ("test_hard", test_hard_loader),
    ]:
        print(f"\n-- {eval_name} --")
        knn_results = knn_eval(
            model, train_loader, eval_loader,
            layers=LAYERS, k_values=K_VALUES, device=device
        )
        exp_result[f"knn_{eval_name}"] = {
            str(k): float(v) for k, v in knn_results.items()
        }

    # ── Save results ──────────────────────────────────────
    exp_result = {
        "experiment":   exp_name,
        "mask_ratio":   mask_ratio,
        "decoder_depth": dec_depth,
        "decoder_dim":  dec_dim,
        "epochs":       epochs,
        "batch_size":   batch_size,
        "best_pretrain_loss": best_loss,
        "knn_results":  {str(k): float(v) for k, v in knn_results.items()},
    }
    all_experiment_results.append(exp_result)

    # Save checkpoint per experiment
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results",     exist_ok=True)
    with open(f"results/{exp_name}.json", "w") as f:
        json.dump(exp_result, f, indent=2)

    print(f"Best loss: {best_loss:.4f}")
    print(f"Saved: results/{exp_name}.json")


# ── Build final big table ──────────────────────────────────
print("\n\n" + "="*60)
print("FINAL RESULTS TABLE")
print("="*60)

# Best results per experiment
summary_rows = []
for res in all_experiment_results:
    best_knn = max(res["knn_results"].values())
    best_key = max(res["knn_results"], key=res["knn_results"].get)
    summary_rows.append({
        "Experiment":       res["experiment"],
        "mask_ratio":       res["mask_ratio"],
        "decoder_depth":    res["decoder_depth"],
        "decoder_dim":      res["decoder_dim"],
        "batch_size":       res["batch_size"],
        "pretrain_loss":    f"{res['best_pretrain_loss']:.4f}",
        "best_KNN_acc":     f"{best_knn:.3f}",
        "best_at":          best_key,
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("results/mae_experiment_summary.csv", index=False)
print("\nSaved: results/mae_experiment_summary.csv")
