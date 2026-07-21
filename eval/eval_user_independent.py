# eval_user_independent.py
# Evaluation with truly unseen users in test set

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

CHECKPOINTS = {
    "mae_341k_200":  ("checkpoints/mae_ep200_mask0.75_dec128_bs64_best.pth",  128),
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

    mean = train_feats.mean(0, keepdim=True)
    std  = train_feats.std(0,  keepdim=True) + 1e-8
    train_feats = (train_feats - mean) / std
    eval_feats  = (eval_feats  - mean) / std

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


def run_finetune(model, train_loader, eval_loader, device, epochs=100):
    head = nn.Linear(config.D_MODEL, 2).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.patch_embedding.parameters(), "lr": 1e-5},
        {"params": [model.encoder_pos_embed],          "lr": 1e-5},
        {"params": model.encoder_blocks.parameters(),  "lr": 1e-5},
        {"params": model.encoder_norm.parameters(),    "lr": 1e-5},
        {"params": head.parameters(),                  "lr": 1e-3},
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_labels = torch.cat([y for _, y in train_loader])
    counts  = torch.bincount(all_labels, minlength=2).float()
    weights = (1.0 / (counts + 1e-8))
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_acc   = 0.0
    no_improve = 0

    for epoch in range(epochs):
        model.train(); head.train()
        for x, y in train_loader:
            x, y    = x.to(device), y.to(device)
            tokens  = model.patch_embedding(x) + model.encoder_pos_embed
            encoded = model.encoder_blocks(tokens)
            encoded = model.encoder_norm(encoded)
            feat    = encoded.mean(dim=1)
            loss    = criterion(head(feat), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        model.eval(); head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y    = x.to(device), y.to(device)
                tokens  = model.patch_embedding(x) + model.encoder_pos_embed
                encoded = model.encoder_blocks(tokens)
                feat    = encoded.mean(dim=1)
                preds   = head(feat).argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)

        acc = correct / total
        if acc > best_acc:
            best_acc   = acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:03d} | Eval: {acc:.3f} | Best: {best_acc:.3f}")

    return best_acc


# ── Main ──────────────────────────────────────────────────
task      = "FallDetection"
label_map = {"Fall": 0, "Nonfall": 1}

train_df = pd.read_csv(f"{SPLITS_DIR}/{task}_user_train.csv")
val_df   = pd.read_csv(f"{SPLITS_DIR}/{task}_user_val.csv")
test_df  = pd.read_csv(f"{SPLITS_DIR}/{task}_user_test.csv")

print(f"User-independent FallDetection split:")
print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"  Train users: {sorted(train_df['user'].unique())}")
print(f"  Test users:  {sorted(test_df['user'].unique())}")
print(f"  Overlap:     {set(train_df['user'].unique()) & set(test_df['user'].unique())}")

train_loader = DataLoader(
    MultiTaskDataset(train_df, config.DATA_ROOT, task, label_map),
    batch_size=64, shuffle=True,  num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    MultiTaskDataset(val_df, config.DATA_ROOT, task, label_map),
    batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    MultiTaskDataset(test_df, config.DATA_ROOT, task, label_map),
    batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

all_results = {}

for ckpt_name, (ckpt_path, dec_dim) in CHECKPOINTS.items():
    if not os.path.exists(ckpt_path):
        print(f"\nSkip {ckpt_name}: not found")
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

    ckpt_results = {}

    for eval_name, eval_loader in [("val", val_loader), ("test", test_loader)]:
        print(f"\n  -- {eval_name} (unseen users) --")

        # KNN
        knn_res  = knn_eval(model, train_loader, eval_loader,
                            layers=LAYERS, k_values=K_VALUES, device=device)
        best_knn = max(float(v) for v in knn_res.values())
        best_k   = max(knn_res, key=lambda k: float(knn_res[k]))
        print(f"  Best KNN: {best_knn:.3f} (at {best_k})")

        # LP
        best_lp = 0.0
        for layer in LAYERS:
            acc = run_linear_probe(model, train_loader, eval_loader,
                                   layer, 2, device)
            print(f"  LP L{layer}: {acc:.3f}")
            if acc > best_lp:
                best_lp = acc

        # Fine-tune (reload fresh weights)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Fine-tuning...")
        ft_acc = run_finetune(model, train_loader, eval_loader, device)
        print(f"  FT: {ft_acc:.3f}")

        ckpt_results[eval_name] = {
            "best_knn": round(best_knn, 4),
            "best_lp":  round(best_lp,  4),
            "finetune": round(ft_acc,    4),
        }
        print(f"  → KNN={best_knn:.3f}  LP={best_lp:.3f}  FT={ft_acc:.3f}")

    all_results[ckpt_name] = ckpt_results

# Save
with open("results/user_independent_eval.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Summary
print("\n\n" + "="*65)
print("USER-INDEPENDENT EVALUATION (truly unseen users)")
print("="*65)
print(f"{'Checkpoint':<18} {'Split':<8} {'KNN':>6} {'LP':>6} {'FT':>6}")
print("-"*65)
for ckpt, splits in all_results.items():
    for split, m in splits.items():
        print(f"{ckpt:<18} {split:<8} "
              f"{m['best_knn']:>6.3f} {m['best_lp']:>6.3f} {m['finetune']:>6.3f}")

print("\nSaved: results/user_independent_eval.json")
