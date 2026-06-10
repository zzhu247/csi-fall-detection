# train_mae_run.py
import os, sys, time, json, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from models.mae import MAE
from data.dataset import CSIPretrainDatasetV2, CombinedDataset, MultiTaskDataset
from eval.knn_probe import knn_eval
import config

config.N_LAYERS = 12

# Parse args
num_epochs  = int(sys.argv[1])     # 200, 300, 500
mask_ratio  = float(sys.argv[2])   # 0.75
decoder_dim = int(sys.argv[3])     # 128
batch_size  = int(sys.argv[4])     # 64

exp_name = f"mae_ep{num_epochs}_mask{mask_ratio}_dec{decoder_dim}_bs{batch_size}"
print(f"Starting: {exp_name}")

device = torch.device("cuda")

# ── Dataloaders ───────────────────────────────────────────
pretrain_df = pd.read_csv("/home/zhuzih19/data/combined_all_clean.csv")
train_df    = pd.read_csv("/home/zhuzih19/data/combined_train_clean.csv")
val_df      = pd.read_csv("/home/zhuzih19/data/combined_val_clean.csv")
test_df     = pd.read_csv("/home/zhuzih19/data/combined_test_clean.csv")

# Build label map from train (consistent across splits)
label_map = {}
for task in train_df["task"].unique():
    task_labels = sorted(train_df[train_df["task"]==task]["label"].unique(), key=str)
    for l in task_labels:
        key = f"{task}_{l}"
        if key not in label_map:
            label_map[key] = len(label_map)

# Add task-specific label to df
def add_global_label(df, label_map):
    df = df.copy()
    df["global_label"] = df.apply(
        lambda r: label_map.get(f"{r['task']}_{r['label']}", 0), axis=1
    )
    return df

train_df = add_global_label(train_df, label_map)
val_df   = add_global_label(val_df,   label_map)
test_df  = add_global_label(test_df,  label_map)

pretrain_loader = DataLoader(
    CSIPretrainDatasetV2(pretrain_df, config.DATA_ROOT),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
train_loader = DataLoader(
    CombinedDataset(train_df, config.DATA_ROOT),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    CombinedDataset(val_df, config.DATA_ROOT),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    CombinedDataset(test_df, config.DATA_ROOT),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Pretrain: {len(pretrain_df)} | Train: {len(train_df)} | "
      f"Val: {len(val_df)} | Test: {len(test_df)}")

# ── Model ─────────────────────────────────────────────────
model = MAE(
    in_channels=1, img_h=232, img_w=500, patch_h=8, patch_w=25,
    encoder_dim=128, encoder_ff_dim=512, encoder_heads=4, encoder_depth=12,
    decoder_dim=decoder_dim, decoder_heads=2, decoder_depth=2,
    mask_ratio=mask_ratio
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ── Pretraining loop ──────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",     exist_ok=True)

best_loss = float("inf")
loss_log  = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for csi in pretrain_loader:
        csi = csi.to(device)
        optimizer.zero_grad()
        loss, _, _ = model(csi)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(pretrain_loader)
    scheduler.step()
    loss_log.append({"epoch": epoch+1, "loss": avg_loss})

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(),
                   f"checkpoints/{exp_name}_best.pth")

    if (epoch+1) % 50 == 0 or epoch == 0:
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:03d}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {lr:.2e}")
        sys.stdout.flush()

# ── KNN evaluation ────────────────────────────────────────
print("\nRunning KNN evaluation...")
results = {
    "experiment": exp_name,
    "best_loss":  best_loss,
    "knn":        {},
    "loss_log":   loss_log
}

for eval_name, eval_loader in [
    ("val",  val_loader),
    ("test", test_loader),
]:
    print(f"\n-- KNN on {eval_name} --")
    knn_res = knn_eval(
        model, train_loader, eval_loader,
        layers=[1, 4, 8, 12], k_values=[5, 10, 20],
        device=device
    )
    results["knn"][eval_name] = {
        str(k): float(v) for k, v in knn_res.items()
    }

# ── Linear Probe evaluation ───────────────────────────────
print("\nRunning Linear Probe evaluation...")
results["linear_probe"] = {}

def get_feats(model, loader, layer, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch
                y = torch.zeros(x.shape[0], dtype=torch.long)
            x = x.to(device)
            emb = model.extract_layer_embeddings(x, [layer])
            feats.append(emb[layer].cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def run_linear_probe(model, train_loader, eval_loader, layer, device, epochs=100):
    train_feats, train_labels = get_feats(model, train_loader, layer, device)
    eval_feats,  eval_labels  = get_feats(model, eval_loader,  layer, device)

    # Normalize
    mean = train_feats.mean(0, keepdim=True)
    std  = train_feats.std(0,  keepdim=True) + 1e-8
    train_feats = (train_feats - mean) / std
    eval_feats  = (eval_feats  - mean) / std

    num_classes = int(train_labels.max().item()) + 1

    # Weighted loss
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    weights  = 1.0 / (class_counts + 1e-8)
    weights  = weights / weights.sum()
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

for eval_name, eval_loader in [
    ("val",  val_loader),
    ("test", test_loader),
]:
    results["linear_probe"][eval_name] = {}
    print(f"\n-- Linear Probe on {eval_name} --")
    for layer in [1, 4, 8, 12]:
        acc = run_linear_probe(model, train_loader, eval_loader,
                               layer, device)
        results["linear_probe"][eval_name][f"layer_{layer}"] = round(acc, 4)
        print(f"  Layer {layer:2d}  →  {acc:.3f}")

# ── Save results ──────────────────────────────────────────
with open(f"results/{exp_name}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Saved: results/{exp_name}.json")
print(f"Best pretrain loss: {best_loss:.4f}")