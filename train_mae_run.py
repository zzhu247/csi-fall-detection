# train_mae_run.py
import os, sys, time, json, torch, pandas as pd
from torch.utils.data import DataLoader
from models.mae import MAE
from data.dataset import CSIFallDataset, CSIPretrainDatasetV2
from eval.knn_probe import knn_eval
import config

config.N_LAYERS = 12

# Parse args
num_epochs  = int(sys.argv[1])   # 200, 300, or 500
mask_ratio  = float(sys.argv[2]) # 0.75
decoder_dim = int(sys.argv[3])   # 64 or 128
batch_size  = int(sys.argv[4])   # 32 or 64

exp_name = f"mae_ep{num_epochs}_mask{mask_ratio}_dec{decoder_dim}_bs{batch_size}"
print(f"Starting: {exp_name}")

device = torch.device("cuda")

# Dataloaders
pretrain_df  = pd.read_csv("/home/zhuzih19/data/pretrain_combined.csv")
train_df     = pd.read_csv("/home/zhuzih19/data/train.csv")
val_df       = pd.read_csv("/home/zhuzih19/data/val.csv")
test_easy_df = pd.read_csv("/home/zhuzih19/data/test_easy.csv")
test_hard_df = pd.read_csv("/home/zhuzih19/data/test_hard.csv")

pretrain_loader  = DataLoader(CSIPretrainDatasetV2(pretrain_df, config.DATA_ROOT),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
train_loader     = DataLoader(CSIFallDataset(train_df,     config.DATA_ROOT),
                              batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader       = DataLoader(CSIFallDataset(val_df,       config.DATA_ROOT),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_easy_loader = DataLoader(CSIFallDataset(test_easy_df, config.DATA_ROOT),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_hard_loader = DataLoader(CSIFallDataset(test_hard_df, config.DATA_ROOT),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Model
model = MAE(
    in_channels=1, img_h=232, img_w=500, patch_h=8, patch_w=25,
    encoder_dim=128, encoder_ff_dim=512, encoder_heads=4, encoder_depth=12,
    decoder_dim=decoder_dim, decoder_heads=2, decoder_depth=2,
    mask_ratio=mask_ratio
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",     exist_ok=True)

best_loss = float("inf")
log = []

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
    log.append({"epoch": epoch+1, "loss": avg_loss})

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"checkpoints/{exp_name}_best.pth")

    if (epoch+1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{num_epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
        sys.stdout.flush()

# KNN evaluation
print("\nRunning KNN evaluation...")
results = {"experiment": exp_name, "best_loss": best_loss, "knn": {}}

for eval_name, eval_loader in [
    ("val",       val_loader),
    ("test_easy", test_easy_loader),
    ("test_hard", test_hard_loader),
]:
    print(f"\n-- KNN on {eval_name} --")
    knn_res = knn_eval(model, train_loader, eval_loader,
                       layers=[1,4,8,12], k_values=[5,10,20], device=device)
    results["knn"][eval_name] = {str(k): float(v) for k, v in knn_res.items()}

results["loss_log"] = log

with open(f"results/{exp_name}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Results saved to results/{exp_name}.json")

# ── Linear Probe ──────────────────────────────────────────
from models.vit import ViT
import torch.nn as nn

print("\nRunning Linear Probe evaluation...")

def linear_probe(encoder_model, train_loader, eval_loader, 
                 layer_idx, device, epochs=20):
    """
    Freeze encoder, train only a linear head on top of layer_idx embedding.
    """
    # Extract features using specified layer
    def get_feats(loader):
        encoder_model.eval()
        feats, labels = [], []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                    labels.append(y)
                else:
                    x = batch
                x = x.to(device)
                emb = encoder_model.extract_layer_embeddings(x, [layer_idx])
                feats.append(emb[layer_idx].cpu())
        return torch.cat(feats), torch.cat(labels)

    train_feats, train_labels = get_feats(train_loader)
    eval_feats,  eval_labels  = get_feats(eval_loader)

    # Train linear head
    head = nn.Linear(train_feats.shape[1], 2).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)

    for _ in range(epochs):
        head.train()
        logits = head(train_feats)
        loss = criterion(logits, train_labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Evaluate
    head.eval()
    eval_feats = eval_feats.to(device)
    with torch.no_grad():
        preds = head(eval_feats).argmax(dim=1).cpu()
    acc = (preds == eval_labels).float().mean().item()
    return acc


# Run linear probe on best layers [1, 4, 8, 12]
results["linear_probe"] = {}
for eval_name, eval_loader in [
    ("val",       val_loader),
    ("test_easy", test_easy_loader),
    ("test_hard", test_hard_loader),
]:
    results["linear_probe"][eval_name] = {}
    print(f"\n-- Linear Probe on {eval_name} --")
    for layer in [1, 4, 8, 12]:
        acc = linear_probe(model, train_loader, eval_loader,
                          layer_idx=layer, device=device)
        results["linear_probe"][eval_name][f"layer_{layer}"] = acc
        print(f"  Layer {layer:2d}  →  {acc:.3f}")
