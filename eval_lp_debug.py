# eval_lp_debug.py
# Systematically diagnose why LP < KNN

import torch, pandas as pd, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from models.mae import MAE
from data.dataset import MultiTaskDataset
import config, os

config.N_LAYERS = 12
device = torch.device("cuda")

SPLITS_DIR = os.path.expanduser("~/data/splits")

# Load model
model = MAE(
    in_channels=1, img_h=232, img_w=500, patch_h=8, patch_w=25,
    encoder_dim=128, encoder_ff_dim=512, encoder_heads=4, encoder_depth=12,
    decoder_dim=128, decoder_heads=2, decoder_depth=2, mask_ratio=0.75
).to(device)
model.load_state_dict(torch.load(
    "checkpoints/mae_ep500_mask0.75_dec128_bs64_best.pth",
    map_location=device
))
model.eval()

# Load data
task = "FallDetection"
train_df = pd.read_csv(f"{SPLITS_DIR}/{task}_train.csv")
test_dfs = [pd.read_csv(f"{SPLITS_DIR}/{task}_{s}.csv")
            for s in ["test_easy", "test_hard"]]
test_df  = pd.concat(test_dfs, ignore_index=True)

label_map = {l: i for i, l in
             enumerate(sorted(train_df["label"].unique(), key=str))}

train_loader = DataLoader(
    MultiTaskDataset(train_df, config.DATA_ROOT, task, label_map),
    batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    MultiTaskDataset(test_df, config.DATA_ROOT, task, label_map),
    batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

# Extract features at best layer (L12)
def get_feats(loader, layer=12):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model.extract_layer_embeddings(x, [layer])
            feats.append(emb[layer].cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)

print("Extracting features at Layer 12...")
X_train, y_train = get_feats(train_loader, layer=12)
X_test,  y_test  = get_feats(test_loader,  layer=12)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train label dist: {np.bincount(y_train)}")
print(f"Test  label dist: {np.bincount(y_test)}")

# ── Test 1: KNN (baseline) ────────────────────────────────
X_tr_norm = normalize(X_train)
X_te_norm = normalize(X_test)

for k in [5, 10, 20]:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(X_tr_norm, y_train)
    acc = knn.score(X_te_norm, y_test)
    print(f"KNN k={k:2d}: {acc:.3f}")

# ── Test 2: sklearn LogisticRegression (no training instability) ──
print("\nSklearn LogisticRegression (C=1.0):")
for norm in [True, False]:
    X_tr = normalize(X_train) if norm else X_train
    X_te = normalize(X_test)  if norm else X_test
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    lr.fit(X_tr, y_train)
    acc = lr.score(X_te, y_test)
    print(f"  normalized={norm}: {acc:.3f}")

# ── Test 3: sklearn LR with different C ──────────────────
print("\nSklearn LR with different C (normalized):")
X_tr_norm = normalize(X_train)
X_te_norm = normalize(X_test)
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    lr = LogisticRegression(max_iter=1000, C=C, class_weight="balanced")
    lr.fit(X_tr_norm, y_train)
    acc = lr.score(X_te_norm, y_test)
    print(f"  C={C:6.2f}: {acc:.3f}")

# ── Test 4: PyTorch LP with different lr and epochs ──────
print("\nPyTorch LP ablation (normalized features):")
X_tr_t = torch.tensor(normalize(X_train), dtype=torch.float32)
X_te_t = torch.tensor(normalize(X_test),  dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.long)
y_te_t = torch.tensor(y_test,  dtype=torch.long)

dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

for lr_val in [1e-4, 1e-3, 1e-2]:
    for n_ep in [100, 200, 500]:
        head = nn.Linear(X_tr_t.shape[1], 2).to(device)
        opt  = torch.optim.Adam(head.parameters(), lr=lr_val)
        crit = nn.CrossEntropyLoss()

        for _ in range(n_ep):
            head.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = crit(head(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            preds = head(X_te_t.to(device)).argmax(1).cpu()
        acc = (preds == y_te_t).float().mean().item()
        print(f"  lr={lr_val:.0e} ep={n_ep:3d}: {acc:.3f}")
