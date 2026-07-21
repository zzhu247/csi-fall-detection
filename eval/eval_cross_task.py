# eval_cross_task.py
# Foundation model evaluation:
# Train KNN/LP on ALL tasks combined, evaluate per task
# This is the true test of foundation model generalization

import os, json, torch, pandas as pd, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.linear_model import LogisticRegression
from models.mae import MAE
from data.dataset import MultiTaskDataset, load_and_normalize_csi
import config

config.N_LAYERS = 12
device = torch.device("cuda")

SPLITS_DIR     = os.path.expanduser("~/data/splits")
COMBINED_TRAIN = "/home/zhuzih19/data/combined_train_clean.csv"

EVAL_TASKS = {
    "FallDetection":           {"num_classes": 2},
    "MotionSourceRecognition": {"num_classes": 4},
}

ALL_PRETRAIN_TASKS = [
    "FallDetection",
    "MotionSourceRecognition",
    "BreathingDetection",
    "Localization",
    "HumanActivityRecognition",
    "HumanIdentification",
    "ProximityRecognition",
]

CHECKPOINTS = {
    "mae_341k_200": ("checkpoints/mae_ep200_mask0.75_dec128_bs64_best.pth", 128),
    "mae_341k_300": ("checkpoints/mae_ep300_mask0.75_dec128_bs64_best.pth", 128),
    "mae_341k_500": ("checkpoints/mae_ep500_mask0.75_dec128_bs64_best.pth", 128),
}

LAYERS   = [1, 4, 8, 12]
K_VALUES = [5, 10, 20]


# ── Dataset ───────────────────────────────────────────────
class CombinedLabeledDataset(Dataset):
    """
    Multi-task dataset with global label = task_label string.
    Used for extracting features across all tasks.
    """
    def __init__(self, meta_df, data_root):
        self.meta      = meta_df.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row     = self.meta.iloc[idx]
        
        # Use pre-resolved h5_path if available, else resolve on the fly
        if "h5_path" in row.index and pd.notna(row.get("h5_path", None)):
            h5_path = row["h5_path"]
        else:
            task    = row["task"]
            h5_path = os.path.join(
                self.data_root, task, row["file_path"].lstrip("./")
            )
            # Handle Multitask folder structure
            if "../../sub_Human_h5" in row["file_path"]:
                rel     = row["file_path"].replace("../../", "")
                h5_path = os.path.join(self.data_root, "Multitask", rel)
        
        csi     = load_and_normalize_csi(h5_path)
        csi     = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi     = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)
        return csi, f"{row['task']}_{row['label']}"


def collate_fn(batch):
    """Custom collate to handle string labels."""
    csi    = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return csi, labels


# ── Feature extraction ────────────────────────────────────
def extract_all_features(model, loader, layers, device):
    """
    Extract features from all samples.
    Returns dict of {layer: np.array [N, D]}, list of string labels.
    """
    model.eval()
    all_feats  = {l: [] for l in layers}
    all_labels = []

    with torch.no_grad():
        for csi, labels in loader:
            csi = csi.to(device)
            layer_embs = model.extract_layer_embeddings(csi, layers)
            for l in layers:
                all_feats[l].append(layer_embs[l].cpu().numpy())
            all_labels.extend(labels)

    for l in layers:
        all_feats[l] = np.concatenate(all_feats[l], axis=0)

    return all_feats, all_labels


# ── Cross-task KNN evaluation ─────────────────────────────
def cross_task_knn_eval(train_feats, train_labels, test_feats, test_labels,
                        eval_task, k_values, layer):
    """
    KNN trained on ALL tasks, evaluated only on eval_task samples.
    train_labels / test_labels are strings: "FallDetection_Fall" etc.
    """
    # Filter test to only eval_task
    test_mask  = np.array([l.startswith(eval_task) for l in test_labels])
    X_test     = test_feats[test_mask]
    y_test_str = [l for l in test_labels if l.startswith(eval_task)]

    if len(X_test) == 0:
        return {}

    # Encode labels: only task-specific labels
    task_labels_set = sorted(set(y_test_str))
    label2idx       = {l: i for i, l in enumerate(task_labels_set)}
    y_test          = np.array([label2idx[l] for l in y_test_str])

    # For training: use ALL task labels
    # KNN will naturally find nearest neighbors from any task
    y_train_str = train_labels
    all_label_set = sorted(set(y_train_str))
    all_label2idx = {l: i for i, l in enumerate(all_label_set)}
    y_train = np.array([all_label2idx[l] for l in y_train_str])

    # But evaluation: remap predictions to task-specific labels
    # Only keep training samples whose labels appear in this task
    task_train_mask = np.array([l.startswith(eval_task) for l in y_train_str])

    results = {}
    for k in k_values:
        # Normalize
        X_tr_norm = normalize(train_feats)
        X_te_norm = normalize(X_test)

        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(X_tr_norm, y_train_str)   # train on all tasks with string labels
        preds = knn.predict(X_te_norm)    # predict string labels

        # Accuracy: correct if prediction matches eval_task label
        correct = sum(p == t for p, t in zip(preds, y_test_str))
        acc     = correct / len(y_test_str)
        results[k] = acc

    return results


# ── Cross-task LP evaluation ──────────────────────────────
def cross_task_lp_eval(train_feats, train_labels, test_feats, test_labels,
                       eval_task, device, epochs=100):
    """
    Linear probe trained on ALL tasks (multi-task head),
    evaluated only on eval_task samples.
    """
    # Encode all labels
    all_label_set = sorted(set(train_labels + test_labels))
    label2idx     = {l: i for i, l in enumerate(all_label_set)}
    num_classes   = len(all_label_set)

    y_train = np.array([label2idx[l] for l in train_labels])
    y_test  = np.array([label2idx[l] for l in test_labels])

    # Normalize features
    mean       = train_feats.mean(0, keepdims=True)
    std        = train_feats.std(0,  keepdims=True) + 1e-8
    X_train_n  = (train_feats - mean) / std
    X_test_n   = (test_feats  - mean) / std

    # Filter test to only eval_task
    test_mask = np.array([l.startswith(eval_task) for l in test_labels])
    X_te      = torch.tensor(X_test_n[test_mask], dtype=torch.float32)
    y_te_str  = [l for l in test_labels if l.startswith(eval_task)]
    y_te      = torch.tensor(y_test[test_mask], dtype=torch.long)

    # Train linear head on ALL tasks
    X_tr  = torch.tensor(X_train_n, dtype=torch.float32)
    y_tr  = torch.tensor(y_train,   dtype=torch.long)

    # Weighted loss
    class_counts = torch.bincount(y_tr, minlength=num_classes).float()
    weights  = 1.0 / (class_counts + 1e-8)
    weights  = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    head    = nn.Linear(X_tr.shape[1], num_classes).to(device)
    optim   = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    best_acc   = 0.0
    no_improve = 0

    for ep in range(epochs):
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(head(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

        if (ep + 1) % 10 == 0:
            head.eval()
            with torch.no_grad():
                preds = head(X_te.to(device)).argmax(1).cpu()
            acc = (preds == y_te).float().mean().item()
            if acc > best_acc:
                best_acc   = acc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 3:   # early stop after 3 checks (30 epochs)
                break

    return best_acc


# ── Main ──────────────────────────────────────────────────
all_results = {}

combined_train = pd.read_csv(COMBINED_TRAIN)
print(f"Combined train: {len(combined_train)} samples")
print(f"Tasks: {combined_train['task'].value_counts().to_dict()}")

for ckpt_name, (ckpt_path, dec_dim) in CHECKPOINTS.items():
    if not os.path.exists(ckpt_path):
        print(f"\nSkip {ckpt_name}: not found")
        continue

    print(f"\n{'='*65}")
    print(f"Checkpoint: {ckpt_name}")

    model = MAE(
        in_channels=1, img_h=232, img_w=500, patch_h=8, patch_w=25,
        encoder_dim=128, encoder_ff_dim=512, encoder_heads=4, encoder_depth=12,
        decoder_dim=dec_dim, decoder_heads=2, decoder_depth=2,
        mask_ratio=0.75
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ── Build combined train loader ────────────────────────
    train_loader = DataLoader(
        CombinedLabeledDataset(combined_train, config.DATA_ROOT),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=collate_fn
    )

    # ── Extract train features (all tasks) ────────────────
    print("\nExtracting train features (all tasks)...")
    train_feats_all, train_labels_all = extract_all_features(
        model, train_loader, LAYERS, device
    )
    print(f"Train features shape: {train_feats_all[LAYERS[0]].shape}")
    print(f"Unique train labels: {len(set(train_labels_all))}")

    all_results[ckpt_name] = {}

    for task in EVAL_TASKS:
        print(f"\n  ── {task} ──")

        # Build test loader
        test_dfs = []
        for split in ["test_easy", "test_hard"]:
            path = os.path.join(SPLITS_DIR, f"{task}_{split}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                if len(df) > 0:
                    test_dfs.append(df)

        if not test_dfs:
            print(f"  No test splits found")
            continue

        test_merged = pd.concat(test_dfs, ignore_index=True)
        print(f"  test: {len(test_merged)} samples (easy+hard merged)")

        label_map = {l: i for i, l in
                     enumerate(sorted(test_merged["label"].unique(), key=str))}

        test_loader = DataLoader(
            CombinedLabeledDataset(test_merged, config.DATA_ROOT),
            batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True,
            collate_fn=collate_fn
        )

        print("  Extracting test features...")
        test_feats_all, test_labels_all = extract_all_features(
            model, test_loader, LAYERS, device
        )

        task_results = {}

        for layer in LAYERS:
            print(f"\n  Layer {layer}:")

            # KNN
            knn_res  = cross_task_knn_eval(
                train_feats_all[layer], train_labels_all,
                test_feats_all[layer],  test_labels_all,
                task, K_VALUES, layer
            )
            best_knn = max(knn_res.values()) if knn_res else 0
            print(f"    KNN: {' '.join(f'k={k}:{v:.3f}' for k,v in knn_res.items())}")
            print(f"    Best KNN: {best_knn:.3f}")

            # LP
            lp_acc = cross_task_lp_eval(
                train_feats_all[layer], train_labels_all,
                test_feats_all[layer],  test_labels_all,
                task, device
            )
            print(f"    LP:  {lp_acc:.3f}")

            task_results[f"layer_{layer}"] = {
                "knn": {str(k): round(v, 4) for k, v in knn_res.items()},
                "best_knn": round(best_knn, 4),
                "lp":       round(lp_acc, 4),
            }

        # Best across layers
        best_knn_overall = max(
            v["best_knn"] for v in task_results.values()
        )
        best_lp_overall = max(
            v["lp"] for v in task_results.values()
        )
        task_results["best_knn"] = round(best_knn_overall, 4)
        task_results["best_lp"]  = round(best_lp_overall,  4)

        print(f"\n  → Best KNN: {best_knn_overall:.3f}  "
              f"Best LP: {best_lp_overall:.3f}")

        all_results[ckpt_name][task] = task_results

# ── Save ──────────────────────────────────────────────────
with open("results/cross_task_eval.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ── Summary ───────────────────────────────────────────────
print("\n\n" + "="*65)
print("CROSS-TASK EVALUATION SUMMARY")
print("(KNN/LP trained on ALL 7 tasks, evaluated per task)")
print("="*65)
print(f"{'Checkpoint':<18} {'Task':<25} {'KNN':>6} {'LP':>6}")
print("-"*65)
for ckpt, tasks in all_results.items():
    for task, res in tasks.items():
        print(f"{ckpt:<18} {task:<25} "
              f"{res['best_knn']:>6.3f} {res['best_lp']:>6.3f}")

print("\nSaved: results/cross_task_eval.json")