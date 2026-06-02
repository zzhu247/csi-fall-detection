# eval/knn_probe.py

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import pandas as pd


def extract_features(model, loader, layers, device):
    """
    Extract layer embeddings for all samples in loader.
    Returns features dict and labels array.
    """
    model.eval()
    all_feats  = {l: [] for l in layers}
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            # Support both (csi, label) and csi-only loaders
            if isinstance(batch, (list, tuple)):
                csi, label = batch
                all_labels.append(label.numpy())
            else:
                csi = batch

            csi = csi.to(device)
            layer_embs = model.extract_layer_embeddings(csi, layers)
            for l in layers:
                all_feats[l].append(layer_embs[l].cpu().numpy())

    for l in layers:
        all_feats[l] = np.concatenate(all_feats[l], axis=0)

    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    return all_feats, all_labels


def knn_eval(model, train_loader, test_loader,
             layers=[1, 4, 8, 12],
             k_values=[5, 10, 20],
             device="cpu"):
    """
    Run KNN probe on each layer x each k value.
    Returns results dict: {(layer, k): accuracy}
    """
    print("Extracting train features...")
    train_feats, train_labels = extract_features(model, train_loader, layers, device)

    print("Extracting test features...")
    test_feats, test_labels = extract_features(model, test_loader, layers, device)

    results = {}
    for l in layers:
        for k in k_values:
            X_train = normalize(train_feats[l])
            X_test  = normalize(test_feats[l])

            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
            knn.fit(X_train, train_labels)
            acc = knn.score(X_test, test_labels)

            results[(l, k)] = acc
            print(f"  Layer {l:2d}  k={k:2d}  →  {acc:.3f}")

    return results


def build_table(all_results, layers, k_values, batch_sizes):
    """
    Build a DataFrame table from results across batch sizes.

    all_results: {batch_size: {(layer, k): acc}}
    """
    rows = []
    for l in layers:
        for k in k_values:
            row = {"Layer": f"L{l}", "K": k}
            for bs in batch_sizes:
                row[f"batch={bs}"] = f"{all_results[bs].get((l,k), 0):.3f}"
            rows.append(row)

    df = pd.DataFrame(rows)
    return df
