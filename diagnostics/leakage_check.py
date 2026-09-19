# diagnostics/leakage_check.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

def run_leakage_diagnostic(X_train, X_splits, threshold=0.05, save_path=None):
    """
    X_splits: dict, e.g., {'test_id': X_test_id, 'test_cross_env': X_test_cross_env}
    """
    nn = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1)
    nn.fit(X_train)
    
    results = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for split_name, X_data in X_splits.items():
        dists, _ = nn.kneighbors(X_data)
        dists = dists.ravel()
        leak_ratio = np.mean(dists < threshold)
        results[split_name] = {'distances': dists, 'leak_ratio': leak_ratio}
        
        print(f"[{split_name}] Ratio with cosine dist < {threshold}: {leak_ratio:.4f}")
        
        sns.histplot(
            dists, kde=True, stat="density", bins=50,
            alpha=0.35, label=f"{split_name} (leak: {leak_ratio*100:.1f}%)", ax=ax
        )
        
    ax.set_xlabel("Nearest Neighbor Cosine Distance to Train")
    ax.set_ylabel("Density")
    ax.set_title("Data Leakage Diagnostic: NN Distance Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return results