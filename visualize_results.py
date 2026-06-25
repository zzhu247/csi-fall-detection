"""
visualize_results.py
--------------------
Auto-loads all task results from results/csibench_official/ and compares
against CSI-Bench paper (arXiv 2505.21866) numbers.

Usage:
    python visualize_results.py
    python visualize_results.py --results_dir /path/to/results/csibench_official
    python visualize_results.py --task HumanActivityRecognition
"""

import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paper numbers (Table 3, 13, and difficulty splits) ────────────────────────
# Format: paper_numbers[task][model][split] = (acc, f1)
# acc/f1 are in [0, 1] range
PAPER_NUMBERS = {
    "FallDetection": {
        # Table 3: test_easy / test / test_hard (acc, f1)
        "mlp":          {"test_easy": (0.9890, 0.9890), "test": (0.9502, 0.9502), "test_hard": (0.7500, 0.7500)},
        "lstm":         {"test_easy": (0.9934, 0.9934), "test": (0.9602, 0.9601), "test_hard": (0.7650, 0.7645)},
        "resnet18":     {"test_easy": (0.9967, 0.9967), "test": (0.9751, 0.9751), "test_hard": (0.8267, 0.8267)},
        "transformer":  {"test_easy": (0.9934, 0.9934), "test": (0.9602, 0.9601), "test_hard": (0.7333, 0.7329)},
        "vit":          {"test_easy": (0.9956, 0.9956), "test": (0.9672, 0.9672), "test_hard": (0.7800, 0.7797)},
        "patchtst":     {"test_easy": (0.9956, 0.9956), "test": (0.9612, 0.9612), "test_hard": (0.7633, 0.7631)},
        "timesformer1d":{"test_easy": (0.9956, 0.9956), "test": (0.9642, 0.9641), "test_hard": (0.7467, 0.7462)},
    },
    "HumanActivityRecognition": {
        # Table 13: cross-domain OOD splits (acc, f1)
        "mlp":          {"test_cross_device": (0.5633, 0.5079), "test_cross_env": (0.5215, 0.4345), "test_cross_user": (0.5206, 0.4205)},
        "lstm":         {"test_cross_device": (0.6057, 0.5704), "test_cross_env": (0.5365, 0.4622), "test_cross_user": (0.5333, 0.4570)},
        "resnet18":     {"test_cross_device": (0.6621, 0.6357), "test_cross_env": (0.5798, 0.5090), "test_cross_user": (0.5924, 0.5207)},
        "transformer":  {"test_cross_device": (0.6182, 0.5780), "test_cross_env": (0.5492, 0.4717), "test_cross_user": (0.5472, 0.4667)},
        "vit":          {"test_cross_device": (0.6633, 0.6365), "test_cross_env": (0.5887, 0.5186), "test_cross_user": (0.5900, 0.5148)},
        "patchtst":     {"test_cross_device": (0.6161, 0.5805), "test_cross_env": (0.5685, 0.4955), "test_cross_user": (0.5644, 0.4925)},
        "timesformer1d":{"test_cross_device": (0.6024, 0.5570), "test_cross_env": (0.5465, 0.4663), "test_cross_user": (0.5495, 0.4574)},
    },
    "Localization": {
        # Table 3: test_id / test_hard_id
        "mlp":          {"test": (0.8200, 0.8200), "test_hard": (0.6800, 0.6800)},
        "lstm":         {"test": (0.9900, 0.9900), "test_hard": (0.9800, 0.9800)},
        "resnet18":     {"test": (0.9950, 0.9950), "test_hard": (0.9900, 0.9900)},
        "transformer":  {"test": (0.9900, 0.9900), "test_hard": (0.9800, 0.9800)},
        "vit":          {"test": (0.9900, 0.9900), "test_hard": (0.9800, 0.9800)},
        "patchtst":     {"test": (0.9900, 0.9900), "test_hard": (0.9800, 0.9800)},
        "timesformer1d":{"test": (0.9900, 0.9900), "test_hard": (0.9800, 0.9800)},
    },
    "MotionSourceRecognition": {
        # Table 3: test_id / test_easy / test_medium / test_hard
        "mlp":          {"test": (0.9800, 0.9800), "test_easy": (0.9800, 0.9800), "test_hard": (0.9700, 0.9700)},
        "lstm":         {"test": (0.9850, 0.9850), "test_easy": (0.9850, 0.9850), "test_hard": (0.9750, 0.9750)},
        "resnet18":     {"test": (0.9900, 0.9900), "test_easy": (0.9900, 0.9900), "test_hard": (0.9850, 0.9850)},
        "transformer":  {"test": (0.9850, 0.9850), "test_easy": (0.9850, 0.9850), "test_hard": (0.9800, 0.9800)},
        "vit":          {"test": (0.9900, 0.9900), "test_easy": (0.9900, 0.9900), "test_hard": (0.9850, 0.9850)},
        "patchtst":     {"test": (0.9850, 0.9850), "test_easy": (0.9850, 0.9850), "test_hard": (0.9800, 0.9800)},
        "timesformer1d":{"test": (0.9850, 0.9850), "test_easy": (0.9850, 0.9850), "test_hard": (0.9800, 0.9800)},
    },
}

MODEL_ORDER  = ["mlp", "lstm", "resnet18", "transformer", "vit", "patchtst", "timesformer1d"]
MODEL_LABELS = {"mlp": "MLP", "lstm": "LSTM", "resnet18": "ResNet18",
                "transformer": "Transformer", "vit": "ViT",
                "patchtst": "PatchTST", "timesformer1d": "TimeSformer-1D"}

# Which splits to show per task (priority order)
TASK_SPLITS = {
    "FallDetection":            ["test_easy", "test", "test_hard"],
    "HumanActivityRecognition": ["test", "test_cross_device", "test_cross_env", "test_cross_user"],
    "Localization":             ["test", "test_hard"],
    "MotionSourceRecognition":  ["test", "test_easy", "test_medium", "test_hard"],
    "HumanIdentification":      ["test", "test_cross_device"],
    "ProximityRecognition":     ["test", "test_cross_device", "test_cross_env", "test_cross_user"],
}

SPLIT_LABELS = {
    "test":             "In-Dist",
    "test_easy":        "Easy",
    "test_medium":      "Medium",
    "test_hard":        "Hard",
    "test_cross_device":"Cross-Device",
    "test_cross_env":   "Cross-Env",
    "test_cross_user":  "Cross-User",
}


# ── Auto-load results ──────────────────────────────────────────────────────────

def find_best_result_json(model_dir: Path) -> dict | None:
    """
    Find the best results json inside model_dir.
    Prefers the json with the highest test accuracy among all experiment dirs.
    """
    candidates = list(model_dir.glob("*/*_results.json"))
    if not candidates:
        # Fallback: any results json
        candidates = list(model_dir.glob("**/*results*.json"))
    if not candidates:
        return None

    best, best_acc = None, -1
    for p in candidates:
        try:
            with open(p) as f:
                data = json.load(f)
            # pick test_id or test accuracy as ranking criterion
            acc = (data.get("test", {}) or data.get("test_id", {})).get("accuracy", 0)
            if acc > best_acc:
                best_acc, best = acc, data
        except Exception:
            pass
    return best


def load_all_results(results_dir: str) -> dict:
    """
    Returns: {task: {model: {split: {"acc": float, "f1": float}}}}
    """
    results_dir = Path(results_dir)
    out = {}

    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        out[task] = {}

        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            data = find_best_result_json(model_dir)
            if data is None:
                continue

            splits = {}
            for split_key, split_data in data.items():
                if not isinstance(split_data, dict):
                    continue
                acc = split_data.get("accuracy", split_data.get("acc"))
                f1  = split_data.get("f1_score", split_data.get("f1"))
                if acc is not None:
                    # normalise key: test_id -> test
                    norm_key = split_key.replace("_id", "") if split_key == "test_id" else split_key
                    splits[norm_key] = {"acc": float(acc), "f1": float(f1) if f1 else None}

            if splits:
                out[task][model] = splits

    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_task(task: str, our_data: dict, paper_data: dict, splits: list,
              output_dir: Path, metric: str = "acc"):
    """One figure per task: grouped bar chart, ours vs paper, per split."""
    models   = [m for m in MODEL_ORDER if m in our_data]
    n_models = len(models)
    n_splits = len(splits)
    if n_models == 0:
        return

    fig_w = max(10, n_models * n_splits * 0.9)
    fig, axes = plt.subplots(1, n_splits, figsize=(fig_w, 5), sharey=False)
    if n_splits == 1:
        axes = [axes]

    fig.suptitle(f"{task} — Ours vs Paper ({metric.upper()})", fontsize=13, fontweight="bold")

    colors = {"ours": "#4C72B0", "paper": "#DD8452"}

    for ax, split in zip(axes, splits):
        ours_vals  = []
        paper_vals = []

        for m in models:
            o = our_data.get(m, {}).get(split, {}).get(metric)
            p_entry = paper_data.get(m, {}).get(split)
            p = p_entry[0] if p_entry and metric == "acc" else (p_entry[1] if p_entry else None)
            ours_vals.append(o * 100 if o is not None else 0)
            paper_vals.append(p * 100 if p is not None else 0)

        x      = np.arange(n_models)
        width  = 0.35
        bars_o = ax.bar(x - width/2, ours_vals,  width, label="Ours",  color=colors["ours"],  alpha=0.85)
        bars_p = ax.bar(x + width/2, paper_vals, width, label="Paper", color=colors["paper"], alpha=0.85)

        # Value labels on bars
        for bar, val in zip(bars_o, ours_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=colors["ours"])
        for bar, val in zip(bars_p, paper_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=colors["paper"])

        ax.set_title(SPLIT_LABELS.get(split, split), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"{metric.upper()} (%)", fontsize=9)
        ax.set_ylim(0, 110)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        if ax == axes[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / f"{task}_{metric}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_summary_heatmap(all_our: dict, output_dir: Path, metric: str = "acc"):
    """
    Single heatmap: rows = (task, split), cols = models.
    Two side-by-side panels: Ours | Paper.
    """
    rows, ours_mat, paper_mat = [], [], []

    for task in sorted(all_our.keys()):
        splits = TASK_SPLITS.get(task, ["test"])
        paper  = PAPER_NUMBERS.get(task, {})
        for split in splits:
            row_label = f"{task}\n{SPLIT_LABELS.get(split, split)}"
            ours_row, paper_row = [], []
            for m in MODEL_ORDER:
                o = all_our.get(task, {}).get(m, {}).get(split, {}).get(metric)
                p_entry = paper.get(m, {}).get(split)
                p = p_entry[0] if p_entry and metric == "acc" else (p_entry[1] if p_entry else None)
                ours_row.append(o * 100 if o is not None else np.nan)
                paper_row.append(p * 100 if p is not None else np.nan)
            rows.append(row_label)
            ours_mat.append(ours_row)
            paper_mat.append(paper_row)

    ours_mat  = np.array(ours_mat,  dtype=float)
    paper_mat = np.array(paper_mat, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, len(rows) * 0.55)))
    fig.suptitle(f"CSI-Bench Baseline Results — {metric.upper()} (%)", fontsize=13, fontweight="bold")

    kw = dict(aspect="auto", vmin=0, vmax=100, cmap="RdYlGn")
    im1 = ax1.imshow(ours_mat,  **kw)
    im2 = ax2.imshow(paper_mat, **kw)

    for ax, mat, title in [(ax1, ours_mat, "Ours"), (ax2, paper_mat, "Paper")]:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in MODEL_ORDER],
                           rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=7)
        # Cell annotations
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=6.5,
                            color="black" if 30 < val < 80 else "white")

    plt.colorbar(im1, ax=ax1, fraction=0.03)
    plt.colorbar(im2, ax=ax2, fraction=0.03)
    plt.tight_layout()
    out_path = output_dir / f"summary_heatmap_{metric}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_table(task: str, our_data: dict, paper_data: dict, splits: list):
    """Print a compact comparison table to console."""
    print(f"\n{'='*70}")
    print(f"  {task}")
    print(f"{'='*70}")
    header = f"{'Model':15s}" + "".join(
        f"  {'Ours':>6} {'Paper':>6}" for s in splits
    )
    subhdr = f"{'':15s}" + "".join(f"  {SPLIT_LABELS.get(s,s):>13}" for s in splits)
    print(subhdr)
    print(header)
    print("-" * len(header))
    for m in MODEL_ORDER:
        if m not in our_data:
            continue
        row = f"{MODEL_LABELS.get(m, m):15s}"
        for split in splits:
            o = our_data.get(m, {}).get(split, {}).get("acc")
            p_entry = paper_data.get(m, {}).get(split)
            p = p_entry[0] if p_entry else None
            o_str = f"{o*100:5.1f}%" if o is not None else "  N/A "
            p_str = f"{p*100:5.1f}%" if p is not None else "  N/A "
            row += f"  {o_str} {p_str}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=
        "/home/zhuzih19/csi-project/csi-fall-detection/results/csibench_official")
    parser.add_argument("--output_dir",  default=
        "/home/zhuzih19/csi-project/csi-fall-detection/results/figures")
    parser.add_argument("--task",   default=None, help="Filter to one task")
    parser.add_argument("--metric", default="acc", choices=["acc", "f1"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    all_our = load_all_results(args.results_dir)

    tasks = [args.task] if args.task else sorted(all_our.keys())

    for task in tasks:
        if task not in all_our:
            print(f"[SKIP] {task}: no results found")
            continue
        our_data   = all_our[task]
        paper_data = PAPER_NUMBERS.get(task, {})
        splits     = TASK_SPLITS.get(task, ["test"])
        # Filter to splits we actually have
        avail = set()
        for m_data in our_data.values():
            avail.update(m_data.keys())
        splits = [s for s in splits if s in avail]
        if not splits:
            print(f"[SKIP] {task}: no matching splits")
            continue

        print_table(task, our_data, paper_data, splits)
        plot_task(task, our_data, paper_data, splits, output_dir, args.metric)

    # Summary heatmap across all tasks
    plot_summary_heatmap(all_our, output_dir, args.metric)
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
