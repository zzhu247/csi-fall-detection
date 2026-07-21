"""
visualize_mask_ablation.py
--------------------------
Reads all mask ratio ablation JSON results and generates:
  Fig 1 — Line chart: LP accuracy vs training epoch, per mask ratio
          (mean across seeds, shaded std band)
  Fig 2 — Bar chart with error bars: final LP accuracy per OOD split,
          per mask ratio (mean ± std across seeds)
  Fig 3 — Seed comparison: side-by-side per mask ratio, seed42 vs seed43
  Fig 4 — Heatmap: mask ratio × OOD split, final Layer 1 LP accuracy

Usage:
    python visualize_mask_ablation.py
Output:
    results/figures/mask_ablation_*.png
"""

import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/home/zhuzih19/csi-project/csi-fall-detection/results/mae_har")
FIG_DIR     = Path("/home/zhuzih19/csi-project/csi-fall-detection/results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASK_RATIOS = [0.5, 0.75, 0.875, 0.95]
SEEDS       = [42, 43]
LAYER       = "layer_1"          # Layer 1 = best for OOD generalization
METRIC      = "lp_acc"           # linear probe accuracy

OOD_SPLITS  = ["test_cross_device", "test_cross_env", "test_cross_user"]
ALL_SPLITS  = ["test_id"] + OOD_SPLITS

SPLIT_LABELS = {
    "test_id":           "test_id\n(in-dist)",
    "test_cross_device": "cross_device\n(OOD)",
    "test_cross_env":    "cross_env\n(OOD)",
    "test_cross_user":   "cross_user\n(OOD)",
}

# Color per mask ratio
COLORS = {
    0.5:   "#0891B2",   # teal
    0.75:  "#7C3AED",   # purple
    0.875: "#D97706",   # amber
    0.95:  "#DC2626",   # red
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   150,
})

# ── Helper: load one JSON ─────────────────────────────────────────────────────
def load_json(mask_ratio, seed):
    pattern = str(RESULTS_DIR / f"*mask{mask_ratio}*seed{seed}*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"  [WARN] No file found for mask={mask_ratio} seed={seed}")
        return None
    with open(files[0]) as f:
        return json.load(f)

# ── Helper: extract epoch-level LP curve for one split ───────────────────────
def extract_curve(data, split, layer=LAYER, metric=METRIC):
    """Returns (epochs[], values[]) from data['evals']"""
    epochs, vals = [], []
    for key, entry in sorted(data["evals"].items(), key=lambda x: int(x[0].split("_")[1])):
        ep = int(key.split("_")[1])
        val = entry[layer][split][metric]
        epochs.append(ep)
        vals.append(val * 100)   # convert to %
    return np.array(epochs), np.array(vals)

# ── Helper: extract final epoch value ────────────────────────────────────────
def final_val(data, split, layer=LAYER, metric=METRIC):
    last_key = sorted(data["evals"].keys(), key=lambda x: int(x.split("_")[1]))[-1]
    return data["evals"][last_key][layer][split][metric] * 100

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading JSON results...")
all_data = {}   # all_data[mask_ratio][seed] = json dict
for mr in MASK_RATIOS:
    all_data[mr] = {}
    for seed in SEEDS:
        d = load_json(mr, seed)
        if d:
            all_data[mr][seed] = d
            print(f"  Loaded mask={mr} seed={seed}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Line chart: LP accuracy vs epoch, per mask ratio
#          One subplot per OOD split (+ test_id for reference)
# 意义：看不同 mask ratio 的 representation 质量随 training 的变化趋势
# ─────────────────────────────────────────────────────────────────────────────
print("\nFig 1: Training curves...")
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
fig.suptitle("MAE Linear Probe Accuracy vs Training Epoch\n"
             "Layer 1 features · mean ± std across seeds 42 & 43",
             fontsize=13, fontweight="bold", y=1.02)

for ax, split in zip(axes, ALL_SPLITS):
    for mr in MASK_RATIOS:
        curves = []
        for seed in SEEDS:
            if seed in all_data[mr]:
                ep, vals = extract_curve(all_data[mr][seed], split)
                curves.append(vals)
        if not curves:
            continue
        curves = np.array(curves)
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        color = COLORS[mr]
        ax.plot(ep, mean, color=color, linewidth=2,
                label=f"mask={mr}")
        ax.fill_between(ep, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.set_title(SPLIT_LABELS[split], fontsize=11)
    ax.set_xlabel("Epoch")
    if split == "test_id":
        ax.set_ylabel("LP Accuracy (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# Shared legend
handles = [mpatches.Patch(color=COLORS[mr], label=f"mask={mr}")
           for mr in MASK_RATIOS]
fig.legend(handles=handles, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.08), fontsize=11,
           title="Mask Ratio")

plt.tight_layout()
out = FIG_DIR / "fig1_training_curves.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Bar chart: final LP accuracy per OOD split, mean ± std across seeds
# 意义：最终性能的 ablation summary，每个 split 一组 bars
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: Bar chart with error bars...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle("Final LP Accuracy by OOD Split (mean ± std, seeds 42 & 43)\n"
             "Layer 1 features · epoch 300",
             fontsize=13, fontweight="bold")

x = np.arange(len(MASK_RATIOS))
bar_w = 0.6

for ax, split in zip(axes, OOD_SPLITS):
    means, stds = [], []
    for mr in MASK_RATIOS:
        vals = [final_val(all_data[mr][s], split)
                for s in SEEDS if s in all_data[mr]]
        means.append(np.mean(vals))
        stds.append(np.std(vals) if len(vals) > 1 else 0)

    bars = ax.bar(x, means, bar_w,
                  color=[COLORS[mr] for mr in MASK_RATIOS],
                  yerr=stds, capsize=5,
                  error_kw={"linewidth": 1.5, "color": "#1E293B"})

    # Value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_title(SPLIT_LABELS[split].replace("\n", " "), fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(mr) for mr in MASK_RATIOS])
    ax.set_xlabel("Mask Ratio")
    if split == OOD_SPLITS[0]:
        ax.set_ylabel("LP Accuracy (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ymin = max(0, min(means) - max(stds) - 5)
    ax.set_ylim(ymin, max(means) + max(stds) + 6)

plt.tight_layout()
out = FIG_DIR / "fig2_final_ood_bars.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Seed comparison: seed42 vs seed43 per mask ratio
#          Scatter plot — x=seed42, y=seed43, one point per (mask, split)
# 意义：直观展示 run-to-run variance，点离对角线越远说明 variance 越大
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: Seed comparison scatter...")
fig, ax = plt.subplots(figsize=(6.5, 6.5))

all_xy = []
for mr in MASK_RATIOS:
    if 42 not in all_data[mr] or 43 not in all_data[mr]:
        continue
    for split in OOD_SPLITS:
        v42 = final_val(all_data[mr][42], split)
        v43 = final_val(all_data[mr][43], split)
        all_xy.append((v42, v43))
        ax.scatter(v42, v43, color=COLORS[mr], s=80, zorder=3,
                   label=f"mask={mr}" if split == OOD_SPLITS[0] else "")
        ax.annotate(f"{split.split('_')[-1][:3]}",
                    (v42, v43), textcoords="offset points",
                    xytext=(4, 3), fontsize=8, color=COLORS[mr])

# Diagonal y=x line (perfect agreement between seeds)
all_v = [v for pair in all_xy for v in pair]
lo, hi = min(all_v) - 2, max(all_v) + 2
ax.plot([lo, hi], [lo, hi], "--", color="#94A3B8", linewidth=1.2, label="y = x (perfect agreement)")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)

ax.set_xlabel("Seed 42 — LP Accuracy (%)", fontsize=11)
ax.set_ylabel("Seed 43 — LP Accuracy (%)", fontsize=11)
ax.set_title("Seed-to-Seed Variance in OOD LP Accuracy\n"
             "Points off diagonal = run-to-run variance\n"
             "各点偏离对角线的程度代表 seed 间的不稳定性",
             fontsize=11, fontweight="bold")

handles = [mpatches.Patch(color=COLORS[mr], label=f"mask={mr}")
           for mr in MASK_RATIOS if 42 in all_data[mr]]
handles.append(plt.Line2D([0], [0], linestyle="--", color="#94A3B8",
                           label="y=x (perfect)"))
ax.legend(handles=handles, fontsize=10)
ax.grid(linestyle="--", alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
out = FIG_DIR / "fig3_seed_comparison.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Heatmap: mask ratio × split, final Layer 1 LP accuracy (mean)
# 意义：一张图看清所有 mask ratio 在所有 splits 上的综合表现
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: Heatmap...")
fig, ax = plt.subplots(figsize=(8, 4))

splits_to_show = ALL_SPLITS
matrix = np.zeros((len(MASK_RATIOS), len(splits_to_show)))

for i, mr in enumerate(MASK_RATIOS):
    for j, split in enumerate(splits_to_show):
        vals = [final_val(all_data[mr][s], split)
                for s in SEEDS if s in all_data[mr]]
        matrix[i, j] = np.mean(vals)

im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
               vmin=matrix.min() - 2, vmax=matrix.max() + 2)

# Annotate cells
for i in range(len(MASK_RATIOS)):
    for j in range(len(splits_to_show)):
        val = matrix[i, j]
        color = "white" if val < (matrix.min() + matrix.max()) / 2 else "black"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1E293B")

ax.set_xticks(range(len(splits_to_show)))
ax.set_xticklabels([SPLIT_LABELS[s].replace("\n", " ") for s in splits_to_show],
                   fontsize=10)
ax.set_yticks(range(len(MASK_RATIOS)))
ax.set_yticklabels([f"mask={mr}" for mr in MASK_RATIOS], fontsize=10)
ax.set_title("Final LP Accuracy Heatmap — Layer 1 (mean of seeds 42 & 43)\n"
             "越亮 = accuracy 越高",
             fontsize=12, fontweight="bold")

plt.colorbar(im, ax=ax, label="LP Accuracy (%)", shrink=0.8)
plt.tight_layout()
out = FIG_DIR / "fig4_heatmap.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Summary table printed to console ─────────────────────────────────────────
print("\n" + "="*70)
print("MASK RATIO ABLATION SUMMARY — Layer 1 LP Accuracy (mean ± std)")
print("="*70)
header = f"{'Mask':>6}  {'test_id':>12}  {'cross_dev':>12}  {'cross_env':>12}  {'cross_usr':>12}  {'OOD avg':>10}"
print(header)
print("-"*70)
for mr in MASK_RATIOS:
    row = f"{mr:>6}"
    ood_vals = []
    for split in ALL_SPLITS:
        vals = [final_val(all_data[mr][s], split)
                for s in SEEDS if s in all_data[mr]]
        m, s = np.mean(vals), np.std(vals) if len(vals) > 1 else 0
        row += f"  {m:5.1f}±{s:.1f}"
        if split in OOD_SPLITS:
            ood_vals.append(m)
    ood_avg = np.mean(ood_vals)
    row += f"  {ood_avg:>8.1f}"
    print(row)

print("="*70)
print(f"\nAll figures saved to: {FIG_DIR}/")
