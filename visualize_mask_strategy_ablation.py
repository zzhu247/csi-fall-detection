"""
Visualize MAE-HAR masking-strategy ablation results.

Auto-discovers result JSON files in a directory, groups them by
`args.mask_strategy`, aggregates across `args.seed` (mean +/- std),
and produces:
    1. loss_curve_by_strategy.png   - mean loss curve per strategy, std band across seeds
    2. bar_lp_acc_by_strategy.png   - final-epoch LP accuracy per test split, grouped by strategy
    3. bar_knn_acc_by_strategy.png  - final-epoch KNN accuracy per test split, grouped by strategy
    4. summary table printed to stdout

Filtering: only files whose args match --mask_ratio / --encoder_depth / --patch_h / --patch_w
are included, so mask-ratio-ablation or layer-depth-ablation runs (e.g. enc12) don't leak in.

Usage:
    python visualize_mask_strategy_ablation.py --dir /path/to/results \
        --mask_ratio 0.75 --encoder_depth 6 --patch_h 29 --patch_w 25 \
        --outdir ./figs
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TEST_SPLITS = ["test_id", "test_cross_device", "test_cross_env", "test_cross_user"]
SPLIT_LABELS = {
    "test_id": "ID (in-distribution)",
    "test_cross_device": "Cross-Device",
    "test_cross_env": "Cross-Env",
    "test_cross_user": "Cross-User",
}
STRATEGY_COLORS = {
    "random": "tab:blue",
    "freq": "tab:green",
    "mixed": "tab:orange",
    "time": "tab:red",
}


def load_matching_results(directory, mask_ratio, encoder_depth, patch_h, patch_w, debug=False):
    """Load all JSON files under `directory` (recursively) whose args match the given filter values.

    Files that fail to parse, lack an 'args' key, or don't match the filter
    are skipped. With debug=True, prints why each candidate file was rejected,
    which is the fastest way to diagnose a "no matching files" result.
    """
    results = []
    candidates = sorted(Path(directory).rglob("*.json"))
    if debug:
        print(f"[debug] found {len(candidates)} .json files under {directory} (recursive)")

    for fp in candidates:
        try:
            data = json.loads(fp.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            if debug:
                print(f"[skip] {fp.name}: failed to parse ({e})")
            continue

        args = data.get("args")
        if args is None:
            if debug:
                print(f"[skip] {fp.name}: no 'args' key")
            continue
        if "mask_strategy" not in args:
            if debug:
                print(f"[skip] {fp.name}: args has no 'mask_strategy' key -> {list(args.keys())}")
            continue

        reasons = []
        # Cast defensively: args values may come through as str depending on how the
        # training script serialized argparse output (e.g. "0.75" instead of 0.75).
        file_mask_ratio = args.get("mask_ratio")
        if mask_ratio is not None:
            try:
                if not np.isclose(float(file_mask_ratio), mask_ratio):
                    reasons.append(f"mask_ratio={file_mask_ratio!r} != {mask_ratio}")
            except (TypeError, ValueError):
                reasons.append(f"mask_ratio={file_mask_ratio!r} not numeric")

        file_encoder_depth = args.get("encoder_depth")
        if encoder_depth is not None and int(file_encoder_depth) != encoder_depth:
            reasons.append(f"encoder_depth={file_encoder_depth!r} != {encoder_depth}")

        file_patch_h = args.get("patch_h")
        if patch_h is not None and file_patch_h is not None and int(file_patch_h) != patch_h:
            reasons.append(f"patch_h={file_patch_h!r} != {patch_h}")
        if patch_h is not None and file_patch_h is None:
            reasons.append("patch_h missing from args (older run without configurable patch size?)")

        file_patch_w = args.get("patch_w")
        if patch_w is not None and file_patch_w is not None and int(file_patch_w) != patch_w:
            reasons.append(f"patch_w={file_patch_w!r} != {patch_w}")
        if patch_w is not None and file_patch_w is None:
            reasons.append("patch_w missing from args (older run without configurable patch size?)")

        if reasons:
            if debug:
                print(f"[skip] {fp.name}: " + "; ".join(reasons))
            continue

        if debug:
            print(f"[match] {fp.name}")
        results.append(data)
    return results


def group_by_strategy(results):
    """Group loaded results by args.mask_strategy -> list of result dicts (one per seed)."""
    groups = defaultdict(list)
    for r in results:
        groups[r["args"]["mask_strategy"]].append(r)
    return groups


def get_layers(result):
    first_ckpt = next(iter(result["evals"].values()))
    layers = list(first_ckpt.keys())
    layers.sort(key=lambda x: int(re.search(r"\d+", x).group()))
    return layers


def get_checkpoints(result):
    ckpts = list(result["evals"].keys())
    ckpts.sort(key=lambda x: int(re.search(r"\d+", x).group()))
    return ckpts


# ---------------------------------------------------------------------------
# Plot 1: loss curve, mean +/- std across seeds, one line per strategy
# ---------------------------------------------------------------------------
def plot_loss_curve_by_strategy(groups, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, runs in sorted(groups.items()):
        # assume all runs share the same epoch grid
        epochs = np.array([pt["epoch"] for pt in runs[0]["loss_log"]])
        loss_matrix = np.array([[pt["loss"] for pt in r["loss_log"]] for r in runs])
        mean_loss = loss_matrix.mean(axis=0)
        std_loss = loss_matrix.std(axis=0)
        color = STRATEGY_COLORS.get(strategy, None)
        ax.plot(epochs, mean_loss, label=f"{strategy} (n={len(runs)})", color=color, linewidth=1.5)
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Reconstruction Loss")
    ax.set_title("Training Loss by Mask Strategy (mean +/- std across seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "loss_curve_by_strategy.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2/3: grouped bar chart, final-epoch accuracy per test split, by strategy
# ---------------------------------------------------------------------------
def plot_final_accuracy_by_strategy(groups, outdir, metric="lp_acc", layer=None):
    strategies = sorted(groups.keys())
    x = np.arange(len(TEST_SPLITS))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strategy in enumerate(strategies):
        runs = groups[strategy]
        target_layer = layer or get_layers(runs[0])[-1]  # default: deepest layer
        means, stds = [], []
        for split in TEST_SPLITS:
            vals = []
            for r in runs:
                final_ckpt = get_checkpoints(r)[-1]
                vals.append(r["evals"][final_ckpt][target_layer][split][metric])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
               label=f"{strategy} (n={len(runs)})", color=STRATEGY_COLORS.get(strategy, None))

    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in TEST_SPLITS])
    ax.set_ylabel(metric)
    layer_note = layer or f"{get_layers(next(iter(groups.values()))[0])[-1]} (deepest)"
    ax.set_title(f"Final-Epoch {metric} by Mask Strategy @ {layer_note}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"bar_{metric}_by_strategy.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary_table(groups, layer=None, metric="lp_acc"):
    print(f"\n=== Summary: {metric} (mean +/- std across seeds) ===")
    header = f"{'strategy':<10}" + "".join(f"{SPLIT_LABELS[s]:>20}" for s in TEST_SPLITS)
    print(header)
    for strategy in sorted(groups.keys()):
        runs = groups[strategy]
        target_layer = layer or get_layers(runs[0])[-1]
        row = f"{strategy:<10}"
        for split in TEST_SPLITS:
            vals = []
            for r in runs:
                final_ckpt = get_checkpoints(r)[-1]
                vals.append(r["evals"][final_ckpt][target_layer][split][metric])
            row += f"{np.mean(vals):>14.4f} +/- {np.std(vals):<4.3f}"
        print(row)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize MAE HAR mask-strategy ablation results")
    parser.add_argument("--dir", default=".", help="Directory containing result JSON files")
    parser.add_argument("--outdir", default="./figs", help="Output directory for figures")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Filter: args.mask_ratio")
    parser.add_argument("--encoder_depth", type=int, default=6, help="Filter: args.encoder_depth")
    parser.add_argument("--patch_h", type=int, default=29, help="Filter: args.patch_h")
    parser.add_argument("--patch_w", type=int, default=25, help="Filter: args.patch_w")
    parser.add_argument("--layer", default=None,
                         help="Layer key to use for bar charts, e.g. 'layer_6'. Default: deepest available layer")
    parser.add_argument("--debug", action="store_true",
                         help="Print per-file match/skip reasons (use this first if 0 files match)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = load_matching_results(args.dir, args.mask_ratio, args.encoder_depth, args.patch_h, args.patch_w,
                                     debug=args.debug)
    if not results:
        print("No matching result files found. Re-run with --debug to see why each file was skipped, "
              "e.g.:\n  python visualize_mask_strategy_ablation.py --dir "
              f"{args.dir} --debug")
        return

    groups = group_by_strategy(results)
    print(f"Found {len(results)} matching files across {len(groups)} strategies: "
          f"{ {k: len(v) for k, v in groups.items()} }")

    plot_loss_curve_by_strategy(groups, outdir)
    plot_final_accuracy_by_strategy(groups, outdir, metric="lp_acc", layer=args.layer)
    plot_final_accuracy_by_strategy(groups, outdir, metric="knn_acc", layer=args.layer)

    print_summary_table(groups, layer=args.layer, metric="lp_acc")
    print_summary_table(groups, layer=args.layer, metric="knn_acc")

    print(f"\nSaved figures to {outdir.resolve()}")


if __name__ == "__main__":
    main()