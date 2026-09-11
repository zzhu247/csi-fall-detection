"""
plot_mask_strategies.py
--------------------------
Visualizes all five masking strategies (random, time, freq, mixed, 2d) applied to
the SAME real, preprocessed CSI sample -- for a "Fig X: masking strategies" figure
in the Methods section.

Uses the SAME preprocessing pipeline as training (load_and_normalize_csi -> crop/pad
to 232x500 -> per-sample z-score -> pad to the patch-size-aligned grid), and calls
the REAL random_mask()/block_mask() functions from models/mae_v2.py directly --
this is not a re-implementation or approximation of the masking logic, it runs the
exact same code path used during pretraining.

Usage:
    python plot_mask_strategies.py \
        --h5_path /path/to/a/real/session_XXXXX.h5 \
        --patch_h 29 --patch_w 25 --mask_ratio 0.75 \
        --seed 42 \
        --out figures/mask_strategies_example.pdf
"""
import argparse
import sys

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')  # adjust if needed
from data.dataset import load_and_normalize_csi
from models.mae_v2 import random_mask, block_mask


def compute_padded_size(orig_size, patch_size):
    """Same logic as train_mae_har.py -- smallest multiple of patch_size >= orig_size."""
    return ((orig_size + patch_size - 1) // patch_size) * patch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', required=True)
    parser.add_argument('--patch_h', type=int, default=29)
    parser.add_argument('--patch_w', type=int, default=25)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='mask_strategies_example.pdf')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ── Exact same preprocessing as training ──────────────────────────────
    csi = load_and_normalize_csi(args.h5_path)          # -> (232, 500), crop/pad done
    csi = (csi - csi.mean()) / (csi.std() + 1e-8)        # per-sample z-score, matches MultiTaskDataset

    raw_h, raw_w = csi.shape
    padded_h = compute_padded_size(raw_h, args.patch_h)
    padded_w = compute_padded_size(raw_w, args.patch_w)
    if (padded_h, padded_w) != (raw_h, raw_w):
        pad_amt_h, pad_amt_w = padded_h - raw_h, padded_w - raw_w
        csi = np.pad(csi, ((0, pad_amt_h), (0, pad_amt_w)), mode='constant')
        print(f"Padded {raw_h}x{raw_w} -> {padded_h}x{padded_w} for patch alignment "
              f"(patch_h={args.patch_h}, patch_w={args.patch_w})")

    n_h = padded_h // args.patch_h
    n_w = padded_w // args.patch_w
    N = n_h * n_w
    print(f"Patch grid: {n_h} x {n_w} = {N} patches total")

    strategies = ['random', 'time', 'freq', 'mixed', '2d']
    fig, axes = plt.subplots(1, len(strategies), figsize=(4 * len(strategies), 4.5))

    for ax, strategy in zip(axes, strategies):
        torch.manual_seed(args.seed)  # same seed per strategy for a fair visual comparison
        if strategy == 'random':
            ids_masked, ids_visible, _ = random_mask(1, N, args.mask_ratio, device='cpu')
        else:
            ids_masked, ids_visible, _ = block_mask(1, N, args.mask_ratio, n_h, n_w,
                                                     device='cpu', block_mode=strategy)

        # Build a [n_h, n_w] boolean mask from the flat patch indices this strategy produced
        patch_mask = torch.zeros(N, dtype=torch.bool)
        patch_mask[ids_masked[0]] = True
        patch_mask = patch_mask.reshape(n_h, n_w).numpy()

        # Expand the patch-level mask to pixel resolution so it overlays correctly on the CSI heatmap
        pixel_mask = np.kron(patch_mask, np.ones((args.patch_h, args.patch_w), dtype=bool))

        ax.imshow(csi, aspect='auto', cmap='viridis', origin='lower')
        # Overlay masked patches as semi-transparent gray
        overlay = np.zeros((*pixel_mask.shape, 4))
        overlay[pixel_mask] = [0.5, 0.5, 0.5, 0.75]
        ax.imshow(overlay, aspect='auto', origin='lower')

        actual_ratio = patch_mask.sum() / N
        ax.set_title(f"{strategy}\n({actual_ratio*100:.0f}% masked)")
        ax.set_xlabel('Timestep')
        if strategy == strategies[0]:
            ax.set_ylabel('Subcarrier index')

    fig.suptitle(f'Masking strategies at mask ratio = {args.mask_ratio} '
                f'(gray = masked patches, same underlying CSI sample throughout)')
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
