"""
visualize_embeddings.py
------------------------
Loads a trained MAE/MAEv2 checkpoint, extracts frozen embeddings for train_id and one
OOD split, projects them to 2D with t-SNE, and plots two views:
  1. Colored by class label (jumping/running/seated-breathing/walking/wavinghand)
  2. Colored by domain (train_id vs the chosen OOD split), so you can see whether
     the OOD shift itself is visible as a separate cluster in embedding space

Purpose: diagnose why KNN accuracy >> Linear Probe accuracy on test_id (see project
discussion) -- if same-class points form curved / non-convex / multi-modal shapes
rather than clean convex clusters, that's direct visual evidence that the representation
is locally separable (which KNN can exploit) but not linearly separable (which LP can't),
explaining the gap without it being a training bug.

Usage:
    python visualize_embeddings.py \
        --checkpoint checkpoints/mae_har/mae_har_..._best.pt \
        --result_json results/mae_har/mae_har_....json \
        --layer 6 \
        --ood_split test_cross_device \
        --n_samples 2000 \
        --out_dir figs/embeddings

--result_json is used only to read back the training args (mask_ratio, patch_h/w,
encoder_depth/dim, etc.) that the checkpoint doesn't store on its own -- point it at
the paired result JSON for this exp_name (same directory, same filename minus "_best.pt").
"""
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset
from models.mae import MAE
from models.mae_v2 import MAEv2

DATA_ROOT = config.DATA_ROOT
META_PATH = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/splits'

RAW_IMG_H, RAW_IMG_W = 232, 500
ENCODER_HEADS = 4


def compute_padded_size(orig_size, patch_size):
    return ((orig_size + patch_size - 1) // patch_size) * patch_size


def pad_csi(x, padded_h, padded_w):
    _, _, H, W = x.shape
    pad_h, pad_w = padded_h - H, padded_w - W
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h))


def build_model(args, padded_h, padded_w, device):
    common = dict(
        in_channels=1, img_h=padded_h, img_w=padded_w,
        patch_h=args['patch_h'], patch_w=args['patch_w'],
        encoder_dim=args['encoder_dim'], encoder_ff_dim=args['encoder_dim'] * 4,
        encoder_heads=ENCODER_HEADS, encoder_depth=args['encoder_depth'],
        decoder_dim=args['decoder_dim'], decoder_heads=2, decoder_depth=2,
        mask_ratio=args['mask_ratio'],
    )
    if args['mask_strategy'] == 'random':
        model = MAE(**common)
    else:
        model = MAEv2(**common, mask_strategy=args['mask_strategy'])
    return model.to(device)


@torch.no_grad()
def extract_embeddings(model, loader, layer, device, padded_h, padded_w, n_samples):
    model.eval()
    feats, labels = [], []
    total = 0
    for csi, y in loader:
        csi = pad_csi(csi.to(device), padded_h, padded_w)
        emb = model.extract_layer_embeddings(csi, [layer])[layer]
        feats.append(emb.cpu())
        labels.append(y)
        total += csi.shape[0]
        if total >= n_samples:
            break
    feats = torch.cat(feats)[:n_samples]
    labels = torch.cat(labels)[:n_samples]
    return feats, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result_json', required=True,
                         help='Paired result JSON for this exp_name (provides training args)')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--ood_split', default='test_cross_device',
                         choices=['test_cross_device', 'test_cross_env', 'test_cross_user'])
    parser.add_argument('--n_samples', type=int, default=2000,
                         help='Max samples per split fed into t-SNE (t-SNE is O(n^2), keep this modest)')
    parser.add_argument('--out_dir', default='figs/embeddings')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.result_json) as f:
        result = json.load(f)
    train_args = result['args']
    print(f"Loaded args from {args.result_json}: "
          f"patch={train_args['patch_h']}x{train_args['patch_w']}, "
          f"encoder_depth={train_args['encoder_depth']}, mask_strategy={train_args['mask_strategy']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    padded_h = compute_padded_size(RAW_IMG_H, train_args['patch_h'])
    padded_w = compute_padded_size(RAW_IMG_W, train_args['patch_w'])

    model = build_model(train_args, padded_h, padded_w, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}")

    # Data
    import pandas as pd
    meta = pd.read_csv(META_PATH)
    with open(f'{SPLITS_DIR}/train_id.json') as f:
        train_ids = set(json.load(f))
    train_df = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    inv_label_map = {v: k for k, v in label_map.items()}

    train_ds = MultiTaskDataset(train_df, DATA_ROOT, 'Multitask', label_map=label_map)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)

    with open(f'{SPLITS_DIR}/{args.ood_split}.json') as f:
        ood_ids = set(json.load(f))
    ood_df = meta[meta['id'].isin(ood_ids)].reset_index(drop=True)
    ood_ds = MultiTaskDataset(ood_df, DATA_ROOT, 'Multitask', label_map=label_map)
    ood_loader = torch.utils.data.DataLoader(ood_ds, batch_size=128, shuffle=True, num_workers=4)

    print(f"Extracting layer-{args.layer} embeddings (up to {args.n_samples} samples per split)...")
    train_feats, train_labels = extract_embeddings(
        model, train_loader, args.layer, device, padded_h, padded_w, args.n_samples)
    ood_feats, ood_labels = extract_embeddings(
        model, ood_loader, args.layer, device, padded_h, padded_w, args.n_samples)

    # ---- t-SNE ----
    from sklearn.manifold import TSNE
    all_feats = torch.cat([train_feats, ood_feats]).numpy()
    all_labels = torch.cat([train_labels, ood_labels]).numpy()
    domain = np.array(['train_id'] * len(train_feats) + [args.ood_split] * len(ood_feats))

    print(f"Running t-SNE on {len(all_feats)} points...")
    emb_2d = TSNE(n_components=2, init='pca', random_state=42, perplexity=30).fit_transform(all_feats)

    # ---- Plot 1: colored by class label ----
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.colormaps['tab10']
    for cls_idx, cls_name in inv_label_map.items():
        mask = all_labels == cls_idx
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=8, alpha=0.6,
                   color=cmap(cls_idx), label=cls_name)
    ax.set_title(f"t-SNE by class label (layer {args.layer}, train_id + {args.ood_split})")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_by_class_layer{args.layer}.png", dpi=150)
    plt.close(fig)

    # ---- Plot 2: colored by domain (train_id vs OOD split) ----
    fig, ax = plt.subplots(figsize=(8, 7))
    for dom, color in [('train_id', 'tab:blue'), (args.ood_split, 'tab:red')]:
        mask = domain == dom
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=8, alpha=0.5, color=color, label=dom)
    ax.set_title(f"t-SNE by domain (layer {args.layer})")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_by_domain_layer{args.layer}.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved:\n  {out_dir / f'tsne_by_class_layer{args.layer}.png'}\n  {out_dir / f'tsne_by_domain_layer{args.layer}.png'}")


if __name__ == '__main__':
    main()
