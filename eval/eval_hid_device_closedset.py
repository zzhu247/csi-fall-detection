"""
eval_hid_device_closedset.py
-------------------------------
Computes CLOSED-SET accuracy (KNN/LP/MLP) on the KNOWN-IDENTITY SUBSET of
test_cross_device -- i.e. test_cross_device with any identity absent from train_id's
label_map (e.g. 'U02') filtered out.

WHY THIS EXISTS: test_cross_device mixes 6 known identities with 1 unknown identity
('U02'), so train_mae_hid.py routes it through OPEN-SET metrics (Rank-1/VerifAUC), which
are NOT directly comparable to CSI-Bench's official Tab. 13 (multitask LoRA)
Acc_cross_device / F1_cross_device numbers -- those are ordinary closed-set classification
accuracy. This script fills that specific gap: restrict to the SAME 6-identity closed label
space as train_id, and compute accuracy the same way KNN/LP/MLP do everywhere else in this
project, so the number is genuinely apples-to-apples with the baseline table.

Does NOT do this for test_cross_user -- that split is ENTIRELY the unknown identity, so
filtering out unknown identities would leave zero samples. There is no meaningful closed-set
number for test_cross_user; open-set metrics (see eval_hid_openset.py / the integrated
open-set path in train_mae_hid.py) are the only option there, and that's expected -- see
the "cross_user structural zero" note for the official baseline (Tab. 13).

Usage:
    python eval_hid_device_closedset.py \
        --checkpoint checkpoints/mae_hid/<exp_name>_best.pt \
        --result_json results/mae_hid/<exp_name>.json \
        --layers 1,3,6,9,12
"""
import argparse, json, sys
from pathlib import Path

import torch
import pandas as pd

sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset
from models.mae import MAE
from models.mae_v2 import MAEv2

from train.train_mae_hid import (
    compute_padded_size, pad_csi, get_features, knn_eval, linear_probe_eval, mlp_probe_eval,
    ENCODER_HEADS, RAW_IMG_H, RAW_IMG_W, META_PATH, SPLITS_DIR, DATA_ROOT,
)


def build_model(train_args, padded_h, padded_w, device):
    common = dict(
        in_channels=1, img_h=padded_h, img_w=padded_w,
        patch_h=train_args['patch_h'], patch_w=train_args['patch_w'],
        encoder_dim=train_args['encoder_dim'], encoder_ff_dim=train_args['encoder_dim'] * 4,
        encoder_heads=ENCODER_HEADS, encoder_depth=train_args['encoder_depth'],
        decoder_dim=train_args['decoder_dim'], decoder_heads=2, decoder_depth=2,
        mask_ratio=train_args['mask_ratio'],
    )
    if train_args['mask_strategy'] == 'random':
        return MAE(**common).to(device)
    return MAEv2(**common, mask_strategy=train_args['mask_strategy']).to(device)


def get_closedset_subset(split_name, meta, known_identities, splits_dir):
    """Loads a split and filters it down to ONLY the identities present in
    known_identities (typically train_id's label_map keys). Returns
    (filtered_df, dropped_identities, n_dropped)."""
    with open(f'{splits_dir}/{split_name}.json') as f:
        split_ids = set(json.load(f))
    full_df = meta[meta['id'].isin(split_ids)].reset_index(drop=True)
    identities_present = set(full_df['label'].unique())
    unknown = identities_present - known_identities
    filtered_df = full_df[full_df['label'].isin(known_identities)].reset_index(drop=True)
    return filtered_df, unknown, len(full_df) - len(filtered_df), full_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result_json', required=True)
    parser.add_argument('--layers', default='1,3,6,9,12')
    parser.add_argument('--probe_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.result_json) as f:
        result = json.load(f)
    train_args = result['args']
    print(f"Config: {result['exp']}")

    valid_layers = [l for l in layers if l <= train_args['encoder_depth']]
    if len(valid_layers) < len(layers):
        print(f"[warn] dropping layers {[l for l in layers if l > train_args['encoder_depth']]} "
              f"(encoder_depth={train_args['encoder_depth']})")
    layers = valid_layers

    padded_h = compute_padded_size(RAW_IMG_H, train_args['patch_h'])
    padded_w = compute_padded_size(RAW_IMG_W, train_args['patch_w'])

    model = build_model(train_args, padded_h, padded_w, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}\n")

    meta = pd.read_csv(META_PATH)

    # Rebuild train_id's label_map exactly as train_mae_hid.py's main() does
    with open(f'{SPLITS_DIR}/train_id.json') as f:
        train_ids = set(json.load(f))
    train_df = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    num_classes = len(label_map)
    known_identities = set(label_map.keys())
    print(f"label_map (from train_id): {label_map}\n")

    cd_closedset_df, unknown_in_cd, n_dropped, cd_full_df = get_closedset_subset(
        'test_cross_device', meta, known_identities, SPLITS_DIR)

    print(f"test_cross_device: {len(cd_full_df)} total samples, "
          f"identities present={sorted(cd_full_df['label'].unique())}")
    print(f"  Dropping unknown identities {sorted(unknown_in_cd)}: {n_dropped} samples removed")
    print(f"  Closed-set subset: {len(cd_closedset_df)} samples, "
          f"{len(cd_closedset_df['label'].unique())} identities "
          f"({sorted(cd_closedset_df['label'].unique())})\n")

    if len(cd_closedset_df) == 0:
        print("ERROR: no known-identity samples remain in test_cross_device after "
              "filtering -- cannot compute closed-set accuracy. This would mean ALL "
              "identities in this split are unknown, which contradicts the earlier "
              "diagnosis (6 known + 1 unknown). Double check SPLITS_DIR / label_map.")
        sys.exit(1)

    train_ds = MultiTaskDataset(train_df, DATA_ROOT, 'Multitask', label_map=label_map)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4)
    cd_ds = MultiTaskDataset(cd_closedset_df, DATA_ROOT, 'Multitask', label_map=label_map)
    cd_loader = torch.utils.data.DataLoader(cd_ds, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)

    print("=" * 70)
    print("Closed-set Cross-Device accuracy (known identities only -- directly")
    print("comparable to CSI-Bench Tab.13 Acc_cross_device / F1_cross_device)")
    print("=" * 70)

    results_by_layer = {}
    for layer in layers:
        train_feats, train_labels = get_features(model, train_loader, layer, device, padded_h, padded_w)
        eval_feats, eval_labels = get_features(model, cd_loader, layer, device, padded_h, padded_w)

        knn_acc, knn_f1 = knn_eval(train_feats, train_labels, eval_feats, eval_labels, k=10)
        lp_acc, lp_f1 = linear_probe_eval(train_feats, train_labels, eval_feats, eval_labels,
                                          num_classes, device, epochs=args.probe_epochs)
        mlp_acc, mlp_f1 = mlp_probe_eval(train_feats, train_labels, eval_feats, eval_labels,
                                         num_classes, device, epochs=args.probe_epochs)

        results_by_layer[layer] = {
            'knn_acc': knn_acc, 'knn_f1': knn_f1,
            'lp_acc': lp_acc, 'lp_f1': lp_f1,
            'mlp_acc': mlp_acc, 'mlp_f1': mlp_f1,
        }
        print(f"  Layer {layer:2d}: KNN acc={knn_acc:.4f} f1={knn_f1:.4f}   "
              f"LP acc={lp_acc:.4f} f1={lp_f1:.4f}   MLP acc={mlp_acc:.4f} f1={mlp_f1:.4f}")

    print(f"\nFor direct comparison against CSI-Bench Tab.13 (HumanIdentification, multitask LoRA):")
    print(f"  transformer:    Acc={0.2151:.4f}  F1={0.1996:.4f}")
    print(f"  patchtst:       Acc={0.2284:.4f}  F1={0.2565:.4f}")
    print(f"  timesformer1d:  Acc={0.2037:.4f}  F1={0.2722:.4f}")


if __name__ == '__main__':
    main()
