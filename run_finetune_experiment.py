"""
run_finetune_experiment.py
----------------------------
Step 1: sanity-check that MAEDownstreamHead(unfreeze=0) tracks close to mlp_probe_eval()
        (both are frozen-backbone, non-linear head -- should be in the same ballpark).
Step 2: compare frozen / partial-unfreeze / full-unfreeze fine-tuning against the
        existing KNN / LP / MLP-probe numbers already stored in the result JSON, on
        every split (test_id + 3 OOD splits), so you can see whether unfreezing the
        backbone helps or hurts OOD generalization for this specific checkpoint.

Reuses get_features / knn_eval / linear_probe_eval / mlp_probe_eval / finetune_eval /
pad_csi / compute_padded_size directly from train_mae_har.py -- no logic is duplicated.

Usage:
    python run_finetune_experiment.py \
        --checkpoint checkpoints/mae_har/mae_har_..._best.pt \
        --result_json results/mae_har/mae_har_....json \
        --layer 12 \
        --probe_epochs 50 \
        --finetune_epochs 25
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

# Reuse everything from train_mae_har.py directly -- do not reimplement.
from train_mae_har import (
    compute_padded_size, pad_csi, get_features, knn_eval, linear_probe_eval,
    mlp_probe_eval, finetune_eval, ENCODER_HEADS, RAW_IMG_H, RAW_IMG_W,
)

DATA_ROOT = config.DATA_ROOT
META_PATH = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/splits'
OOD_SPLITS = ['test_id', 'test_cross_device', 'test_cross_env', 'test_cross_user']


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result_json', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--probe_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.result_json) as f:
        result = json.load(f)
    train_args = result['args']
    print(f"Config: {result['exp']}")
    print(f"  patch={train_args['patch_h']}x{train_args['patch_w']}  "
          f"encoder_depth={train_args['encoder_depth']}  mask_strategy={train_args['mask_strategy']}")

    padded_h = compute_padded_size(RAW_IMG_H, train_args['patch_h'])
    padded_w = compute_padded_size(RAW_IMG_W, train_args['patch_w'])

    model = build_model(train_args, padded_h, padded_w, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}\n")

    # ---- Data ----
    meta = pd.read_csv(META_PATH)
    with open(f'{SPLITS_DIR}/train_id.json') as f:
        train_ids = set(json.load(f))
    train_df = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    num_classes = len(label_map)

    train_ds = MultiTaskDataset(train_df, DATA_ROOT, 'Multitask', label_map=label_map)
    train_loader_shuffled = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_loader_eval = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    eval_loaders = {}
    for split in OOD_SPLITS:
        with open(f'{SPLITS_DIR}/{split}.json') as f:
            ids = set(json.load(f))
        df = meta[meta['id'].isin(ids)].reset_index(drop=True)
        ds = MultiTaskDataset(df, DATA_ROOT, 'Multitask', label_map=label_map)
        eval_loaders[split] = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"  {split}: {len(ds)} samples")
    print()

    # =========================================================================
    # STEP 1 -- sanity check: MAEDownstreamHead(unfreeze=0) vs mlp_probe_eval()
    # Both are frozen-backbone + non-linear head; should be in the same ballpark.
    # =========================================================================
    print("=" * 70)
    print("STEP 1 -- consistency check (frozen backbone, two implementations)")
    print("=" * 70)

    train_feats, train_labels = get_features(model, train_loader_eval, args.layer, device, padded_h, padded_w)
    test_id_feats, test_id_labels = get_features(model, eval_loaders['test_id'], args.layer, device, padded_h, padded_w)

    mlp_acc, mlp_f1 = mlp_probe_eval(train_feats, train_labels, test_id_feats, test_id_labels,
                                     num_classes, device, epochs=args.probe_epochs)
    print(f"mlp_probe_eval (extract-then-probe):     test_id acc={mlp_acc:.4f}  f1={mlp_f1:.4f}")

    finetune_frozen = finetune_eval(model, train_loader_shuffled, {'test_id': eval_loaders['test_id']},
                                    num_classes, args.layer, device, padded_h, padded_w,
                                    epochs=args.probe_epochs, unfreeze_last_n_layers=0)
    print(f"finetune_eval(unfreeze=0) (end-to-end):  test_id acc={finetune_frozen['test_id']['acc']:.4f}  "
          f"f1={finetune_frozen['test_id']['f1']:.4f}")

    diff = abs(mlp_acc - finetune_frozen['test_id']['acc'])
    print(f"\nDifference: {diff*100:.1f}pp", end="  ")
    if diff < 0.05:
        print("-- close, consistent with expectations.")
    else:
        print("-- LARGER than expected, worth double-checking before trusting Step 2 below.")

    # =========================================================================
    # STEP 2 -- frozen / partial / full unfreeze, all splits, vs existing KNN/LP/MLP
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2 -- fine-tune comparison across unfreeze modes, all splits")
    print("=" * 70)

    # Pull existing KNN/LP/MLP numbers already stored in the result JSON for this
    # layer, if present (final checkpoint only).
    final_ckpt = sorted(result['evals'].keys(), key=lambda k: int(k.split('_')[1]))[-1]
    layer_key = f"layer_{args.layer}"
    existing = result['evals'].get(final_ckpt, {}).get(layer_key, {})

    print(f"\n{'split':<20}{'KNN':>10}{'LP':>10}{'finetune(0)':>14}{'finetune(2)':>14}{'finetune(full)':>16}")

    results_frozen = finetune_eval(model, train_loader_shuffled, eval_loaders, num_classes, args.layer,
                                   device, padded_h, padded_w, epochs=args.finetune_epochs, unfreeze_last_n_layers=0)
    results_partial = finetune_eval(model, train_loader_shuffled, eval_loaders, num_classes, args.layer,
                                    device, padded_h, padded_w, epochs=args.finetune_epochs, unfreeze_last_n_layers=2)
    results_full = finetune_eval(model, train_loader_shuffled, eval_loaders, num_classes, args.layer,
                                 device, padded_h, padded_w, epochs=args.finetune_epochs, unfreeze_last_n_layers=None)

    for split in OOD_SPLITS:
        knn = existing.get(split, {}).get('knn_acc')
        lp = existing.get(split, {}).get('lp_acc')
        knn_s = f"{knn:.4f}" if knn is not None else "n/a"
        lp_s = f"{lp:.4f}" if lp is not None else "n/a"
        print(f"{split:<20}{knn_s:>10}{lp_s:>10}"
              f"{results_frozen[split]['acc']:>14.4f}"
              f"{results_partial[split]['acc']:>14.4f}"
              f"{results_full[split]['acc']:>16.4f}")

    print("\nLook especially at the three OOD rows (not test_id): if finetune(full) is")
    print("noticeably LOWER than finetune(0) there despite being higher on test_id, that's")
    print("catastrophic forgetting -- backbone fine-tuning is trading OOD generalization")
    print("for in-distribution fit. If finetune(full) is higher on OOD too, fine-tuning helps.")


if __name__ == '__main__':
    main()
