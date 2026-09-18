"""
add_attentive_probe_to_results.py
-------------------------------------
Retroactively adds attentive-probe (light + heavy) evaluation to an EXISTING
train_mae_har.py result JSON, without re-running pretraining. Loads the already-trained
checkpoint (*_best.pt), re-derives the padded input shape and eval_layers from the
saved config, extracts UNPOOLED sequence features (get_sequence_features /
extract_sequence_embeddings -- distinct from the pooled get_features() used by
KNN/LP/MLP), trains both attentive-probe variants (light + heavy) on train_id and
evaluates on every OOD split, then writes attn_light_acc/attn_light_f1/attn_heavy_acc/
attn_heavy_f1 into the existing result JSON's layer_results for each split -- alongside
the knn_acc/lp_acc/mlp_acc entries that are already there, not replacing them.

This is the same "load an existing checkpoint, add one retroactive metric" pattern as
add_mlp_probe_to_results.py -- reuse instead of a full 300-epoch re-pretrain, which the
attentive-probe columns don't require (they only need the already-pretrained, FROZEN
backbone's sequence features).

Usage:
    python add_attentive_probe_to_results.py \
        --checkpoint checkpoints/mae_har/<exp_name>_best.pt \
        --result_json results/mae_har/<exp_name>.json \
        --layers 1,3,6,9,12 \
        --attentive_probe_epochs 50
"""
import argparse
import json
import sys

import torch
import pandas as pd

sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset
from models.mae import MAE
from models.mae_v2 import MAEv2

# Reuses the exact same functions defined in train_mae_har.py -- not re-implemented here,
# to guarantee bit-identical behavior to what a --run_attentive_probe run would have
# produced inline during training.
from train.train_mae_har import (
    compute_padded_size, pad_csi, get_sequence_features, attentive_probe_eval,
    ENCODER_HEADS, RAW_IMG_H, RAW_IMG_W, META_PATH, SPLITS_DIR, DATA_ROOT, OOD_SPLITS,
    load_split,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result_json', required=True)
    parser.add_argument('--layers', default='1,3,6,9,12')
    parser.add_argument('--attentive_probe_epochs', type=int, default=50)
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

    padded_h = train_args.get('padded_h') or compute_padded_size(RAW_IMG_H, train_args['patch_h'])
    padded_w = train_args.get('padded_w') or compute_padded_size(RAW_IMG_W, train_args['patch_w'])

    model = build_model(train_args, padded_h, padded_w, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}\n")

    meta = pd.read_csv(META_PATH)
    exclude_devices = [d.strip() for d in train_args.get('exclude_devices', '').split(',') if d.strip()]
    if exclude_devices:
        meta = meta[~meta['device'].isin(exclude_devices)].reset_index(drop=True)
        print(f"[exclude-devices] re-applied {exclude_devices} to match the original run's "
              f"filtering (meta now {len(meta)} rows)")

    with open(f'{SPLITS_DIR}/train_id.json') as f:
        train_ids = set(json.load(f))
    train_df = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    num_classes = len(label_map)
    print(f"label_map: {label_map}  num_classes: {num_classes}\n")

    train_ds = MultiTaskDataset(train_df, DATA_ROOT, 'Multitask', label_map=label_map)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4)

    ood_loaders = {}
    for sname in OOD_SPLITS:
        if sname == 'test_id':
            continue
        ds = load_split(sname, meta, label_map)
        ood_loaders[sname] = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                         shuffle=False, num_workers=4)

    # find the final checkpoint key (e.g. 'epoch_300') already present in the result JSON --
    # attentive-probe columns get added to THIS checkpoint's layer_results, matching what
    # a --run_attentive_probe run would have produced at that same eval point.
    final_ckpt = sorted(result['evals'].keys(), key=lambda k: int(k.split('_')[1]))[-1]
    print(f"Adding attentive-probe columns to checkpoint: {final_ckpt}\n")

    for layer in layers:
        layer_key = f'layer_{layer}'
        if layer_key not in result['evals'][final_ckpt]:
            print(f"[skip] {layer_key} not found in {final_ckpt} -- was this layer evaluated "
                  f"in the original run?")
            continue

        print(f"Layer {layer}: extracting unpooled sequence features (train_id)...")
        train_seq_feats, train_labels = get_sequence_features(
            model, train_loader, layer, device, padded_h, padded_w)

        for sname in OOD_SPLITS:
            if sname == 'test_id':
                eval_seq_feats, eval_labels = train_seq_feats, train_labels
            else:
                eval_seq_feats, eval_labels = get_sequence_features(
                    model, ood_loaders[sname], layer, device, padded_h, padded_w)

            attn_light_acc, attn_light_f1 = attentive_probe_eval(
                train_seq_feats, train_labels, eval_seq_feats, eval_labels,
                num_classes, device, epochs=args.attentive_probe_epochs, heavyweight=False)
            attn_heavy_acc, attn_heavy_f1 = attentive_probe_eval(
                train_seq_feats, train_labels, eval_seq_feats, eval_labels,
                num_classes, device, epochs=args.attentive_probe_epochs, heavyweight=True)

            result['evals'][final_ckpt][layer_key][sname]['attn_light_acc'] = attn_light_acc
            result['evals'][final_ckpt][layer_key][sname]['attn_light_f1'] = attn_light_f1
            result['evals'][final_ckpt][layer_key][sname]['attn_heavy_acc'] = attn_heavy_acc
            result['evals'][final_ckpt][layer_key][sname]['attn_heavy_f1'] = attn_heavy_f1

            print(f"  {sname:20s} AttnLight={attn_light_acc*100:.1f}%  "
                  f"AttnHeavy={attn_heavy_acc*100:.1f}%")

    with open(args.result_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nUpdated in place: {args.result_json}")


if __name__ == '__main__':
    main()
