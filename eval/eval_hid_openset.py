"""
eval_hid_openset.py
----------------------
Open-set identity evaluation for HumanIdentification checkpoints (train_mae_hid.py).

WHY THIS IS A SEPARATE FILE, NOT AN EXTENSION OF get_features()/linear_probe_eval() ETC:
HumanIdentification's OOD splits contain identities that NEVER appear in train_id (e.g.
test_cross_user contains ONLY 'U02', who has zero samples in train_id). A classifier with a
fixed output layer (nn.Linear(encoder_dim, num_classes), built from train_id's label_map) is
STRUCTURALLY incapable of predicting a class it was never given an output slot for -- this
isn't a bug to patch, it's the wrong tool for the problem. Open-set identification needs
metrics that don't require a fixed, closed set of identities at all:

  - Rank-1 retrieval accuracy: split a set into gallery/query, and for each query embedding,
    check whether its nearest neighbor (by cosine similarity) in the gallery shares its true
    identity. Works identically whether that identity was ever seen during pretraining or not
    -- it's a similarity search, not a classification.
  - Verification AUC / EER: sample same-identity and different-identity pairs, score each pair
    by embedding cosine similarity, and measure how well that similarity score separates
    "same person" from "different person" pairs. Standard protocol from face/speaker
    verification literature (e.g. LFW-style pair evaluation).

Neither metric ever references label_map or num_classes, so both apply uniformly to every
split here -- including test_cross_device (a MIX of known + one entirely novel identity, U02)
and test_cross_user (entirely the novel identity U02) -- without needing to special-case them.

Usage:
    python eval_hid_openset.py \
        --checkpoint checkpoints/mae_hid/<exp_name>_best.pt \
        --result_json results/mae_hid/<exp_name>.json \
        --layers 1,3,6,9,12 \
        --n_verification_pairs 5000 \
        --seed 42
"""
import argparse, json, sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset
from models.mae import MAE
from models.mae_v2 import MAEv2

from train_mae_hid import compute_padded_size, pad_csi, get_features, ENCODER_HEADS, RAW_IMG_H, RAW_IMG_W

DATA_ROOT = config.DATA_ROOT
TASK = 'HumanIdentification'
META_PATH = f'{DATA_ROOT}/Multitask/{TASK}/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/{TASK}/splits'
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


def load_split_with_identity(name, meta):
    """Loads a split's raw (id, identity_label) rows -- NOT through MultiTaskDataset's
    label_map, since that would fail on identities absent from train_id (the whole point
    of this file). Identity strings (e.g. 'U02') are used directly, never mapped to an int."""
    with open(f'{SPLITS_DIR}/{name}.json') as f:
        ids = set(json.load(f))
    df = meta[meta['id'].isin(ids)].reset_index(drop=True)
    return df


class RawIdentityDataset(torch.utils.data.Dataset):
    """Like MultiTaskDataset, but returns the raw identity STRING (e.g. 'U02') instead of
    an int class index -- sidesteps label_map entirely, so it works for identities that
    were never in train_id."""
    def __init__(self, df, data_root, task):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.task = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import os
        from data.dataset import load_and_normalize_csi
        row = self.df.iloc[idx]
        h5_path = os.path.join(self.data_root, self.task, row["file_path"].lstrip("./"))
        csi = load_and_normalize_csi(h5_path)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)
        return csi, row["label"]  # identity STRING, not an int


@torch.no_grad()
def get_features_with_identity(model, loader, layer, device, padded_h, padded_w):
    """Same as train_mae_hid.py's get_features(), but the loader yields (csi, identity_str)
    instead of (csi, int_label) -- see RawIdentityDataset."""
    model.eval()
    feats, identities = [], []
    for csi, ident in loader:
        csi = pad_csi(csi.to(device), padded_h, padded_w)
        emb = model.extract_layer_embeddings(csi, [layer])
        feats.append(emb[layer].cpu())
        identities.extend(ident)
    return torch.cat(feats), np.array(identities)


def rank1_retrieval_accuracy(feats, identities, seed):
    """Randomly split (feats, identities) into gallery/query (stratified so every identity
    with >=2 samples appears in both halves), then for each query embedding, retrieve its
    nearest neighbor in the gallery (cosine similarity) and check identity match.

    Returns (accuracy, n_query, n_identities). Identities with only 1 sample total can't be
    stratified into both halves and are excluded from the query set (they'd be unqueryable
    by construction, not a bug) -- n_identities reports how many identities actually
    participated.
    """
    rng = np.random.RandomState(seed)
    gallery_idx, query_idx = [], []
    for ident in np.unique(identities):
        idxs = np.where(identities == ident)[0]
        if len(idxs) < 2:
            continue  # can't stratify a singleton identity into gallery+query
        rng.shuffle(idxs)
        split = len(idxs) // 2
        gallery_idx.extend(idxs[:max(split, 1)])
        query_idx.extend(idxs[max(split, 1):])
    gallery_idx, query_idx = np.array(gallery_idx), np.array(query_idx)

    if len(query_idx) == 0:
        return None, 0, 0  # no identity had >=2 samples -- can't evaluate retrieval here

    gallery_feats = feats[gallery_idx]
    gallery_ids = identities[gallery_idx]
    query_feats = feats[query_idx]
    query_ids = identities[query_idx]

    g_norm = gallery_feats / (gallery_feats.norm(dim=1, keepdim=True) + 1e-8)
    q_norm = query_feats / (query_feats.norm(dim=1, keepdim=True) + 1e-8)
    sim = q_norm @ g_norm.T  # [n_query, n_gallery]
    nearest = sim.argmax(dim=1).numpy()
    preds = gallery_ids[nearest]

    acc = (preds == query_ids).mean()
    return acc, len(query_idx), len(np.unique(identities))


def verification_auc_eer(feats, identities, n_pairs, seed):
    """Samples n_pairs//2 same-identity and n_pairs//2 different-identity pairs, scores each
    by cosine similarity, and computes AUC (does similarity separate same vs. different?) and
    EER (equal error rate -- the point where false-accept rate = false-reject rate, a
    standard single-number summary in verification literature)."""
    rng = np.random.RandomState(seed)
    n = len(identities)
    feats_norm = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)

    unique_ids, counts = np.unique(identities, return_counts=True)
    same_id_pool = unique_ids[counts >= 2]  # need >=2 samples to form a same-identity pair
    if len(same_id_pool) == 0:
        return None, None, 0  # can't form any same-identity pairs

    n_each = n_pairs // 2
    same_pairs, diff_pairs = [], []

    for _ in range(n_each):
        ident = rng.choice(same_id_pool)
        idxs = np.where(identities == ident)[0]
        i, j = rng.choice(idxs, size=2, replace=False)
        same_pairs.append((i, j))

    for _ in range(n_each):
        i, j = rng.choice(n, size=2, replace=False)
        if identities[i] == identities[j]:
            continue  # rare collision, just skip -- doesn't bias the pool meaningfully at this n
        diff_pairs.append((i, j))

    pairs = same_pairs + diff_pairs
    labels = np.array([1] * len(same_pairs) + [0] * len(diff_pairs))
    scores = np.array([
        (feats_norm[i] @ feats_norm[j]).item() for i, j in pairs
    ])

    if len(np.unique(labels)) < 2:
        return None, None, len(pairs)  # degenerate -- shouldn't happen given n_each>0 and same_id_pool non-empty

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return auc, eer, len(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result_json', required=True)
    parser.add_argument('--layers', default='1,3,6,9,12')
    parser.add_argument('--n_verification_pairs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
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

    for layer in layers:
        print(f"{'='*70}\nLayer {layer}\n{'='*70}")
        for split in OOD_SPLITS:
            df = load_split_with_identity(split, meta)
            ds = RawIdentityDataset(df, DATA_ROOT, 'Multitask')
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

            feats, identities = get_features_with_identity(model, loader, layer, device, padded_h, padded_w)
            n_identities_total = len(np.unique(identities))

            r1_acc, n_query, n_ident_used = rank1_retrieval_accuracy(feats, identities, args.seed)
            auc, eer, n_pairs_used = verification_auc_eer(feats, identities, args.n_verification_pairs, args.seed)

            r1_str = f"{r1_acc*100:.1f}%" if r1_acc is not None else "n/a (no identity has >=2 samples)"
            auc_str = f"{auc:.4f}" if auc is not None else "n/a"
            eer_str = f"{eer:.4f}" if eer is not None else "n/a"

            print(f"  {split:<20} n={len(df):<6} identities={n_identities_total:<3}  "
                  f"Rank1={r1_str:<12} (n_query={n_query})  "
                  f"VerifAUC={auc_str}  EER={eer_str}  (n_pairs={n_pairs_used})")
        print()


if __name__ == '__main__':
    main()
