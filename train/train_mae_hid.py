"""
train_mae_hid.py
Standalone copy of train_mae_har.py, retargeted at Multitask/HumanIdentification instead of
HumanActivityRecognition. Deliberately a SEPARATE FILE (not a --task flag on train_mae_har.py)
so the two tasks' code never share a runtime path -- changes to one file can't accidentally
affect the other's behavior. The two files WILL drift over time as each is edited independently;
that's an accepted tradeoff for keeping them fully isolated, not an oversight.

MAE pretraining on HumanIdentification train_id, followed by:
  - KNN eval on test_id + OOD splits
  - Linear probe eval on test_id + OOD splits

Usage:
    python train_mae_hid.py --epochs 300 --mask_ratio 0.75 --encoder_depth 6

Patch-size ablation notes (added):
    Square patch sizes that don't evenly divide the standard 232x500 input
    (e.g. 3, 5, 7, 11, 13) are handled by zero-padding the CSI tensor up to
    the nearest multiple of patch_h/patch_w before it enters the model --
    see pad_csi() / compute_padded_size() below. The model is constructed
    with img_h/img_w set to the PADDED size, not the raw 232x500.

    IMPORTANT: models/vit.py's MultiHeadAttention is a naive (non-flash)
    implementation, so its attention score tensor is O(B * heads * N^2) in
    memory, where N = num_patches. Small patch sizes blow this up fast:
        patch= 3x3  -> N=13026  (attention alone: hundreds of GB at bs=128)
        patch= 5x5  -> N= 4700  (tens of GB at bs=128)
        patch= 7x7  -> N= 2448  (~11 GB/layer at bs=128 -- still risky)
        patch=11x11 -> N= 1012  (~2 GB/layer at bs=128 -- fine)
        patch=13x13 -> N=  702  (~1 GB/layer at bs=128 -- fine)
    check_attention_memory() below estimates this before training starts
    and hard-stops with a suggested safe --batch_size instead of letting
    you OOM 20+ minutes into a run. Use --skip_mem_check to bypass (not
    recommended unless you've already sized batch_size yourself).

Split-selection knobs (added):
    Previously the train split ('train_id') and eval splits (OOD_SPLITS,
    hardcoded to ['test_id', 'test_cross_device', 'test_cross_env',
    'test_cross_user']) were hardcoded, so there was no way to point this
    script at the corrected, session-disjoint splits (train_final/val_final/
    test_final, or the nested train_hp/val_hp used for hyperparameter
    search) without editing the file. --train_split and --eval_splits make
    both configurable from the CLI:
        - Hyperparameter search (choosing mask_ratio, patch size, etc.):
              --train_split train_hp --eval_splits val_hp,test_cross_device,test_cross_env,test_cross_user
          Never pass test_final or test_id here -- selecting hyperparameters
          against a split, then reporting on that same split, reintroduces
          exactly the kind of meta-level leakage the corrected splits were
          built to avoid.
        - Final reported run (hyperparameters already locked in):
              --train_split train_final --eval_splits test_final,test_cross_device,test_cross_env,test_cross_user
        - test_id is the ORIGINAL, session-leaky in-distribution split
          (confirmed 100% train/test session overlap for HID) -- do not use
          it for any number that goes in the paper. Left as the default only
          for backward compatibility; always pass --train_split/--eval_splits
          explicitly going forward.
    The open-set vs. closed-set routing below (RawIdentityDataset / rank1_retrieval_accuracy
    / verification_auc_eer vs. label_map-based KNN/LP/MLP) now checks EACH split named in
    --eval_splits against label_map, rather than only the hardcoded OOD_SPLITS list -- this
    matters for val_hp/test_final too, in principle, though in practice they're built as
    trial-level subsets of train's own identity pool, so they're expected to be closed-set
    (every identity in val_hp/test_final already has a label_map entry from train_hp/train_final).
"""
import os, sys, json, argparse, random, math, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset, load_and_normalize_csi
from models.mae import MAE
from models.mae_v2 import MAEv2

DATA_ROOT  = config.DATA_ROOT
TASK       = 'HumanIdentification'  # the one thing this file changes vs. train_mae_har.py's data source
META_PATH  = f'{DATA_ROOT}/Multitask/{TASK}/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/{TASK}/splits'
RESULTS_DIR = '/home/zhuzih19/csi-project/csi-fall-detection/results/mae_hid'
CKPT_DIR    = '/home/zhuzih19/csi-project/csi-fall-detection/checkpoints/mae_hid'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

# NOTE: kept for backward compatibility (nothing else in this file references it any
# more -- main() now builds its eval split list from args.eval_splits). See the
# "Split-selection knobs" note in the module docstring above for what to pass instead.
OOD_SPLITS = ['test_id', 'test_cross_device', 'test_cross_env', 'test_cross_user']

RAW_IMG_H, RAW_IMG_W = 232, 500  # standard CSI-Bench input shape (subcarriers x timesteps)
ENCODER_HEADS = 4                # hardcoded to match existing model construction below

# ── Domain-adversarial pretraining (DANN, Ganin & Lempitsky 2016) ─────────────
class GradientReversalFunction(torch.autograd.Function):
    """Forward: identity. Backward: negates and scales the incoming gradient by lambda_.
    This is the entire mechanism behind domain-adversarial training -- the domain
    classifier downstream of this layer is trained NORMALLY (gradient flows through
    unchanged for its own parameters), but the SAME loss's gradient into whatever comes
    BEFORE this layer (the encoder) is reversed, pushing the encoder to make domain
    prediction harder rather than easier."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    """Small MLP predicting domain (e.g. device id) from a pooled encoder embedding.
    Trained adversarially via GradientReversalLayer -- see train_mae_har.py's main()
    for how this is wired into the pretraining loop."""
    def __init__(self, encoder_dim, num_domains, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        return self.net(x)


class DomainLabeledDataset(torch.utils.data.Dataset):
    """
    Same loading logic as MultiTaskDataset (models/data.dataset.py), but additionally
    returns a domain label (e.g. device/environment/user id) alongside (csi, task_label).
    Needed because MultiTaskDataset.__getitem__ only returns (csi, task_label) -- no
    domain info -- and domain-adversarial pretraining needs a domain label per sample.

    domain_map should be built from train_id's domain column ONLY (not the full metadata
    table) -- the OOD splits' domains (e.g. held-out devices for test_cross_device) should
    never appear in this label space, since the adversarial classifier's job is to fail to
    distinguish among the domains seen DURING pretraining, not to have OOD domain classes
    defined at all.
    """
    def __init__(self, meta_df, data_root, task, label_map, domain_column, domain_map):
        self.meta = meta_df.reset_index(drop=True)
        self.data_root = data_root
        self.task = task
        self.label_map = label_map
        self.domain_column = domain_column
        self.domain_map = domain_map

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        h5_path = os.path.join(self.data_root, self.task, row["file_path"].lstrip("./"))
        csi = load_and_normalize_csi(h5_path)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.label_map[row["label"]], dtype=torch.long)
        domain_value = row[self.domain_column]
        if domain_value not in self.domain_map:
            raise KeyError(
                f"domain value {domain_value!r} (column={self.domain_column!r}) not in "
                f"domain_map -- this should only be built from train_id, so seeing an "
                f"unexpected value here suggests train_df and domain_map were built from "
                f"different data.")
        domain = torch.tensor(self.domain_map[domain_value], dtype=torch.long)
        return csi, label, domain


# ── Open-set identity evaluation (for splits containing identities never seen in
# train_id, e.g. test_cross_device/test_cross_user -- see main()'s OOD-loader
# construction for why these can't use the closed-set label_map classifier eval) ──
class RawIdentityDataset(torch.utils.data.Dataset):
    """Like MultiTaskDataset, but returns the raw identity STRING (e.g. 'U02') instead of
    an int class index via label_map -- works for identities never seen in train_id."""
    def __init__(self, df, data_root, task):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.task = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        h5_path = os.path.join(self.data_root, self.task, row["file_path"].lstrip("./"))
        csi = load_and_normalize_csi(h5_path)
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)
        return csi, row["label"]  # identity STRING, not an int


@torch.no_grad()
def get_features_with_identity(model, loader, layer, device, padded_h, padded_w):
    """Same as get_features(), but the loader yields (csi, identity_str) -- see
    RawIdentityDataset. Returns (feats, identities_ndarray)."""
    model.eval()
    feats, identities = [], []
    for csi, ident in loader:
        csi = pad_csi(csi.to(device), padded_h, padded_w)
        emb = model.extract_layer_embeddings(csi, [layer])
        feats.append(emb[layer].cpu())
        identities.extend(ident)
    return torch.cat(feats), np.array(identities)


def rank1_retrieval_accuracy(feats, identities, seed):
    """Randomly splits (feats, identities) into gallery/query (stratified so every identity
    with >=2 samples appears in both halves), then for each query, retrieves its nearest
    gallery neighbor (cosine similarity) and checks identity match. Works identically for
    identities seen or never seen in train_id -- pure similarity search, no classifier.
    Returns (accuracy, n_query, n_identities); (None, 0, 0) if no identity has >=2 samples.
    CAVEAT: if the split contains only ONE identity, accuracy is trivially 1.0 regardless
    of embedding quality (every gallery member necessarily shares the query's identity) --
    not a bug, just uninformative for single-identity splits like test_cross_user."""
    rng = np.random.RandomState(seed)
    gallery_idx, query_idx = [], []
    for ident in np.unique(identities):
        idxs = np.where(identities == ident)[0]
        if len(idxs) < 2:
            continue
        rng.shuffle(idxs)
        split = len(idxs) // 2
        gallery_idx.extend(idxs[:max(split, 1)])
        query_idx.extend(idxs[max(split, 1):])
    gallery_idx, query_idx = np.array(gallery_idx), np.array(query_idx)
    if len(query_idx) == 0:
        return None, 0, 0

    gallery_feats, gallery_ids = feats[gallery_idx], identities[gallery_idx]
    query_feats, query_ids = feats[query_idx], identities[query_idx]
    g_norm = gallery_feats / (gallery_feats.norm(dim=1, keepdim=True) + 1e-8)
    q_norm = query_feats / (query_feats.norm(dim=1, keepdim=True) + 1e-8)
    sim = q_norm @ g_norm.T
    nearest = sim.argmax(dim=1).numpy()
    preds = gallery_ids[nearest]
    acc = (preds == query_ids).mean()
    return acc, len(query_idx), len(np.unique(identities))


def verification_auc_eer(feats, identities, n_pairs, seed):
    """Samples n_pairs//2 same-identity + n_pairs//2 different-identity pairs, scores by
    cosine similarity, computes AUC and EER. Returns (None, None, n) if <2 identities are
    present (can't form same-identity pairs) or all identities are singletons."""
    rng = np.random.RandomState(seed)
    n = len(identities)
    feats_norm = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
    unique_ids, counts = np.unique(identities, return_counts=True)
    same_id_pool = unique_ids[counts >= 2]
    if len(same_id_pool) == 0:
        return None, None, 0

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
            continue
        diff_pairs.append((i, j))

    pairs = same_pairs + diff_pairs
    labels = np.array([1] * len(same_pairs) + [0] * len(diff_pairs))
    if len(np.unique(labels)) < 2:
        return None, None, len(pairs)
    scores = np.array([(feats_norm[i] @ feats_norm[j]).item() for i, j in pairs])
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return auc, eer, len(pairs)


# ── Patch-size padding utilities ──────────────────────────────────────────────
def compute_padded_size(orig_size, patch_size):
    """Smallest size >= orig_size that's an exact multiple of patch_size.
    Needed because PatchEmbedding (Conv2d, stride=kernel_size) and patchify()
    (tensor.unfold) both silently floor-divide otherwise, dropping the
    remainder subcarriers/timesteps instead of erroring."""
    return ((orig_size + patch_size - 1) // patch_size) * patch_size


def pad_csi(x, padded_h, padded_w):
    """x: [B,1,H,W] -> zero-pad on the bottom/right to [B,1,padded_h,padded_w]."""
    _, _, H, W = x.shape
    pad_h = padded_h - H
    pad_w = padded_w - W
    if pad_h == 0 and pad_w == 0:
        return x
    # F.pad order for a 4D tensor pads the LAST dim first: (w_left, w_right, h_top, h_bottom)
    return F.pad(x, (0, pad_w, 0, pad_h))


def check_attention_memory(num_patches, batch_size, encoder_depth,
                            encoder_heads=ENCODER_HEADS,
                            budget_gb=20.0, skip_check=False):
    """
    Pre-flight check for the naive (non-flash) MultiHeadAttention in models/vit.py.
    Its attention score tensor is [B, heads, N, N] in fp32 -- O(N^2) memory that
    silently OOMs deep into a training run otherwise. This hard-stops with a
    suggested safe batch_size instead.

    IMPORTANT (fixed): the encoder has no gradient checkpointing, so during backward
    ALL encoder_depth layers' attention score tensors are retained simultaneously --
    not just one. The gate below therefore compares the CUMULATIVE estimate
    (per_layer * encoder_depth) against budget_gb, not just a single layer's estimate.
    An earlier version of this check compared only the per-layer number, which let
    patch=7 (11.4 GB/layer, but 68.6 GB cumulative across 6 layers) pass and then
    OOM in production. Do not revert to the per-layer-only comparison.

    budget_gb: ceiling for the CUMULATIVE (all retained layers) attention-score memory.
    Left well under typical 32GB GPUs since model weights, optimizer states, activations
    for the rest of the network, and CUDA/framework overhead also need headroom.
    """
    N = num_patches
    bytes_per_layer = batch_size * encoder_heads * N * N * 4  # fp32
    gb_per_layer = bytes_per_layer / 1024**3
    gb_cumulative = gb_per_layer * encoder_depth

    print(f"[mem-check] num_patches={N}  batch_size={batch_size}  encoder_depth={encoder_depth}  "
          f"attention-score memory: {gb_per_layer:.2f} GB/layer, "
          f"{gb_cumulative:.2f} GB cumulative (all layers retained for backward)")

    if gb_cumulative > budget_gb:
        # Solve for the largest batch_size keeping cumulative memory under budget.
        max_safe_batch = max(1, int(budget_gb * 1024**3 / (encoder_heads * N * N * 4 * encoder_depth)))
        msg = (
            f"\n[mem-check] REFUSING TO START: projected CUMULATIVE attention memory "
            f"({gb_cumulative:.1f} GB across {encoder_depth} layers) exceeds the safety budget ({budget_gb:.1f} GB).\n"
            f"  num_patches={N} at patch size given is too large for batch_size={batch_size}, "
            f"encoder_depth={encoder_depth} with this naive (non-flash) attention implementation.\n"
            f"  Suggested max safe batch_size for this config: ~{max_safe_batch}\n"
            f"  Options:\n"
            f"    1. Re-run with --batch_size {max_safe_batch} (or lower)\n"
            f"    2. Use a larger patch size (fewer patches -> quadratically less attention memory)\n"
            f"    3. Pass --skip_mem_check to bypass this check (not recommended --\n"
            f"       you will very likely hit a mid-run CUDA OOM instead)\n"
        )
        if skip_check:
            print(msg + "  [skip_mem_check=True] Proceeding anyway per user request.\n")
        else:
            print(msg)
            sys.exit(1)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_split(name, meta, label_map, splits_dir=None):
    """splits_dir defaults to the module-level SPLITS_DIR (HumanIdentification, in this
    file) if not given. The optional override param is kept only so this function's
    signature stays identical to train_mae_har.py's -- this file's own call sites don't
    pass it explicitly, since the default already points at the right task."""
    import json as _json
    splits_dir = splits_dir or SPLITS_DIR
    with open(f'{splits_dir}/{name}.json') as f:
        ids = set(_json.load(f))
    df = meta[meta['id'].isin(ids)].reset_index(drop=True)
    return MultiTaskDataset(df, DATA_ROOT, 'Multitask', label_map=label_map)

# ── Evaluation helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def get_features(model, loader, layer, device, padded_h, padded_w):
    model.eval()
    feats, labels = [], []
    for csi, y in loader:
        csi = pad_csi(csi.to(device), padded_h, padded_w)
        emb = model.extract_layer_embeddings(csi, [layer])
        feats.append(emb[layer].cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def compute_domain_shift_metrics(id_features, id_labels, ood_features, ood_labels):
    """
    Computes class-conditional centroid L2 distance and Cosine Similarity
    between In-Distribution (ID) and OOD feature representations.

    Rationale (from postdoc suggestion): classification accuracy is an INDIRECT probe
    of representation quality -- when accuracy drops, it's ambiguous whether that's
    because classes are locally confused or because the whole distribution shifted.
    This is a direct, non-parametric, purely geometric measure: for each class, take
    the ID centroid and OOD centroid (mean embedding), and measure how far apart /
    misaligned they are. No classifier training involved.
    """
    if hasattr(id_features, 'numpy'): id_features = id_features.numpy()
    if hasattr(id_labels, 'numpy'): id_labels = id_labels.numpy()
    if hasattr(ood_features, 'numpy'): ood_features = ood_features.numpy()
    if hasattr(ood_labels, 'numpy'): ood_labels = ood_labels.numpy()

    unique_classes = np.unique(id_labels)
    l2_distances = []
    cosine_similarities = []

    for cls in unique_classes:
        if cls not in ood_labels:
            continue

        cls_id_features = id_features[id_labels == cls]
        cls_ood_features = ood_features[ood_labels == cls]

        centroid_id = np.mean(cls_id_features, axis=0)
        centroid_ood = np.mean(cls_ood_features, axis=0)

        l2_dist = np.linalg.norm(centroid_id - centroid_ood)
        l2_distances.append(l2_dist)

        dot_product = np.dot(centroid_id, centroid_ood)
        norm_id = np.linalg.norm(centroid_id)
        norm_ood = np.linalg.norm(centroid_ood)

        cos_sim = dot_product / (norm_id * norm_ood + 1e-8)
        cosine_similarities.append(cos_sim)

    if not l2_distances:
        return {"centroid_l2_dist": 0.0, "centroid_cos_sim": 0.0}

    return {
        "centroid_l2_dist": float(np.mean(l2_distances)),
        "centroid_cos_sim": float(np.mean(cosine_similarities))
    }


def knn_eval(train_feats, train_labels, eval_feats, eval_labels, k=10):
    mu  = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0,  keepdim=True) + 1e-8
    tf = (train_feats - mu) / std
    ef = (eval_feats  - mu) / std
    tf_n = tf / (tf.norm(dim=1, keepdim=True) + 1e-8)
    ef_n = ef / (ef.norm(dim=1, keepdim=True) + 1e-8)
    sim  = ef_n @ tf_n.T  # [N_eval, N_train]
    topk = sim.topk(k, dim=1).indices
    preds = train_labels[topk].mode(dim=1).values
    acc = (preds == eval_labels).float().mean().item()
    f1  = f1_score(eval_labels.numpy(), preds.numpy(),
                   average='weighted', zero_division=0)
    return acc, f1


def linear_probe_eval(train_feats, train_labels, eval_feats, eval_labels,
                      num_classes, device, epochs=50):
    mu  = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0,  keepdim=True) + 1e-8
    tf = (train_feats - mu) / std
    ef = (eval_feats  - mu) / std

    head  = nn.Linear(tf.shape[1], num_classes).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    ds  = torch.utils.data.TensorDataset(tf, train_labels)
    ldr = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    for _ in range(epochs):
        head.train()
        for xb, yb in ldr:
            loss = crit(head(xb.to(device)), yb.to(device))
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

    head.eval()
    with torch.no_grad():
        preds = head(ef.to(device)).argmax(1).cpu()
    acc = (preds == eval_labels).float().mean().item()
    f1  = f1_score(eval_labels.numpy(), preds.numpy(),
                   average='weighted', zero_division=0)
    return acc, f1


class MAEDownstreamHead(nn.Module):
    """
    Downstream evaluation/fine-tuning wrapper for MAE/MAEv2 backbones.
    """
    def __init__(self, pretrained_model, num_classes, layer=None,
                 hidden_dim=256, unfreeze_last_n_layers=0, norm_type='layernorm',
                 activation='relu'):
        super().__init__()
        import copy
        backbone = copy.deepcopy(pretrained_model)  # never mutate the caller's model

        self.patch_embedding   = backbone.patch_embedding
        self.encoder_pos_embed = backbone.encoder_pos_embed
        self.encoder_blocks    = backbone.encoder_blocks
        self.encoder_norm      = backbone.encoder_norm
        self.layer = layer or len(self.encoder_blocks.layers)  # default: deepest layer
        self.norm_type = norm_type

        if norm_type == 'layernorm':
            norm_layer = nn.LayerNorm(hidden_dim)
        elif norm_type == 'batchnorm':
            norm_layer = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'none':
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"norm_type must be 'layernorm', 'batchnorm', or 'none', got {norm_type!r}")

        if activation == 'relu':
            act_layer = nn.ReLU()
        elif activation == 'none':
            act_layer = nn.Identity()
        else:
            raise ValueError(f"activation must be 'relu' or 'none', got {activation!r}")
        self.activation = activation

        self.mlp_head = nn.Sequential(
            nn.Linear(backbone.encoder_dim, hidden_dim),
            norm_layer,
            act_layer,
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.set_backbone_trainable(unfreeze_last_n_layers)

        self._pretrained_backbone_state = {
            name: p.detach().clone()
            for name, p in self._backbone_named_parameters()
        }

    def _backbone_named_parameters(self):
        for name, p in self.patch_embedding.named_parameters():
            yield f"patch_embedding.{name}", p
        yield "encoder_pos_embed", self.encoder_pos_embed
        for name, p in self.encoder_blocks.named_parameters():
            yield f"encoder_blocks.{name}", p

    def l2sp_penalty(self):
        device = self.encoder_pos_embed.device
        penalty = torch.zeros((), device=device)
        for name, p in self._backbone_named_parameters():
            if p.requires_grad:
                penalty = penalty + (p - self._pretrained_backbone_state[name]).pow(2).sum()
        return penalty

    def set_backbone_trainable(self, unfreeze_last_n_layers):
        for p in self.patch_embedding.parameters():
            p.requires_grad = False
        self.encoder_pos_embed.requires_grad = False
        for block in self.encoder_blocks.layers:
            for p in block.parameters():
                p.requires_grad = False

        if unfreeze_last_n_layers is None:
            for p in self.patch_embedding.parameters():
                p.requires_grad = True
            self.encoder_pos_embed.requires_grad = True
            for block in self.encoder_blocks.layers:
                for p in block.parameters():
                    p.requires_grad = True
        elif unfreeze_last_n_layers > 0:
            for block in list(self.encoder_blocks.layers)[-unfreeze_last_n_layers:]:
                for p in block.parameters():
                    p.requires_grad = True

        status = ("frozen" if unfreeze_last_n_layers == 0
                  else "fully unfrozen" if unfreeze_last_n_layers is None
                  else f"last {unfreeze_last_n_layers} block(s) unfrozen")
        print(f"[MAEDownstreamHead] backbone: {status}, probing layer {self.layer}")

    def backbone_parameters(self):
        params = list(self.patch_embedding.parameters()) + [self.encoder_pos_embed] + \
                 list(self.encoder_blocks.parameters())
        return [p for p in params if p.requires_grad]

    def forward(self, x):
        h = self.patch_embedding(x) + self.encoder_pos_embed
        for i, block in enumerate(self.encoder_blocks.layers):
            h = block(h)
            if (i + 1) == self.layer:
                h = self.encoder_norm(h)
                break
        features = h.mean(dim=1)  # [B, encoder_dim]
        return self.mlp_head(features)


def finetune_eval(model, train_loader, eval_loaders, num_classes, layer, device,
                  padded_h, padded_w, epochs=25, backbone_lr=1e-5, head_lr=1e-3,
                  unfreeze_last_n_layers=None, hidden_dim=256, norm_type='layernorm',
                  activation='relu',
                  eval_every=5, early_stop_patience=5, early_stop_split='ood_avg',
                  monitor_metric='loss', use_plateau_scheduler=False,
                  l2sp_lambda=0.0, verbose=True):
    """
    Full (or partial) fine-tuning of the pretrained encoder + a downstream MLP head,
    evaluated end-to-end -- NOT a frozen-feature probe (see mlp_probe_eval for that).

    Returns (results, history):
        results -- {split_name: {'acc': ..., 'f1': ..., 'loss': ...}}, evaluated at the
                   BEST epoch found (by early_stop_split + monitor_metric), not
                   necessarily the last epoch trained.
        history -- list of dicts, one per evaluation.
    """
    import copy

    wrapper = MAEDownstreamHead(model, num_classes, layer=layer, hidden_dim=hidden_dim,
                                unfreeze_last_n_layers=unfreeze_last_n_layers,
                                norm_type=norm_type, activation=activation).to(device)

    param_groups = [{'params': wrapper.mlp_head.parameters(), 'lr': head_lr, 'weight_decay': 0.05}]
    backbone_params = wrapper.backbone_parameters()
    if backbone_params:
        backbone_wd = 0.0 if l2sp_lambda > 0 else 0.05
        param_groups.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_wd})
    optim = torch.optim.AdamW(param_groups)

    plateau_mode = 'min' if monitor_metric == 'loss' else 'max'
    if use_plateau_scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode=plateau_mode, factor=0.5, patience=2)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    def evaluate_all_splits():
        wrapper.eval()
        split_results = {}
        with torch.no_grad():
            for split_name, loader in eval_loaders.items():
                preds_all, labels_all, losses_all, n_all = [], [], 0.0, 0
                for csi, y in loader:
                    csi = pad_csi(csi.to(device), padded_h, padded_w)
                    y_dev = y.to(device)
                    logits = wrapper(csi)
                    batch_loss = crit(logits, y_dev)
                    losses_all += batch_loss.item() * y.shape[0]
                    n_all += y.shape[0]
                    preds_all.append(logits.argmax(1).cpu())
                    labels_all.append(y)
                preds_all = torch.cat(preds_all)
                labels_all = torch.cat(labels_all)
                acc = (preds_all == labels_all).float().mean().item()
                f1 = f1_score(labels_all.numpy(), preds_all.numpy(), average='weighted', zero_division=0)
                avg_loss = losses_all / n_all
                split_results[split_name] = {'acc': acc, 'f1': f1, 'loss': avg_loss}
        wrapper.train()
        return split_results

    def compute_monitored_metric(split_results):
        if early_stop_split == 'ood_avg':
            ood_splits = [s for s in split_results if s != 'test_id']
            if not ood_splits:
                raise ValueError("No OOD splits found in eval_loaders for 'ood_avg' -- "
                                 "pass an explicit split name via early_stop_split.")
            return sum(split_results[s][monitor_metric] for s in ood_splits) / len(ood_splits)
        if early_stop_split not in split_results:
            raise ValueError(f"early_stop_split={early_stop_split!r} not in eval_loaders "
                             f"keys: {list(split_results.keys())}")
        return split_results[early_stop_split][monitor_metric]

    def is_improvement(monitored, best):
        return monitored < best if monitor_metric == 'loss' else monitored > best

    best_metric = float('inf') if monitor_metric == 'loss' else -float('inf')
    best_state = None
    best_epoch = 0
    evals_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        wrapper.train()
        for csi, y in train_loader:
            csi = pad_csi(csi.to(device), padded_h, padded_w)
            y = y.to(device)
            loss = crit(wrapper(csi), y)
            if l2sp_lambda > 0:
                loss = loss + l2sp_lambda * wrapper.l2sp_penalty()
            optim.zero_grad(); loss.backward(); optim.step()
        if not use_plateau_scheduler:
            sched.step()

        if epoch % eval_every == 0 or epoch == epochs:
            split_results = evaluate_all_splits()
            monitored = compute_monitored_metric(split_results)
            history.append({'epoch': epoch, 'monitored_metric': monitored, **split_results})

            is_best = is_improvement(monitored, best_metric)
            if verbose:
                lr_now = optim.param_groups[-1]['lr']
                print(f"[finetune_eval] epoch {epoch}/{epochs}  "
                      f"{early_stop_split} {monitor_metric}={monitored:.4f}  lr={lr_now:.2e}"
                      + ("  <- best" if is_best else ""))

            if use_plateau_scheduler:
                sched.step(monitored)

            if is_best:
                best_metric = monitored
                best_state = copy.deepcopy(wrapper.state_dict())
                best_epoch = epoch
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= early_stop_patience:
                    if verbose:
                        print(f"[finetune_eval] early stopping at epoch {epoch} "
                              f"(no improvement in {early_stop_patience} evals since "
                              f"epoch {best_epoch}, best {early_stop_split} "
                              f"{monitor_metric}={best_metric:.4f})")
                    break

    if best_state is not None:
        wrapper.load_state_dict(best_state)
        if verbose:
            print(f"[finetune_eval] reporting results from epoch {best_epoch} "
                  f"(best {early_stop_split} {monitor_metric}={best_metric:.4f}), "
                  f"not the final epoch trained")
    results = evaluate_all_splits()
    return results, history


def mlp_probe_eval(train_feats, train_labels, eval_feats, eval_labels,
                   num_classes, device, epochs=50, hidden_dim=128):
    """
    Same protocol as linear_probe_eval (identical normalization, optimizer, schedule,
    epoch count, batch size) but with a 1-hidden-layer MLP instead of nn.Linear.
    """
    mu  = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0,  keepdim=True) + 1e-8
    tf = (train_feats - mu) / std
    ef = (eval_feats  - mu) / std

    head = nn.Sequential(
        nn.Linear(tf.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    ).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    ds  = torch.utils.data.TensorDataset(tf, train_labels)
    ldr = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    for _ in range(epochs):
        head.train()
        for xb, yb in ldr:
            loss = crit(head(xb.to(device)), yb.to(device))
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

    head.eval()
    with torch.no_grad():
        preds = head(ef.to(device)).argmax(1).cpu()
    acc = (preds == eval_labels).float().mean().item()
    f1  = f1_score(eval_labels.numpy(), preds.numpy(),
                   average='weighted', zero_division=0)
    return acc, f1


# ── Attentive probing (postdoc suggestion, V-JEPA style) ──────────────────────
def extract_sequence_embeddings(model, x, layer):
    """
    Same layer-by-layer walk as model.extract_layer_embeddings() (models/mae.py),
    but returns the UNPOOLED per-patch-token sequence [B, N, encoder_dim] at the given
    layer instead of the mean-pooled [B, encoder_dim] vector.
    """
    h = model.patch_embedding(x) + model.encoder_pos_embed
    for i, block in enumerate(model.encoder_blocks.layers):
        h = block(h)
        if (i + 1) == layer:
            return model.encoder_norm(h)  # [B, N, encoder_dim] -- NOT pooled
    raise ValueError(f"layer={layer} exceeds encoder depth {len(model.encoder_blocks.layers)}")


@torch.no_grad()
def get_sequence_features(model, loader, layer, device, padded_h, padded_w):
    """Sequence-level counterpart to get_features()."""
    model.eval()
    feats, labels = [], []
    for csi, y in loader:
        csi = pad_csi(csi.to(device), padded_h, padded_w)
        seq = extract_sequence_embeddings(model, csi, layer)
        feats.append(seq.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


class AttentiveProbe(nn.Module):
    """
    Attentive probe (V-JEPA style, arxiv.org/abs/2404.08471): a single learnable query
    vector cross-attends over the full, UNPOOLED patch-token sequence to extract a
    task-relevant summary.
    """
    def __init__(self, encoder_dim, num_classes, num_heads=4, heavyweight=False,
                ff_dim=None, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, encoder_dim) * 0.02)
        self.attn = nn.MultiheadAttention(encoder_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.heavyweight = heavyweight
        if heavyweight:
            ff_dim = ff_dim or encoder_dim * 4
            self.norm1 = nn.LayerNorm(encoder_dim)
            self.ffn = nn.Sequential(
                nn.Linear(encoder_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ff_dim, encoder_dim), nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(encoder_dim)
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, sequence):
        """sequence: [B, N, encoder_dim], NOT pooled (from get_sequence_features())."""
        B = sequence.shape[0]
        q = self.query.expand(B, -1, -1)                    # [B, 1, D]
        attended, _ = self.attn(q, sequence, sequence)       # [B, 1, D]
        attended = self.attn_dropout(attended.squeeze(1))    # [B, D]
        if self.heavyweight:
            attended = self.norm1(attended)
            attended = attended + self.ffn(attended)
            attended = self.norm2(attended)
        return self.classifier(attended)


def attentive_probe_eval(train_feats, train_labels, eval_feats, eval_labels,
                         num_classes, device, epochs=50, heavyweight=False,
                         num_heads=4, dropout=0.1):
    """
    train_feats/eval_feats: [N_samples, N_tokens, encoder_dim] -- UNPOOLED sequences.
    """
    mu = train_feats.mean(dim=(0, 1), keepdim=True)
    std = train_feats.std(dim=(0, 1), keepdim=True) + 1e-8
    tf = (train_feats - mu) / std
    ef = (eval_feats - mu) / std

    encoder_dim = tf.shape[-1]
    probe = AttentiveProbe(encoder_dim, num_classes, num_heads=num_heads,
                           heavyweight=heavyweight, dropout=dropout).to(device)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(tf, train_labels)
    ldr = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    for _ in range(epochs):
        probe.train()
        for xb, yb in ldr:
            loss = crit(probe(xb.to(device)), yb.to(device))
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(ef.to(device)).argmax(1).cpu()
    acc = (preds == eval_labels).float().mean().item()
    f1 = f1_score(eval_labels.numpy(), preds.numpy(), average='weighted', zero_division=0)
    return acc, f1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',        type=int,   default=300)
    parser.add_argument('--mask_ratio',    type=float, default=0.75)
    parser.add_argument('--encoder_depth', type=int,   default=6)
    parser.add_argument('--encoder_dim',   type=int,   default=128)
    parser.add_argument('--decoder_dim',   type=int,   default=64)
    parser.add_argument('--batch_size',    type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=1.5e-4)
    parser.add_argument('--eval_layers',   type=str,   default='1,3,6,9,12')
    parser.add_argument('--eval_every',    type=int,   default=50)
    parser.add_argument('--mask_strategy', type=str, default='random', choices=['random','time','freq','mixed','2d'])
    parser.add_argument('--patch_h',       type=int,   default=29)
    parser.add_argument('--patch_w',       type=int,   default=25)
    parser.add_argument('--seed',          type=int,   default=42)

    # ── Split-selection knobs (added) ─────────────────────────────────────
    parser.add_argument('--train_split', type=str, default='train_id',
                         help="Which split JSON (under SPLITS_DIR) to train on. Use "
                              "'train_hp' for hyperparameter search, 'train_final' "
                              "only for the final reported run. Default ('train_id') "
                              "is the ORIGINAL, session-leaky split (confirmed 100%% "
                              "train/test session overlap for HID) -- kept only for "
                              "backward compatibility; do not use it for paper numbers.")
    parser.add_argument('--eval_splits', type=str,
                         default='test_id,test_cross_device,test_cross_env,test_cross_user',
                         help="Comma-separated split names (under SPLITS_DIR) to "
                              "evaluate on each --eval_every checkpoint. For HP search: "
                              "'val_hp,test_cross_device,test_cross_env,test_cross_user' "
                              "(never test_final or test_id here). For the final reported "
                              "run: 'test_final,test_cross_device,test_cross_env,test_cross_user'. "
                              "Each named split is checked against label_map and routed to "
                              "closed-set (KNN/LP/MLP) or open-set (Rank-1/AUC/EER) eval "
                              "accordingly -- see module docstring.")

    parser.add_argument('--skip_mem_check', action='store_true',
                         help='Bypass the pre-flight attention-memory safety check (not recommended)')
    parser.add_argument('--mem_budget_gb', type=float, default=20.0,
                         help='Safety budget (GB) for the cumulative attention memory before hard-stopping')
    parser.add_argument('--domain_adv_lambda', type=float, default=0.0,
                         help='Domain-adversarial pretraining strength (DANN, Ganin & Lempitsky 2016). '
                              '0.0 (default) = off. When > 0, adds a domain classifier on the FULL '
                              '(unmasked) pooled encoder embedding, trained via gradient reversal to push '
                              'the encoder toward domain-invariant representations.')
    parser.add_argument('--domain_adv_column', default='device', choices=['device', 'environment', 'user'],
                         help='Which metadata column to use as the domain-adversarial target')
    parser.add_argument('--domain_adv_layer', type=int, default=None,
                         help='Which encoder layer to attach the domain classifier to (default: '
                              'encoder_depth, i.e. the deepest layer)')
    parser.add_argument('--domain_adv_hidden_dim', type=int, default=128)
    parser.add_argument('--domain_adv_optimizer', default='adamw', choices=['adamw', 'sgd'],
                         help="Optimizer for the domain classifier's own parameters.")
    parser.add_argument('--domain_adv_lr', type=float, default=0.01,
                         help="Learning rate for the domain classifier's own optimizer")
    parser.add_argument('--run_attentive_probe', action='store_true',
                         help='Also run lightweight + heavyweight attentive probing at every eval checkpoint.')
    parser.add_argument('--attentive_probe_epochs', type=int, default=50)
    parser.add_argument('--resume_from', default=None,
                         help='Path to a _best.pt checkpoint to resume from. Loads model_state '
                              'and continues epoch numbering from ckpt["epoch"]+1.')
    parser.add_argument('--n_verification_pairs', type=int, default=5000,
                         help='Number of same/different-identity pairs sampled for the '
                              'open-set verification AUC/EER metric.')
    args = parser.parse_args()

    # Seed control for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    eval_layers = [int(x) for x in args.eval_layers.split(',')]
    valid_eval_layers = [l for l in eval_layers if l <= args.encoder_depth]
    if len(valid_eval_layers) < len(eval_layers):
        dropped = [l for l in eval_layers if l > args.encoder_depth]
        print(f"[eval-layers] WARNING: dropping requested eval layer(s) {dropped} -- "
              f"encoder_depth={args.encoder_depth} only has layers 1..{args.encoder_depth}. "
              f"Evaluating layers {valid_eval_layers} instead. Pass --eval_layers explicitly "
              f"to silence this (e.g. --eval_layers 1,3,6 for encoder_depth=6).")
    if not valid_eval_layers:
        valid_eval_layers = [args.encoder_depth]
        print(f"[eval-layers] No requested layers were valid -- falling back to "
              f"the deepest layer only: {valid_eval_layers}")
    eval_layers = valid_eval_layers

    exp_name = (f"mae_hid_ep{args.epochs}_mask{args.mask_ratio}_strategy{args.mask_strategy}_ph{args.patch_h}pw{args.patch_w}_seed{args.seed}"
                f"_enc{args.encoder_depth}_dim{args.encoder_dim}_bs{args.batch_size}")
    print(f"\nExperiment: {exp_name}  (task={TASK})")

    # ── Patch-size padding: compute target shape and warn if padding is added ──
    padded_h = compute_padded_size(RAW_IMG_H, args.patch_h)
    padded_w = compute_padded_size(RAW_IMG_W, args.patch_w)
    n_h, n_w = padded_h // args.patch_h, padded_w // args.patch_w
    num_patches = n_h * n_w
    if (padded_h, padded_w) != (RAW_IMG_H, RAW_IMG_W):
        print(f"[patch-pad] patch_h={args.patch_h}, patch_w={args.patch_w} do not evenly divide "
              f"{RAW_IMG_H}x{RAW_IMG_W} -- zero-padding input to {padded_h}x{padded_w} "
              f"(+{padded_h - RAW_IMG_H} subcarriers, +{padded_w - RAW_IMG_W} timesteps). "
              f"num_patches={num_patches} (n_h={n_h}, n_w={n_w})")
    else:
        print(f"[patch-pad] patch_h={args.patch_h}, patch_w={args.patch_w} evenly divide "
              f"{RAW_IMG_H}x{RAW_IMG_W}, no padding needed. num_patches={num_patches}")

    # ── Pre-flight memory check (naive attention is O(N^2) -- fail fast, not mid-run) ──
    check_attention_memory(num_patches, args.batch_size, args.encoder_depth,
                            budget_gb=args.mem_budget_gb, skip_check=args.skip_mem_check)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ──────────────────────────────────────────────────────────────
    meta = pd.read_csv(META_PATH)
    import json as _json
    with open(f'{SPLITS_DIR}/{args.train_split}.json') as f:      # <-- CHANGED (was hardcoded 'train_id')
        train_ids = set(_json.load(f))
    print(f"[split] training on: {args.train_split}")             # <-- CHANGED (new)
    train_df  = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    num_classes = len(label_map)
    print(f"label_map: {label_map}  num_classes: {num_classes}")

    train_ds = MultiTaskDataset(train_df, DATA_ROOT, 'Multitask', label_map=label_map)
    train_feat_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4)

    domain_map = None
    if args.domain_adv_lambda > 0:
        domain_values = sorted(train_df[args.domain_adv_column].unique(), key=str)
        domain_map = {v: i for i, v in enumerate(domain_values)}
        num_domains = len(domain_map)
        print(f"[domain-adv] column={args.domain_adv_column!r}  num_domains={num_domains}  "
              f"domain_map={domain_map}")
        pretrain_ds = DomainLabeledDataset(train_df, DATA_ROOT, 'Multitask', label_map,
                                           args.domain_adv_column, domain_map)
        pretrain_loader = DataLoader(pretrain_ds, batch_size=args.batch_size,
                                     shuffle=True, num_workers=4, pin_memory=True)
    else:
        pretrain_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                     shuffle=True, num_workers=4, pin_memory=True)

    # ── Eval loaders: closed-set (label_map) vs. open-set (unknown identities) ──
    # IMPORTANT (HumanIdentification-specific): label_map is a CLOSED set built from
    # args.train_split's identities only. Some eval splits contain identities that never
    # appear there (e.g. test_cross_user is entirely 'U02', who may have zero train-split
    # samples) -- MultiTaskDataset.__getitem__ does label_map[row["label"]], which raises
    # KeyError for any such identity. A classifier with a fixed output layer built from
    # label_map is structurally incapable of predicting an identity it was never given an
    # output slot for -- so splits with unknown identities go through a DIFFERENT eval path:
    # open-set retrieval/verification (rank1_retrieval_accuracy, verification_auc_eer),
    # which never reference label_map and work correctly regardless of whether the identity
    # was seen during pretraining.
    #
    # This check now runs over args.eval_splits (whatever the caller passed), not the
    # hardcoded OOD_SPLITS -- val_hp/test_final are expected to be closed-set in practice
    # (they're trial-level subsets of train_hp's/train_final's own identity pool), but the
    # check is applied uniformly rather than assumed.
    eval_split_names = args.eval_splits.split(',')                 # <-- CHANGED (was hardcoded OOD_SPLITS)
    print(f"[split] evaluating on: {eval_split_names}")            # <-- CHANGED (new)
    ood_loaders = {}
    openset_loaders = {}
    for sname in eval_split_names:                                 # <-- CHANGED (was `for sname in OOD_SPLITS:`)
        with open(f'{SPLITS_DIR}/{sname}.json') as f:
            split_ids = set(_json.load(f))
        split_df = meta[meta['id'].isin(split_ids)]
        split_identities = set(split_df['label'].unique())
        unknown_identities = split_identities - set(label_map.keys())
        if unknown_identities:
            openset_ds = RawIdentityDataset(split_df, DATA_ROOT, 'Multitask')
            openset_loaders[sname] = DataLoader(openset_ds, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
            print(f"  {sname}: {len(split_df)} samples -- OPEN-SET eval (contains identities "
                  f"not in {args.train_split}'s label_map: {sorted(unknown_identities)}). Uses "
                  f"Rank-1 retrieval + verification AUC/EER instead of KNN/LP/MLP.")
            continue
        ds = load_split(sname, meta, label_map)
        ood_loaders[sname] = DataLoader(ds, batch_size=args.batch_size,
                                        shuffle=False, num_workers=4)
        print(f"  {sname}: {len(ds)} samples")

    # Model -- constructed with the PADDED image size, not the raw 232x500
    if args.mask_strategy == 'random':
        model = MAE(
            in_channels=1, img_h=padded_h, img_w=padded_w,
            patch_h=args.patch_h, patch_w=args.patch_w,
            encoder_dim=args.encoder_dim,
            encoder_ff_dim=args.encoder_dim * 4,
            encoder_heads=ENCODER_HEADS, encoder_depth=args.encoder_depth,
            decoder_dim=args.decoder_dim,
            decoder_heads=2, decoder_depth=2,
            mask_ratio=args.mask_ratio
        ).to(device)
    else:
        model = MAEv2(
            in_channels=1, img_h=padded_h, img_w=padded_w,
            patch_h=args.patch_h, patch_w=args.patch_w,
            encoder_dim=args.encoder_dim,
            encoder_ff_dim=args.encoder_dim * 4,
            encoder_heads=ENCODER_HEADS, encoder_depth=args.encoder_depth,
            decoder_dim=args.decoder_dim,
            decoder_heads=2, decoder_depth=2,
            mask_ratio=args.mask_ratio,
            mask_strategy=args.mask_strategy
        ).to(device)
    print(f"MAE params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Resume from a previous checkpoint, if given ───────────────────────────
    start_epoch = 1
    resumed_best_loss = None
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(resume_ckpt['model_state'])
        start_epoch = resume_ckpt['epoch'] + 1
        resumed_best_loss = resume_ckpt['loss']
        print(f"[resume] loaded {args.resume_from}: was at epoch {resume_ckpt['epoch']}, "
              f"loss={resumed_best_loss:.4f} -- continuing from epoch {start_epoch}")
        if start_epoch > args.epochs:
            print(f"[resume] start_epoch ({start_epoch}) > --epochs ({args.epochs}) -- "
                  f"this checkpoint already reached or passed the target, nothing to do. "
                  f"Increase --epochs if you want to train further.")
            sys.exit(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"[resume] fast-forwarded LR schedule to epoch {start_epoch} "
              f"(lr={scheduler.get_last_lr()[0]:.2e})")

    # ── Domain-adversarial setup (off unless --domain_adv_lambda > 0) ────────
    domain_clf, grl, domain_optimizer = None, None, None
    if args.domain_adv_lambda > 0:
        domain_adv_layer = args.domain_adv_layer or args.encoder_depth
        domain_clf = DomainClassifier(args.encoder_dim, num_domains,
                                      hidden_dim=args.domain_adv_hidden_dim).to(device)
        grl = GradientReversalLayer(lambda_=args.domain_adv_lambda)
        if args.domain_adv_optimizer == 'sgd':
            domain_optimizer = torch.optim.SGD(domain_clf.parameters(), lr=args.domain_adv_lr, momentum=0.9)
        else:
            domain_optimizer = torch.optim.AdamW(domain_clf.parameters(), lr=args.domain_adv_lr)
        print(f"[domain-adv] classifier attached at layer {domain_adv_layer}, "
              f"lambda={args.domain_adv_lambda}, optimizer={args.domain_adv_optimizer}")
        print(f"[domain-adv] WATCH the printed domain accuracy below -- it should trend "
              f"DOWN toward chance level (~{1/num_domains:.3f}). If it stays high, the "
              f"adversarial objective isn't working (see --domain_adv_lambda's help text).")

    saved_args = vars(args).copy()
    saved_args['padded_h'] = padded_h
    saved_args['padded_w'] = padded_w
    saved_args['num_patches'] = num_patches
    if domain_map is not None:
        saved_args['domain_map'] = {str(k): v for k, v in domain_map.items()}
    results = {'exp': exp_name, 'args': saved_args, 'loss_log': [], 'evals': {}}
    best_loss = resumed_best_loss if resumed_best_loss is not None else float('inf')

    # ── Pretraining loop ──────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        total_domain_loss, domain_correct, domain_total = 0.0, 0, 0

        if args.domain_adv_lambda > 0:
            domain_clf.train()
            for csi, _, domain_label in pretrain_loader:
                csi = pad_csi(csi.to(device), padded_h, padded_w)
                domain_label = domain_label.to(device)

                out = model(csi); recon_loss = out[0]

                full_seq = extract_sequence_embeddings(model, csi, domain_adv_layer)
                full_emb = full_seq.mean(dim=1)
                domain_logits = domain_clf(grl(full_emb))
                domain_loss = F.cross_entropy(domain_logits, domain_label)

                total = recon_loss + args.domain_adv_lambda * domain_loss

                optimizer.zero_grad()
                domain_optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                domain_optimizer.step()

                total_loss += recon_loss.item()
                total_domain_loss += domain_loss.item()
                domain_correct += (domain_logits.argmax(1) == domain_label).sum().item()
                domain_total += domain_label.shape[0]
        else:
            for csi, _ in pretrain_loader:
                csi = pad_csi(csi.to(device), padded_h, padded_w)
                optimizer.zero_grad()
                out = model(csi); loss = out[0]
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(pretrain_loader)
        results['loss_log'].append({'epoch': epoch, 'loss': avg_loss})
        if args.domain_adv_lambda > 0:
            avg_domain_loss = total_domain_loss / len(pretrain_loader)
            domain_acc = domain_correct / domain_total
            results['loss_log'][-1]['domain_loss'] = avg_domain_loss
            results['loss_log'][-1]['domain_acc'] = domain_acc

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'loss': avg_loss,
                        'model_state': model.state_dict()},
                       f'{CKPT_DIR}/{exp_name}_best.pt')

        if epoch % 10 == 0:
            line = (f"Epoch {epoch:03d}/{args.epochs} | "
                   f"loss={avg_loss:.4f} | best={best_loss:.4f} | "
                   f"lr={scheduler.get_last_lr()[0]:.2e}")
            if args.domain_adv_lambda > 0:
                line += (f" | domain_loss={avg_domain_loss:.4f} | domain_acc={domain_acc:.3f} "
                        f"(chance={1/num_domains:.3f})")
            print(line)
            sys.stdout.flush()

        # ── Periodic eval ─────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n--- Eval at epoch {epoch} ---")
            epoch_results = {}

            for layer in eval_layers:
                print(f"  Layer {layer}:")
                train_feats, train_labels = get_features(
                    model, train_feat_loader, layer, device, padded_h, padded_w)

                if args.run_attentive_probe:
                    train_seq_feats, _ = get_sequence_features(
                        model, train_feat_loader, layer, device, padded_h, padded_w)

                layer_results = {}
                for sname, ldr in ood_loaders.items():
                    eval_feats, eval_labels = get_features(model, ldr, layer, device, padded_h, padded_w)

                    knn_acc, knn_f1 = knn_eval(
                        train_feats, train_labels, eval_feats, eval_labels, k=10)
                    lp_acc, lp_f1 = linear_probe_eval(
                        train_feats, train_labels, eval_feats, eval_labels,
                        num_classes, device, epochs=50)
                    mlp_acc, mlp_f1 = mlp_probe_eval(
                        train_feats, train_labels, eval_feats, eval_labels,
                        num_classes, device, epochs=50)

                    shift_metrics = compute_domain_shift_metrics(
                        train_feats, train_labels, eval_feats, eval_labels)

                    line = (f"    {sname:25s} {'(in-dist)' if sname in ('test_id', 'val_hp', 'test_final') else '(OOD)    '} "
                            f"KNN={knn_acc*100:.1f}% LP={lp_acc*100:.1f}% MLP={mlp_acc*100:.1f}% "
                            f"L2_dist={shift_metrics['centroid_l2_dist']:.2f} "
                            f"Cos_sim={shift_metrics['centroid_cos_sim']:.3f}")

                    layer_results[sname] = {
                        'knn_acc': knn_acc, 'knn_f1': knn_f1,
                        'lp_acc':  lp_acc,  'lp_f1':  lp_f1,
                        'mlp_acc': mlp_acc, 'mlp_f1': mlp_f1,
                        'centroid_l2_dist': shift_metrics['centroid_l2_dist'],
                        'centroid_cos_sim': shift_metrics['centroid_cos_sim'],
                    }

                    if args.run_attentive_probe:
                        eval_seq_feats, eval_seq_labels = get_sequence_features(
                            model, ldr, layer, device, padded_h, padded_w)
                        attn_light_acc, attn_light_f1 = attentive_probe_eval(
                            train_seq_feats, train_labels, eval_seq_feats, eval_seq_labels,
                            num_classes, device, epochs=args.attentive_probe_epochs, heavyweight=False)
                        attn_heavy_acc, attn_heavy_f1 = attentive_probe_eval(
                            train_seq_feats, train_labels, eval_seq_feats, eval_seq_labels,
                            num_classes, device, epochs=args.attentive_probe_epochs, heavyweight=True)
                        line += (f" AttnLight={attn_light_acc*100:.1f}% AttnHeavy={attn_heavy_acc*100:.1f}%")
                        layer_results[sname].update({
                            'attn_light_acc': attn_light_acc, 'attn_light_f1': attn_light_f1,
                            'attn_heavy_acc': attn_heavy_acc, 'attn_heavy_f1': attn_heavy_f1,
                        })

                    print(line)

                # ── Open-set eval for splits containing identities unseen in the train split ──
                for sname, ldr in openset_loaders.items():
                    eval_feats, eval_identities = get_features_with_identity(
                        model, ldr, layer, device, padded_h, padded_w)
                    n_identities_total = len(np.unique(eval_identities))

                    r1_acc, n_query, n_ident_used = rank1_retrieval_accuracy(
                        eval_feats, eval_identities, args.seed)
                    auc, eer, n_pairs_used = verification_auc_eer(
                        eval_feats, eval_identities, args.n_verification_pairs, args.seed)

                    r1_str = f"{r1_acc*100:.1f}%" if r1_acc is not None else "n/a"
                    auc_str = f"{auc:.4f}" if auc is not None else "n/a"
                    eer_str = f"{eer:.4f}" if eer is not None else "n/a"
                    trivial_note = " (TRIVIAL -- single identity in this split)" if n_identities_total == 1 else ""
                    print(f"    {sname:25s} (OPEN-SET) identities={n_identities_total:<3} "
                          f"Rank1={r1_str:<8} (n_query={n_query})  VerifAUC={auc_str}  "
                          f"EER={eer_str}{trivial_note}")

                    layer_results[sname] = {
                        'openset': True,
                        'n_identities': n_identities_total,
                        'rank1_acc': r1_acc, 'rank1_n_query': n_query,
                        'verif_auc': auc, 'verif_eer': eer, 'verif_n_pairs': n_pairs_used,
                    }

                epoch_results[f'layer_{layer}'] = layer_results

            results['evals'][f'epoch_{epoch}'] = epoch_results
            with open(f'{RESULTS_DIR}/{exp_name}.json', 'w') as f:
                json.dump(results, f, indent=2)
            print()

    print(f"\nDone. Best pretrain loss: {best_loss:.4f}")
    print(f"Results: {RESULTS_DIR}/{exp_name}.json")
    print(f"Checkpoint: {CKPT_DIR}/{exp_name}_best.pt")

if __name__ == '__main__':
    main()