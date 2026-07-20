"""
train_mae_har.py
MAE pretraining on HAR train_id, followed by:
  - KNN eval on test_id + OOD splits
  - Linear probe eval on test_id + OOD splits

Usage:
    python train_mae_har.py --epochs 300 --mask_ratio 0.75 --encoder_depth 6

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
"""
import os, sys, json, argparse, random, math, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset
from models.mae import MAE
from models.mae_v2 import MAEv2

DATA_ROOT  = config.DATA_ROOT
META_PATH  = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/splits'
RESULTS_DIR = '/home/zhuzih19/csi-project/csi-fall-detection/results/mae_har'
CKPT_DIR    = '/home/zhuzih19/csi-project/csi-fall-detection/checkpoints/mae_har'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

OOD_SPLITS = ['test_id', 'test_cross_device', 'test_cross_env', 'test_cross_user']

RAW_IMG_H, RAW_IMG_W = 232, 500  # standard CSI-Bench input shape (subcarriers x timesteps)
ENCODER_HEADS = 4                # hardcoded to match existing model construction below

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
def load_split(name, meta, label_map):
    import json as _json
    with open(f'{SPLITS_DIR}/{name}.json') as f:
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

def knn_eval(train_feats, train_labels, eval_feats, eval_labels, k=10):
    # Normalize
    mu  = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0,  keepdim=True) + 1e-8
    tf = (train_feats - mu) / std
    ef = (eval_feats  - mu) / std
    # Cosine similarity
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

    best_acc, best_f1 = 0.0, 0.0
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

    This is the single, shared implementation for fine-tuning -- it replaces two
    previously-separate, non-comparable implementations (finetune_eval()\'s old inline
    head, and a standalone MAEv2ForDownstream class) so there is one source of truth
    for fine-tune numbers instead of two head designs that couldn\'t be compared.

    Supports:
    1. Extracting embeddings at any specific encoder `layer` (not just the final one),
       matching the `layer` argument used by every other eval protocol (KNN / Linear
       Probe / MLP Probe), so fine-tune results are directly comparable layer-for-layer.
    2. Freezing the backbone entirely, unfreezing only the last k encoder blocks, or
       unfreezing the whole backbone -- see `unfreeze_last_n_layers`.
    3. Safety: the constructor deepcopies the passed-in model internally, so training
       this wrapper (even fully unfrozen) NEVER mutates the caller\'s original model --
       this was a real bug in the standalone-class version this replaces (it stored
       direct references to the pretrained model\'s submodules, so training it would
       have silently rewritten the caller\'s checkpoint in place).
    """
    def __init__(self, pretrained_model, num_classes, layer=None,
                 hidden_dim=256, unfreeze_last_n_layers=0):
        super().__init__()
        import copy
        backbone = copy.deepcopy(pretrained_model)  # never mutate the caller's model

        self.patch_embedding   = backbone.patch_embedding
        self.encoder_pos_embed = backbone.encoder_pos_embed
        self.encoder_blocks    = backbone.encoder_blocks
        self.encoder_norm      = backbone.encoder_norm
        self.layer = layer or len(self.encoder_blocks.layers)  # default: deepest layer

        self.mlp_head = nn.Sequential(
            nn.Linear(backbone.encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.set_backbone_trainable(unfreeze_last_n_layers)

        # L2-SP (Li et al. 2018): snapshot pretrained backbone weights ONCE, at construction
        # time, before any fine-tuning happens. l2sp_penalty() later measures how far the
        # (currently trainable) backbone params have drifted from this snapshot -- this is
        # taken BEFORE set_backbone_trainable() has any chance to be called again, so it is
        # always the true pretrained starting point, never a partially-fine-tuned state.
        # Only backbone params are snapshotted -- the head is randomly initialized and has
        # no meaningful "starting point" to regularize toward.
        self._pretrained_backbone_state = {
            name: p.detach().clone()
            for name, p in self._backbone_named_parameters()
        }

    def _backbone_named_parameters(self):
        """Yields (name, param) for every backbone parameter (patch_embedding,
        encoder_pos_embed, encoder_blocks), regardless of requires_grad -- used both to
        build the L2-SP snapshot and to compute the penalty against it."""
        for name, p in self.patch_embedding.named_parameters():
            yield f"patch_embedding.{name}", p
        yield "encoder_pos_embed", self.encoder_pos_embed
        for name, p in self.encoder_blocks.named_parameters():
            yield f"encoder_blocks.{name}", p

    def l2sp_penalty(self):
        """Sum of squared L2 distance between each currently-TRAINABLE backbone parameter
        and its pretrained (snapshotted) value. Frozen parameters are skipped (they can't
        have moved, so their contribution would always be exactly zero anyway, but skipping
        them avoids walking the whole backbone every step when most of it is frozen).
        Returns a 0-dim tensor on the same device as the model; safe to add directly into
        a training loss. Multiply by an l2sp_lambda coefficient before adding -- this
        function does not apply any weighting itself."""
        device = self.encoder_pos_embed.device
        penalty = torch.zeros((), device=device)
        for name, p in self._backbone_named_parameters():
            if p.requires_grad:
                penalty = penalty + (p - self._pretrained_backbone_state[name]).pow(2).sum()
        return penalty

    def set_backbone_trainable(self, unfreeze_last_n_layers):
        """
        unfreeze_last_n_layers:
            0    -> freeze the entire backbone (linear/MLP probing, via end-to-end training
                    rather than the two-stage extract-then-probe pipeline mlp_probe_eval uses
                    -- included mainly as a cross-check that the two give similar numbers)
            k>0  -> unfreeze only the last k encoder blocks, freeze the rest (a middle ground
                    that's less prone to catastrophic forgetting than a full unfreeze)
            None -> unfreeze the entire backbone (full fine-tune)
        """
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
        # unfreeze_last_n_layers == 0 -> backbone stays fully frozen (nothing more to do)

        status = ("frozen" if unfreeze_last_n_layers == 0
                  else "fully unfrozen" if unfreeze_last_n_layers is None
                  else f"last {unfreeze_last_n_layers} block(s) unfrozen")
        print(f"[MAEDownstreamHead] backbone: {status}, probing layer {self.layer}")

    def backbone_parameters(self):
        """Trainable backbone params only (excludes mlp_head) -- used to build the
        differential-LR optimizer param groups in finetune_eval()."""
        params = list(self.patch_embedding.parameters()) + [self.encoder_pos_embed] + \
                 list(self.encoder_blocks.parameters())
        return [p for p in params if p.requires_grad]

    def forward(self, x):
        """x: [B, 1, H, W] (already padded to a patch-size-compatible shape).
        Processes the FULL sequence, no masking (masking is pretraining-only),
        mean-pools over patch tokens at self.layer -- matches extract_layer_embeddings()."""
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
                  unfreeze_last_n_layers=None, hidden_dim=256,
                  eval_every=5, early_stop_patience=5, early_stop_split='ood_avg',
                  monitor_metric='loss', use_plateau_scheduler=False,
                  l2sp_lambda=0.0, verbose=True):
    """
    Full (or partial) fine-tuning of the pretrained encoder + a downstream MLP head,
    evaluated end-to-end -- NOT a frozen-feature probe (see mlp_probe_eval for that).
    Thin wrapper around MAEDownstreamHead (see its docstring for design details);
    this function owns the training loop and differential-LR optimizer setup.

    BREAKING CHANGE from the previous version: now returns (results, history) instead
    of just results -- update any caller that does `x = finetune_eval(...)` to
    `x, history = finetune_eval(...)`.

    OOD-aware early stopping + best-checkpoint selection:
    Fixed-epoch full-backbone fine-tuning showed monotonic OOD degradation even with a
    low backbone_lr (catastrophic forgetting continues to accumulate epoch over epoch,
    it doesn't just plateau) -- see the "2d" enc12 result in RESULTS.md. Lowering
    backbone_lr further only slows this down, it doesn't necessarily stop it from
    happening by the time training finishes. So instead of reporting whatever the model
    looks like at the LAST epoch, this evaluates every `eval_every` epochs, tracks the
    epoch with the best OOD performance, and reports/returns THAT checkpoint's results
    (reloaded via a deepcopied state_dict) -- not the final epoch's.

    monitor_metric: 'loss' (default) or 'acc'. 'loss' monitors CROSS-ENTROPY LOSS on the
        monitored split(s) -- computed at eval time (model.eval(), no_grad), NOT the
        training loss (which is only ever computed on train_id and never touches OOD
        data). Lower is better for 'loss', so the improvement direction, best_metric
        initialization, and ReduceLROnPlateau's `mode` all flip relative to 'acc' --
        this is handled internally, you don't need to adjust anything else when switching.
    early_stop_split: which split is monitored (via monitor_metric) for both early
        stopping and the optional plateau scheduler. 'ood_avg' (default) averages across
        every split except 'test_id' -- monitoring test_id (or anything that includes it)
        would miss forgetting entirely, since test_id keeps improving even as OOD
        degrades. Pass an explicit split name (e.g. 'test_cross_device') to monitor a
        single split instead. Requires at least one non-test_id split in eval_loaders
        unless you pass an explicit split name.
    early_stop_patience: stop if the monitored metric hasn't improved for this many
        *evaluations* (i.e. patience * eval_every epochs of no improvement), not epochs.
    use_plateau_scheduler: if True, use ReduceLROnPlateau (factor=0.5, patience=2 evals)
        driven by the same monitored metric, instead of CosineAnnealingLR. Off by
        default -- CosineAnnealingLR remains the default schedule for backward compatibility.
    l2sp_lambda: L2-SP regularization strength (Li et al. 2018). 0.0 (default) = off,
        matching prior behavior. When > 0, adds `l2sp_lambda * ||backbone_params -
        pretrained_backbone_params||^2` to the training loss -- this penalizes the
        backbone for drifting from its pretrained starting point directly, rather than
        relying on a low backbone_lr to indirectly limit drift. Motivation: the "2d"
        enc12 catastrophic-forgetting result showed OOD loss degrading monotonically
        from the very first evaluated epoch even with backbone_lr=1e-5 and OOD-aware
        early stopping -- suggesting the backbone's gradient direction itself (not just
        how far it moves) is the problem, which L2-SP addresses more directly than LR
        alone. When l2sp_lambda > 0, standard weight_decay on the backbone param group is
        automatically set to 0 (L2-SP replaces it for backbone params, per Li et al.'s
        formulation -- combining both would pull the same parameters toward two different
        targets, zero and the pretrained value, at the same time). weight_decay on the
        head remains unchanged (the head has no pretrained starting point to regularize
        toward -- ordinary L2-to-zero is the standard choice there).

    Returns (results, history):
        results -- {split_name: {'acc': ..., 'f1': ..., 'loss': ...}}, evaluated at the
                   BEST epoch found (by early_stop_split + monitor_metric), not
                   necessarily the last epoch trained. 'loss' is a new key added to each
                   split's dict alongside the existing 'acc'/'f1' -- still plugs into
                   plot_final_accuracy_by_group / plot_summary_table_image /
                   print_summary_table unchanged (they only ever read 'acc'/'f1'/etc by
                   name, so the extra 'loss' key is simply ignored by those functions).
        history -- list of dicts, one per evaluation:
                   [{'epoch': ..., 'monitored_metric': ..., split_name: {'acc':...,'f1':...,'loss':...}, ...}, ...]
                   Useful for plotting OOD loss (or accuracy) vs. epoch to see exactly
                   where forgetting starts.
    """
    import copy

    wrapper = MAEDownstreamHead(model, num_classes, layer=layer, hidden_dim=hidden_dim,
                                unfreeze_last_n_layers=unfreeze_last_n_layers).to(device)

    param_groups = [{'params': wrapper.mlp_head.parameters(), 'lr': head_lr, 'weight_decay': 0.05}]
    backbone_params = wrapper.backbone_parameters()
    if backbone_params:
        # If L2-SP is active, zero out ordinary weight_decay on the backbone group --
        # L2-SP (added into the loss below) replaces it for backbone params. See the
        # l2sp_lambda docstring above for why combining both would be ill-posed.
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
                    losses_all += batch_loss.item() * y.shape[0]  # sum, weighted by batch size
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

    Purpose: directly test whether the LP accuracy ceiling is a REPRESENTATIONAL limit
    (a single hyperplane per class structurally cannot separate a non-convex, multi-modal
    class distribution) rather than an optimization/undertraining issue -- if MLP >> LP,
    that confirms the ceiling is about linear separability specifically. Note this also
    sidesteps the t-SNE-distortion caveat: t-SNE's local-neighborhood objective doesn't
    preserve linear relationships, so "looks non-convex in a 2D t-SNE plot" alone doesn't
    prove "not linearly separable in the original embedding space" -- this eval operates
    on the real, full-dimensional embedding, not a 2D projection, so it's decisive either way.
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
    parser.add_argument('--eval_layers',   type=str,   default='1,3,6')
    parser.add_argument('--eval_every',    type=int,   default=50)
    parser.add_argument('--mask_strategy', type=str, default='random', choices=['random','time','freq','mixed','2d'])
    parser.add_argument('--patch_h',       type=int,   default=29)
    parser.add_argument('--patch_w',       type=int,   default=25)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--skip_mem_check', action='store_true',
                         help='Bypass the pre-flight attention-memory safety check (not recommended)')
    parser.add_argument('--mem_budget_gb', type=float, default=20.0,
                         help='Safety budget (GB) for a single attention layer before hard-stopping')
    args = parser.parse_args()

    # Seed control for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    eval_layers = [int(x) for x in args.eval_layers.split(',')]
    exp_name = (f"mae_har_ep{args.epochs}_mask{args.mask_ratio}_strategy{args.mask_strategy}_ph{args.patch_h}pw{args.patch_w}_seed{args.seed}"
                f"_enc{args.encoder_depth}_dim{args.encoder_dim}_bs{args.batch_size}")
    print(f"\nExperiment: {exp_name}")

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

    # Data
    meta = pd.read_csv(META_PATH)
    import json as _json
    with open(f'{SPLITS_DIR}/train_id.json') as f:
        train_ids = set(_json.load(f))
    train_df  = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    num_classes = len(label_map)
    print(f"label_map: {label_map}  num_classes: {num_classes}")

    train_ds = MultiTaskDataset(train_df, DATA_ROOT, 'Multitask', label_map=label_map)
    pretrain_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
    # For feature extraction (no shuffle)
    train_feat_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4)

    # OOD loaders
    ood_loaders = {}
    for sname in OOD_SPLITS:
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Persist padding metadata alongside args so downstream visualization/analysis
    # can tell exactly what shape the model actually trained on.
    saved_args = vars(args).copy()
    saved_args['padded_h'] = padded_h
    saved_args['padded_w'] = padded_w
    saved_args['num_patches'] = num_patches
    results = {'exp': exp_name, 'args': saved_args, 'loss_log': [], 'evals': {}}
    best_loss = float('inf')

    # ── Pretraining loop ──────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'loss': avg_loss,
                        'model_state': model.state_dict()},
                       f'{CKPT_DIR}/{exp_name}_best.pt')

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"loss={avg_loss:.4f} | best={best_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            sys.stdout.flush()

        # ── Periodic eval ─────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n--- Eval at epoch {epoch} ---")
            epoch_results = {}

            # Extract train features once per eval layer
            for layer in eval_layers:
                print(f"  Layer {layer}:")
                train_feats, train_labels = get_features(
                    model, train_feat_loader, layer, device, padded_h, padded_w)

                layer_results = {}
                for sname, ldr in ood_loaders.items():
                    eval_feats, eval_labels = get_features(model, ldr, layer, device, padded_h, padded_w)

                    knn_acc, knn_f1 = knn_eval(
                        train_feats, train_labels, eval_feats, eval_labels, k=10)
                    lp_acc, lp_f1 = linear_probe_eval(
                        train_feats, train_labels, eval_feats, eval_labels,
                        num_classes, device, epochs=50)
                    # Non-linear probe, same protocol as LP (see mlp_probe_eval docstring) --
                    # if this tracks close to KNN rather than LP, that's direct evidence the
                    # LP ceiling is about linear separability specifically, not undertraining
                    # or a t-SNE visualization artifact. Only run at args.eval_every intervals
                    # like the others -- it costs about the same as one extra LP eval.
                    mlp_acc, mlp_f1 = mlp_probe_eval(
                        train_feats, train_labels, eval_feats, eval_labels,
                        num_classes, device, epochs=50)

                    tag = '(in-dist)' if sname == 'test_id' else '(OOD)    '
                    print(f"    {sname:25s} {tag} "
                          f"KNN={knn_acc*100:.1f}% LP={lp_acc*100:.1f}% MLP={mlp_acc*100:.1f}%")
                    layer_results[sname] = {
                        'knn_acc': knn_acc, 'knn_f1': knn_f1,
                        'lp_acc':  lp_acc,  'lp_f1':  lp_f1,
                        'mlp_acc': mlp_acc, 'mlp_f1': mlp_f1,
                    }
                epoch_results[f'layer_{layer}'] = layer_results

            results['evals'][f'epoch_{epoch}'] = epoch_results
            # Save intermediate results
            with open(f'{RESULTS_DIR}/{exp_name}.json', 'w') as f:
                json.dump(results, f, indent=2)
            print()

    print(f"\nDone. Best pretrain loss: {best_loss:.4f}")
    print(f"Results: {RESULTS_DIR}/{exp_name}.json")
    print(f"Checkpoint: {CKPT_DIR}/{exp_name}_best.pt")

if __name__ == '__main__':
    main()