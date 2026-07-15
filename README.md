# CSI Fall Detection / HAR SSL Foundation Model Project

This repository implements a Masked Autoencoder (MAE) self-supervised pretraining pipeline for
WiFi CSI sensing, with the primary active workstream targeting **CSI-Bench HumanActivityRecognition
(HAR)** and cross-device / cross-environment / cross-user out-of-distribution (OOD) generalization.
An earlier, separate experimental track on Fall Detection with I-JEPA and Bootleg pretraining is
preserved in [Legacy Experiments](#legacy-experiments-fall-detection--i-jepa--bootleg) below.

## Table of Contents

- [Current Status](#current-status)
- [Key Findings (Current HAR Ablation Series)](#key-findings-current-har-ablation-series)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Evaluation Protocols](#evaluation-protocols)
- [Known Issues / Open Data Quality Items](#known-issues--open-data-quality-items)
- [Legacy Experiments: Fall Detection / I-JEPA / Bootleg](#legacy-experiments-fall-detection--i-jepa--bootleg)
- [References](#references-and-related-work)

📊 **[See RESULTS.md](RESULTS.md)** for full tables, ablation results, and the cross-depth
replication analysis.

---

## Current Status

**Primary task**: CSI-Bench `HumanActivityRecognition` (5 classes: jumping, running,
seated-breathing, walking, wavinghand). Official splits: `test_id` (in-distribution),
`test_cross_device`, `test_cross_env`, `test_cross_user` (three independent OOD splits).

**Completed ablations** (all via `train_mae_har.py`, results in `results/mae_har/`,
visualized in `consolidated_visualization.ipynb`):

| Ablation | Encoder depth | Status |
|---|---|---|
| Mask ratio (0.5 / 0.75 / 0.875 / 0.95) | enc12 | ✅ Complete, n=2 |
| Mask strategy (random / freq / mixed / time / 2d) | enc6 | ✅ Complete, ⚠️ data quality issue (see below) |
| Mask strategy (random / freq / mixed / time / 2d) | enc12 | ✅ Complete, n=2 clean |
| Patch size (11 / 13 / 15 / 17 / 19) | enc6 | ✅ Complete, n=2 |
| Patch size (11 / 13 / 15 / 17 / 19 / 21) | enc12 | ✅ Complete, n=2 |

**In progress / recently added**:
- `mlp_probe_eval()` — non-linear (1-hidden-layer MLP) frozen-feature probe, added alongside
  KNN and Linear Probe to test whether the LP accuracy ceiling is a linear-separability limit.
- `finetune_eval()` / `MAEDownstreamHead` — full/partial backbone fine-tuning (frozen /
  unfreeze-last-k / full unfreeze), differential learning rate, for comparing frozen-probe
  representation quality against fine-tuned ceiling performance. Two earlier, non-comparable
  implementations were merged into one (`MAEDownstreamHead`, see [Architecture](#architecture))
  after a real bug was found in the older one (direct references to the pretrained model's
  submodules instead of a deep copy, meaning training it would have silently mutated the
  original checkpoint object in memory). **First real result: unfreezing the backbone shows
  clear catastrophic forgetting — OOD accuracy drops as more of the backbone is unfrozen, even
  as `test_id` accuracy rises. See [RESULTS.md](RESULTS.md#fine-tuning-frozen-vs-partial-vs-full-backbone-unfreeze).**
- `visualize_embeddings.py` — t-SNE + silhouette-score diagnostic for inspecting embedding
  geometry (class structure vs. domain/session structure).

---

## Key Findings (Current HAR Ablation Series)

### 1. LP and KNN diverge systematically and predictably — not randomly

Across **all 5 independent ablations** run so far, the direction of the KNN/LP gap is
**100% consistent**:

- **In-distribution (`test_id`)**: KNN always wins, by a wide margin (KNN ~0.92–0.96 vs.
  LP ~0.55–0.59 in every ablation).
- **Out-of-distribution** (any of the three OOD splits): LP always wins on average, though by a
  narrower margin.

This is **not** attributed to train/pretrain data overlap (unlike the legacy Fall Detection
finding below) — `train_id` and all OOD splits are official, disjoint CSI-Bench splits. The
current working explanation, backed by t-SNE + silhouette-score analysis of the embedding space,
is that the representation is **locally separable but not linearly separable** (KNN can exploit
non-convex, multi-modal class clusters; a single linear decision boundary per class cannot), and
that domain/session identity — not class — is the dominant factor structuring the embedding space
(see `visualize_embeddings.py` output, [RESULTS.md](RESULTS.md#embedding-geometry-investigation)).

Even after picking the correct protocol for the split of interest, **the two protocols agree on
which config is "best" only ~10% of the time** (2 of 20 column-wise comparisons across all
ablation summary tables). Any "best config" claim needs to specify both the protocol and the
split, or report both.

### 2. Encoder depth is a confirmed confound — two independent replication failures

Rerunning both mask-strategy and patch-size ablations at `encoder_depth=12` (vs. the original
`encoder_depth=6`) reversed the headline conclusion in both cases:

- **Mask strategy**: `"2d"` was the clear worst strategy on Cross-Device at enc6 (LP 0.2912) —
  it is the clear **best** at enc12 (LP 0.4217), a +13pp swing.
- **Patch size**: `patch=17` was the clear best on Cross-Device at enc6 (LP 0.4638) — it drops to
  near-worst at enc12 (LP 0.2953), a −16.9pp swing, the largest single-config swing observed.
- `patch=15` is the one patch size that stays stable across both depths (0.4191 / 0.4216) and is
  a more defensible "robust default" than either single-depth winner.

No "best config" conclusion from a single encoder depth should be treated as final without a
cross-depth check.

### 3. Data quality issues can silently distort conclusions — two caught, one still open

- A `check_attention_memory()` bug (gating on per-layer instead of cumulative attention memory
  across all retained layers) let `patch=7` pass a pre-flight safety check and then genuinely
  OOM in production — fixed to gate on the cumulative estimate.
- A `loss_log` length-mismatch bug crashed the visualization notebook when a group mixed a
  partial run with complete ones — fixed to auto-truncate with a `[warn]`, which caught a
  genuinely incomplete `patch=19` (enc12) seed automatically.
- **Still open**: the enc6 mask-strategy `"2d"` group shows exactly `std=0.000` on the ID split
  across both seeds (likely a duplicated run, unconfirmed), and the `"random"` group has shown
  `n=3` instead of `n=2` (a stray unseeded file not yet removed from `results/`). The enc6 "2d is
  worst" finding referenced above should be treated as provisional until this is resolved.

---

## Project Structure

```
.
├── config.py                       # DATA_ROOT and other central config
├── train_mae_har.py                # Main MAE training + eval script (current active pipeline)
│                                    #   NOTE: does NOT yet include mlp_probe_eval/MAEDownstreamHead
│                                    #   (local prototypes — see Architecture below)
├── finetune_mae_har.py             # Committed HAR fine-tuning script (two-phase, MAE-only)
├── models/
│   ├── vit.py                      # ViT backbone: PatchEmbedding, Encoder, MultiHeadAttention (naive, non-flash)
│   ├── mae.py                      # MAE (random masking)
│   ├── mae_v2.py                   # MAEv2 (block masking strategies: time/freq/mixed/2d)
│   ├── csibench_models.py          # Baseline architectures for eval_har_ood_v2.py (mlp/lstm/resnet18/
│   │                                #   transformer/vit/patchtst/timesformer1d)
│   ├── resnet.py, baselines.py     # Additional baseline model definitions
│   └── ijepa.py, bootleg_with_recon.py, decoder.py   # Legacy pretraining methods (Fall Detection track)
├── data/
│   └── dataset.py                  # MultiTaskDataset, CSI loading/normalization
├── eval_har_ood_v2.py              # Official CSI-Bench baseline replication (independent of train_mae_har.py's
│                                    #   data pipeline — uses csibench-official's BenchmarkCSIDataset loader)
├── eval_finetune.py                # Legacy Fall Detection / Motion Source fine-tune script (NOT HAR-related)
├── visualize_embeddings.py         # t-SNE + silhouette-score embedding geometry diagnostic
├── consolidated_visualization.ipynb # All ablation results: Sections A-E (ratio/strategy×2 depths/patch×2 depths)
├── launch_mask_strategy_enc12.sh   # Background launch script (nohup + disown)
├── launch_patch_size_ablation_v2.sh    # enc6 patch-size sweep (11/13/15/17/19)
├── launch_patch_size_ablation_enc12.sh # enc12 patch-size sweep (11@bs96, 13/15/17/19/21@bs128)
├── results/mae_har/                # Per-run result JSON (loss_log, evals by layer/split/checkpoint)
├── results/csibench_official/      # Official baseline checkpoints + results (used by eval_har_ood_v2.py)
├── figs/                           # Ablation charts (from consolidated_visualization.ipynb) +
│                                    #   figs/embeddings/ (t-SNE diagnostics from visualize_embeddings.py)
├── checkpoints/mae_har/            # best_model.pt per run (NOT committed to git — see note below)
└── logs/                           # Training logs (per-run + launcher status logs)
```

**Note on checkpoints and git**: model checkpoints are large (100+ MB each) and are **not**
tracked in git — an earlier commit of baseline checkpoints bloated `.git` to 2.95GB and broke
`git push`; history was rewritten with `git filter-repo` and `.gitignore` now excludes
`*.pt`/`*.pth`. Manage checkpoints locally or via a separate artifact store.

---

## Configuration

Key parameters (all configurable via `train_mae_har.py` CLI args):

- **Input shape**: 232 subcarriers × 500 timesteps (standard CSI-Bench HAR input)
- **Patch size**: configurable `--patch_h`/`--patch_w`; square patches that don't evenly divide
  232×500 (e.g. 11, 13, 15, 17, 19, 21) are zero-padded to the nearest compatible size
  automatically (`compute_padded_size` / `pad_csi`)
- **Encoder depth**: 6 or 12 (both actively studied)
- **Encoder dim**: 128, 4 attention heads
- **Mask ratio**: 0.5 / 0.75 / 0.875 / 0.95 (ablated)
- **Mask strategy**: `random` (MAE) / `time`, `freq`, `mixed`, `2d` (MAEv2, block masking)
- **Batch size**: 128 default; smaller patch sizes (higher `num_patches`) may require a reduced
  batch size — see `check_attention_memory()` pre-flight check
- **Data root**: `/home/zhuzih19/data/csi-bench-dataset`

### A note on the attention-memory constraint

`models/vit.py`'s `MultiHeadAttention` is a **naive (non-flash)** implementation — its attention
score tensor is `O(batch_size × heads × num_patches²)` in memory, and because there is no
gradient checkpointing, **all `encoder_depth` layers' attention scores are retained
simultaneously during backward**. `check_attention_memory()` gates on this cumulative estimate
(not just a single layer) and will refuse to start with a suggested safe `--batch_size` if the
projected memory exceeds budget. Small patch sizes (3×3, 5×5, 7×7) are effectively infeasible at
`batch_size=128` under this architecture; see [RESULTS.md](RESULTS.md) for the exact numbers
that motivated the current 11–21 patch-size range.


## Architecture

All HAR pretraining uses the shared building blocks in `models/vit.py`, wrapped by either
`MAE` (`models/mae.py`) or `MAEv2` (`models/mae_v2.py`). Downstream evaluation (fine-tuning
specifically) uses a separate `MAEDownstreamHead` wrapper described below.

### Shared building blocks (`models/vit.py`)

```
PatchEmbedding     — Conv2d(kernel=stride=(patch_h, patch_w)), flattens to [B, N, d_model]
                      (no CLS token in MAE/MAEv2 — see below)
MultiHeadAttention  — standard scaled dot-product attention, NAIVE implementation
                      (Q @ K^T, softmax, @ V) with no flash/memory-efficient kernel.
                      Attention-score memory is O(batch_size × heads × num_patches²) per
                      layer, and — since there's no gradient checkpointing — ALL
                      encoder_depth layers' scores are retained simultaneously during
                      backward. This is why check_attention_memory() gates on the
                      CUMULATIVE estimate, not a per-layer one (see Configuration above).
EncoderBlock        — MultiHeadAttention → AddNorm → FeedForward (GELU) → AddNorm (post-norm)
Encoder             — nn.ModuleList of `N` EncoderBlocks, applied sequentially. Individual
                      blocks are reachable at `model.encoder_blocks.layers` — used by
                      MAEDownstreamHead's `unfreeze_last_n_layers` to partially unfreeze
                      only the last k blocks.
```

`patch_h`/`patch_w` don't need to be equal (non-square patches are used for the default
`29×25` config) and don't need to evenly divide 232×500 — `train_mae_har.py` zero-pads the
input to the nearest compatible multiple before `PatchEmbedding` (see Configuration above).

### MAE (`models/mae.py`)

```
INPUT [B, 1, 232, 500] (zero-padded to nearest patch-size multiple if needed)
  │
  ▼
PatchEmbedding + encoder_pos_embed              →  tokens [B, N, encoder_dim]
  │
  ▼
Random masking (mask_ratio, default 0.75)       →  N_visible = N × (1 - mask_ratio) tokens kept,
  │                                                  N_masked = N × mask_ratio tokens DROPPED
  │                                                  (not zeroed/attention-masked — removed
  │                                                   from the sequence entirely)
  ▼
Encoder (encoder_depth blocks) + LayerNorm      →  encoded [B, N_visible, encoder_dim]
  │
  ▼
Linear(encoder_dim → decoder_dim), then insert a single learned mask_token at every
masked position, unshuffle back to original patch order, + decoder_pos_embed
  │
  ▼
Decoder (decoder_depth blocks, its own smaller Encoder instance) + LayerNorm
  │
  ▼
Linear(decoder_dim → patch_h × patch_w)         →  pred [B, N, patch_h × patch_w]
  │
  ▼
MSE(pred, target), computed ONLY on the N_masked positions
```

The encoder never sees masked patches at all (excluded from its input sequence, not just
attention-masked) — standard MAE design (He et al. 2022), cheaper to pretrain than a
full-sequence masked model. The decoder is discarded after pretraining; only
`patch_embedding` + `encoder_pos_embed` + `encoder_blocks` + `encoder_norm` are used at
evaluation time.

**No CLS token.** `extract_layer_embeddings()` (used by every evaluation protocol — KNN, LP,
MLP-probe, and as the reference implementation for fine-tuning) mean-pools over all patch
tokens at the requested layer instead:

```python
layer_outputs[i + 1] = self.encoder_norm(h).mean(dim=1)   # [B, encoder_dim]
```

This always runs the encoder over the complete, unmasked input — masking is a
pretraining-time-only operation.

### MAEv2 (`models/mae_v2.py`)

Same encoder/decoder skeleton as `MAE`, plus:

1. **Configurable masking strategy** (`mask_strategy`):
   - `"random"` — identical to `MAE`'s masking (kept for parity; in practice
     `train_mae_har.py` routes `mask_strategy="random"` to the `MAE` class directly, not
     `MAEv2` — see below).
   - `"time"` — masks entire columns of the `[n_h, n_w]` patch grid (all subcarriers, for a
     contiguous span of time)
   - `"freq"` — masks entire rows (all timesteps, for a contiguous span of subcarriers)
   - `"mixed"` — each sample independently coin-flips between `"time"` and `"freq"` (one or
     the other per-sample, not a blend)
   - `"2d"` — masks a single contiguous rectangular block (both dimensions narrowed to
     roughly `sqrt(mask_ratio)` each), surrounded on all sides by visible context

2. **Optional physics-aware loss** (`use_physics_loss=True`, off by default): two extra MSE
   terms — `loss_spec` (first-order difference along the subcarrier axis) and `loss_temp`
   (first-order difference along the time axis). **Not currently used anywhere in
   `train_mae_har.py`** — worth knowing about if reconstruction quality ever needs revisiting.

Every `mask_strategy=random` row in every ablation table in RESULTS.md — including *all* of
the mask-ratio and patch-size ablations, which fix `mask_strategy=random` — is trained with
`MAE`. Only the `freq`/`mixed`/`time`/`2d` rows of the mask-strategy tables use `MAEv2`.

### Fine-tuning: `finetune_mae_har.py`

> **⚠️ Status note**: this is the fine-tuning implementation actually committed to the repo.
> An experimental alternative (`MAEDownstreamHead`, supporting partial-layer unfreezing,
> `MAEv2`/block-masking checkpoints, and an arbitrary probing `layer`) was prototyped locally
> during development and used to produce the catastrophic-forgetting result in
> [RESULTS.md](RESULTS.md#fine-tuning-frozen-vs-partial-vs-full-backbone-unfreeze), but **has
> not been merged into `train_mae_har.py` in this repo** — those specific numbers cannot yet
> be reproduced by cloning this repo as-is. Treat that result as a preliminary finding pending
> the merge, not as reproducible from the code below.

`finetune_mae_har.py` is a standalone script (does not import from `train_mae_har.py`) with a
simpler, two-phase design:

```python
# Phase 1 (--freeze_epochs, default 10): encoder fully frozen, train only a linear head
for p in mae.parameters(): p.requires_grad_(False)
optim = torch.optim.Adam(head.parameters(), lr=args.lr)

# Phase 2 (--epochs - --freeze_epochs): encoder fully unfrozen, end-to-end fine-tune
optim = torch.optim.AdamW([
    {'params': mae.parameters(), 'lr': args.lr * 0.1},   # encoder: 10x smaller LR
    {'params': head.parameters(), 'lr': args.lr},
], weight_decay=0.05)
```

Key differences from the (currently unmerged) `MAEDownstreamHead` design:

| | `finetune_mae_har.py` (in repo) | `MAEDownstreamHead` (local prototype) |
|---|---|---|
| Unfreeze granularity | Two-phase only: frozen → fully unfrozen | Three modes: frozen / last-`k`-layers / fully unfrozen |
| Head | `nn.Linear(encoder_dim, num_classes)` | `Linear→LayerNorm→ReLU→Dropout(0.3)→Linear` |
| Encoder-vs-head LR ratio | 10× (`lr * 0.1` vs `lr`) | 100× (`1e-5` vs `1e-3`, more conservative) |
| Model support | `MAE` only, `patch_h=29`/`patch_w=25`/`mask_ratio=0.75` hardcoded | `MAE` + `MAEv2`, reads patch/ratio/strategy from the paired result JSON |
| Probing layer | Fixed at the final layer | Configurable `layer`, matches KNN/LP/MLP-probe for direct comparison |
| Model-mutation safety | No explicit deep copy (loads a fresh model each run via CLI, so this hasn't been an issue in practice — but be careful if reusing a `model` object already in memory elsewhere in the same session) | Explicit `copy.deepcopy()` in the constructor |

Usage:

```bash
python finetune_mae_har.py \
    --ckpt checkpoints/mae_har/<exp_name>_best.pt \
    --encoder_depth 6 --encoder_dim 128 \
    --epochs 50 --freeze_epochs 10 --lr 1e-4 --batch_size 128
```

Results are saved to `results/mae_har/finetune_<ckpt_basename>_ep<epochs>.json`, in the same
per-checkpoint eval structure (`evals.epoch_N.{split}.{acc,f1}`) as `train_mae_har.py`'s own
output, evaluated every 10 epochs of Phase 2 on all four splits.

**Not yet run on the current HAR ablation checkpoints** (mask-ratio/strategy/patch-size sweeps)
as of this writing — the only fine-tuning numbers in RESULTS.md so far come from the local
`MAEDownstreamHead` prototype noted above.

### Evaluation-time heads at a glance

| Protocol | Head | Backbone | Implementation |
|---|---|---|---|
| KNN | none (non-parametric) | frozen | `knn_eval()` in `train_mae_har.py` |
| Linear Probe | `nn.Linear(encoder_dim, num_classes)` | frozen | `linear_probe_eval()` in `train_mae_har.py` |
| MLP Probe | `Linear→ReLU→Linear` (hidden=128) | frozen | `mlp_probe_eval()` — **local prototype, not yet merged**, see status note above |
| Fine-tune (committed) | `nn.Linear(encoder_dim, num_classes)` | frozen (Phase 1) → fully unfrozen (Phase 2) | `finetune_mae_har.py` (standalone script) |
| Fine-tune (prototype) | `Linear→LayerNorm→ReLU→Dropout(0.3)→Linear` | frozen / last-k / fully unfrozen | `MAEDownstreamHead` — **local prototype, not yet merged** |

---

## Quick Start

### Run a single MAE pretraining + eval

```bash
python train_mae_har.py \
    --epochs 300 --mask_ratio 0.75 --mask_strategy random \
    --encoder_depth 6 --batch_size 128 \
    --patch_h 29 --patch_w 25 --seed 42
```

### Run a full ablation sweep in the background (survives SSH disconnect)

```bash
cd ~/csi-project/csi-fall-detection
nohup bash launch_mask_strategy_enc12.sh > logs/launcher.log 2>&1 &
disown
```

Monitor with `tail -f logs/mae_har_enc12_strategy<name>_seed<seed>_ep300.log`.

### Visualize all ablation results

Open `consolidated_visualization.ipynb` and run all cells — Sections A (mask ratio, enc12),
B/D (mask strategy, enc6/enc12), C/E (patch size, enc6/enc12) each load, group, and plot
automatically from whatever result JSONs are present in `results/mae_har/`.

### Diagnose the LP/KNN gap on a specific checkpoint

```bash
python visualize_embeddings.py \
    --checkpoint checkpoints/mae_har/<exp_name>_best.pt \
    --result_json results/mae_har/<exp_name>.json \
    --layer 6 --ood_split test_cross_device \
    --color_by device   # or session_id, environment, user, distance
```

### Fine-tune a checkpoint (committed, two-phase: frozen head → full unfreeze)

```bash
python finetune_mae_har.py \
    --ckpt checkpoints/mae_har/<exp_name>_best.pt \
    --encoder_depth 6 --encoder_dim 128 \
    --epochs 50 --freeze_epochs 10 --lr 1e-4 --batch_size 128
```

Currently `MAE`-only (`patch=29×25`, `mask_ratio=0.75` hardcoded) — see
[Architecture](#fine-tuning-finetune_mae_harpy) for the two-phase design.

### ⚠️ Experimental (not yet merged): partial-unfreeze fine-tune comparison

`run_finetune_experiment.py` + `MAEDownstreamHead` (a local prototype, not committed to this
repo) produced the catastrophic-forgetting result in RESULTS.md by comparing frozen /
last-2-layers / fully-unfrozen fine-tuning against KNN/LP on the same checkpoint. Not
reproducible from this repo as-is until `MAEDownstreamHead` is merged into `train_mae_har.py` —
see the status note in [Architecture](#fine-tuning-finetune_mae_harpy).

---

## Evaluation Protocols

| Protocol | What it tests | When to trust it |
|---|---|---|
| **KNN** | Local neighborhood structure; non-parametric, no training | Best for **in-distribution** — degrades under domain shift since it assumes train/test neighbors share label |
| **Linear Probe (LP)** | Linear separability of frozen features | More robust than KNN **out-of-distribution**, but underestimates representation quality if classes are non-convex |
| **MLP Probe** ⚠️ *not yet merged* (`mlp_probe_eval`) | Non-linear separability of frozen features, same protocol as LP | Diagnostic: if MLP ≫ LP, the LP ceiling is a linear-separability limit, not undertraining. Local prototype only — see [Architecture](#fine-tuning-finetune_mae_harpy) |
| **Fine-tune (committed)** (`finetune_mae_har.py`) | End-to-end adaptation ceiling, two-phase (frozen head-only, then fully unfrozen) | Standalone script, `MAE` only, hardcoded `patch=29×25`/`mask_ratio=0.75`. **Not yet run on the current HAR ablation checkpoints** |
| **Fine-tune (prototype)** ⚠️ *not yet merged* (`MAEDownstreamHead`) | End-to-end adaptation ceiling with partial-unfreeze granularity | Local prototype used to produce the **catastrophic forgetting** result in [RESULTS.md](RESULTS.md#fine-tuning-frozen-vs-partial-vs-full-backbone-unfreeze) — unfreezing improves `test_id` but consistently hurts all three OOD splits. Not reproducible from this repo's committed code as-is |

**Always report the split alongside the protocol** — see Key Finding #1 above.

---

## Known Issues / Open Data Quality Items

1. **enc6 mask-strategy `"2d"` — suspected duplicate seed run.** ID-split `std=0.000` exactly
   across both seeds. Unconfirmed; do not treat the enc6 "2d is worst" finding as final until
   resolved.
2. **enc6 mask-strategy `"random"` — stray unseeded file.** Shows `n=3` instead of `n=2` in
   `results/mae_har/`; needs identification and removal.
3. **enc12 patch-size `patch=19` — one incomplete seed.** Auto-detected by the loss-log
   truncation fix (`[warn] group=19: loss_log lengths differ [250, 300]`); rerun to completion
   before treating patch=19's enc12 numbers as final.
4. **Naive attention limits feasible patch sizes.** See [Configuration](#configuration) above —
   patch sizes below ~11 require reduced batch size or are effectively infeasible at
   `encoder_depth≥6`.

---

## Legacy Experiments: Fall Detection / I-JEPA / Bootleg

**⚠️ This section describes an earlier, separate experimental track on a different task
(Fall Detection, 429 labeled samples) with a different pretraining dataset composition. Its
data-leakage finding does NOT apply to the current HAR ablation series above** — the current
HAR splits (`train_id`/`test_id`/OOD) are official, disjoint CSI-Bench splits with no known
train/pretrain overlap, and the LP/KNN gap there has a different, geometry-based explanation
(see Key Finding #1). This section is preserved for historical continuity only.

The original project scope explored fall detection classification via (fine-tuning results
for this track, where reported, use `eval_finetune.py` — a separate, task-specific script
with its own hardcoded checkpoint paths (`mae_ep200/300/500...`) and task configs for
`FallDetection`/`MotionSourceRecognition`; unrelated to `finetune_mae_har.py`, which is HAR-only):
- **Supervised ViT-4L baseline**: 82.2% test accuracy on 429 labeled samples, no pretraining.
- **I-JEPA pretraining**: 65.6% LP accuracy — underperformed supervised baseline, attributed to
  insufficient pretraining data (429–20K samples vs. 1.2M+ in the original I-JEPA paper) and
  short training (10 epochs vs. 600+).
- **Bootleg (contrastive + reconstruction) pretraining**: training instability (loss rebound
  after epoch 5) attributed to CPU-only training, small data (20K samples), high EMA momentum,
  and a too-small predictor network — not resolved; flagged as needing GPU + extended training.
- **MAE pretraining (ViT-12L, 341K multi-task samples)**: MAE-200/300/500/1000 variants. The
  **critical finding** in this track was that the Fall Detection training set (429 samples) was
  a subset of the 341K MAE pretraining set, causing **data leakage**: Linear Probe could exploit
  the encoder's implicit "warm" knowledge of training samples, inflating LP accuracy by up to
  24pp on tasks like Motion Source Recognition relative to a **user-independent** evaluation
  (test users held out from pretraining), which showed only a 1.4–2.1pp KNN/LP gap. The
  user-independent evaluation was adopted as the honest benchmark: MAE-500 achieved 85.91% KNN /
  83.77% LP, a modest ~3.7pp improvement over the 82.2% supervised baseline.

Full historical tables are preserved in [RESULTS.md](RESULTS.md#legacy-fall-detection-results).

---

## References and Related Work

1. **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
2. **I-JEPA**: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture" (CVPR 2023)
3. **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
4. **CSI-Bench**: benchmark for WiFi-based sensing (7 tasks, including HumanActivityRecognition)

**Last updated**: July 2026