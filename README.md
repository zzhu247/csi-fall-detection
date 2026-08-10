# CSI Fall Detection and CSI-Bench SSL Project

This repository contains a WiFi CSI self-supervised learning project centered on Masked Autoencoder (MAE) and MAEv2 pretraining for CSI-Bench tasks. The active work focuses on Human Activity Recognition (HAR) and Human Identification (HID), while the codebase also retains older fall-detection and legacy SSL experiments.

The project is organized around:

- CSI-Bench MAE/MAEv2 pretraining on official task splits
- Frozen-feature evaluation using KNN, linear probe, and MLP probe
- Open-set and closed-set evaluation for Human Identification
- Baseline comparisons against CSI-Bench reference models
- Legacy fall-detection experiments using earlier SSL approaches

See [RESULTS.md](RESULTS.md) for experiment summaries and outcome tables.

## Table of Contents

- [Project Overview](#project-overview)
- [Current Active Workstreams](#current-active-workstreams)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Data Setup](#data-setup)
- [Quick Start](#quick-start)
- [Training Pipelines](#training-pipelines)
- [Evaluation Pipelines](#evaluation-pipelines)
- [Important Notes and Caveats](#important-notes-and-caveats)
- [Legacy Experiments](#legacy-experiments)

---

## Project Overview

The repository uses WiFi CSI amplitude tensors shaped as 232 subcarriers by 500 time steps, converted into patch tokens and trained with a masked autoencoding objective. The encoder is then evaluated through downstream probes and retrieval-based metrics.

The main active tasks are:

- Human Activity Recognition (HAR)
  - Official CSI-Bench splits: test_id, test_cross_device, test_cross_env, test_cross_user
  - MAE/MAEv2 pretraining and representation evaluation
- Human Identification (HID)
  - Train-id + OOD split evaluation
  - Closed-set and open-set identity metrics

This project is not a single end-to-end script. It is a research codebase with multiple training and evaluation entry points, each tuned to a specific task or protocol.

---

## Current Active Workstreams

### 1. HAR MAE pretraining and OOD evaluation

Primary training script:

- [train/train_mae_har.py](train/train_mae_har.py)

This script performs:

- CSI-Bench HAR pretraining
- Random masking and structured masking strategies
- Layer-wise feature extraction
- KNN, linear probe, and MLP probe evaluation
- Optional domain-adversarial training hooks
- Attention memory safety checks before starting training

Key hyperparameters include:

- `--epochs`
- `--mask_ratio`
- `--encoder_depth`
- `--encoder_dim`
- `--decoder_dim`
- `--batch_size`
- `--patch_h` and `--patch_w`
- `--mask_strategy` with choices: `random`, `time`, `freq`, `mixed`, `2d`

### 2. Human Identification self-supervised training

Primary training script:

- [train/train_mae_hid.py](train/train_mae_hid.py)

This is a separate training pipeline for the HID task and includes:

- MAE/MAEv2 pretraining on HumanIdentification splits
- Closed-set evaluation on known identities
- Open-set evaluation for unknown identities
- Rank-1 retrieval, verification AUC, and EER metrics

### 3. Benchmark and comparison scripts

Relevant evaluation scripts:

- [eval/eval_har_ood_v2.py](eval/eval_har_ood_v2.py)
- [eval/eval_hid_device_closedset.py](eval/eval_hid_device_closedset.py)
- [eval/eval_hid_openset.py](eval/eval_hid_openset.py)
- [eval/eval_linear_probe.py](eval/eval_linear_probe.py)
- [eval/eval_multitask.py](eval/eval_multitask.py)
- [eval/eval_user_independent.py](eval/eval_user_independent.py)

---

## Repository Layout

```text
.
├── config.py                              # Shared config and dataset root
├── README.md                              # Project documentation
├── RESULTS.md                             # Summary of experiment results
├── data/
│   └── dataset.py                         # CSI loading, normalization, dataset wrappers
├── train/
│   ├── train_mae_har.py                  # Main HAR pretraining pipeline
│   ├── train_mae_hid.py                  # Main HID training pipeline
│   ├── train_mae.py                      # Older MAE training utilities
│   ├── train_mae_run.py                  # Multi-run experiment orchestration
│   ├── train_ijepa.py                    # Legacy I-JEPA training code
│   ├── train_ablation.py                 # Ablation-related training script
│   └── train_booyleg_recon.py            # Legacy reconstruction-based experiment
├── eval/
│   ├── eval_har_ood_v2.py                # CSI-Bench HAR OOD baseline eval
│   ├── eval_hid_device_closedset.py      # HID known-identity closed-set eval
│   ├── eval_hid_openset.py               # HID open-set eval with retrieval/verification
│   ├── eval_linear_probe.py              # Linear probe helper
│   ├── eval_multitask.py                 # Multi-task evaluation
│   ├── eval_pertask.py                   # Per-task evaluation
│   ├── eval_user_independent.py          # User-held-out evaluation
│   ├── eval_cross_task.py                # Cross-task evaluation
│   └── knn_probe.py                      # KNN evaluation utilities
├── models/
│   ├── mae.py                            # Standard MAE implementation
│   ├── mae_v2.py                         # MAEv2 masked strategies
│   ├── vit.py                            # Patch embedding + encoder blocks + attention
│   ├── csibench_models.py                # Baselines: MLP, LSTM, ResNet18, ViT, etc.
│   ├── decoder.py                        # Decoder utilities
│   ├── resnet.py                         # Additional model definitions
│   ├── baselines.py                      # Baseline modules
│   ├── ijepa.py                          # Legacy SSL module
│   └── bootleg_with_recon.py             # Legacy reconstruction experiment
├── scripts/
│   ├── run_mae_experiments.py            # Sweep helper for older MAE runs
│   ├── run_all_baselines.sh              # Baseline benchmark launcher
│   ├── launch_mask_strategy_enc12.sh     # Background job launcher
│   ├── launch_patch_size_ablation.sh     # Patch-size sweep launcher
│   ├── launch_patch_size_ablation_v2.sh  # Additional patch-size sweep
│   ├── run_mask_ablation.sh              # Mask ablation runner
│   ├── run_strategy_ablation.sh          # Strategy ablation runner
│   └── run_norm_comparison.py            # Normalization comparison script
├── checkpoints/
│   ├── har_models/                       # HAR baseline checkpoints
│   ├── mae_har/                          # HAR MAE checkpoints
│   ├── mae_hid/                          # HID MAE checkpoints
│   └── ablation/                         # Ablation checkpoints
├── results/
│   ├── mae_har/                          # HAR result JSONs and metrics
│   ├── mae_hid/                          # HID result JSONs and metrics
│   ├── csibench_official/               # Official benchmark outputs
│   └── figures/                          # Aggregated result figures
├── figs/
│   ├── embeddings/                       # Embedding visualization outputs
│   └── ...                                # Result plots for ablation runs
├── visualization/
│   ├── visualize_embeddings.py           # Embedding geometry diagnostics
│   ├── visualize_mask_ablation.py        # Mask ablation plotter
│   ├── visualize_results.py              # Result plotting tool
│   └── *.ipynb                          # Analysis notebooks
├── config.py                             # Root config file
├── train.pid                             # PID file for training runs
├── logs/                                 # Training logs
└── data/                                 # CSI benchmark dataset root (external)
```

---

## Environment Setup

This project is built for a Python environment with PyTorch and standard scientific Python packages.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio numpy pandas scikit-learn h5py matplotlib seaborn
```

If you are using a specific local environment or conda environment, ensure the interpreter matches the one used by your editor.

Core repository config is in [config.py](config.py):

- `DATA_ROOT = "/home/zhuzih19/data/csi-bench-dataset"`
- default image shape is 232 x 500
- standard MAE params are defined there as defaults

---

## Data Setup

The project expects the CSI-Bench dataset to live at:

```text
/home/zhuzih19/data/csi-bench-dataset
```

The dataset layout is expected to match the CSI-Bench format used by the code in [data/dataset.py](data/dataset.py), including task folders such as:

- `Multitask/HumanActivityRecognition/...`
- `Multitask/HumanIdentification/...`
- metadata and split JSON files under each task folder

The dataset loader normalizes each CSI sample to 232x500 and applies per-sample standardization before returning tensors.

---

## Quick Start

### 1. Train HAR MAE

```bash
python train/train_mae_har.py \
  --epochs 300 \
  --mask_ratio 0.75 \
  --encoder_depth 6 \
  --encoder_dim 128 \
  --decoder_dim 64 \
  --batch_size 128 \
  --patch_h 29 \
  --patch_w 25 \
  --seed 42
```

### 2. Train Human Identification MAE

```bash
python train/train_mae_hid.py \
  --epochs 300 \
  --mask_ratio 0.75 \
  --encoder_depth 12 \
  --encoder_dim 128 \
  --decoder_dim 64 \
  --batch_size 128 \
  --patch_h 29 \
  --patch_w 25 \
  --seed 42
```

### 3. Run a closed-set HID evaluation on known identities

```bash
python eval/eval_hid_device_closedset.py \
  --checkpoint checkpoints/mae_hid/mae_hid_ep150_mask0.75_strategyrandom_ph29pw25_seed42_enc12_dim128_bs128_best.pt \
  --result_json results/mae_hid/mae_hid_ep150_mask0.75_strategyrandom_ph29pw25_seed42_enc12_dim128_bs128.json \
  --layers 1,3,6,9,12
```

### 4. Run open-set HID evaluation

```bash
python eval/eval_hid_openset.py \
  --checkpoint checkpoints/mae_hid/mae_hid_ep150_mask0.75_strategyrandom_ph29pw25_seed42_enc12_dim128_bs128_best.pt \
  --result_json results/mae_hid/mae_hid_ep150_mask0.75_strategyrandom_ph29pw25_seed42_enc12_dim128_bs128.json \
  --layers 1,3,6,9,12
```

### 5. Run CSI-Bench baseline evaluation

```bash
python eval/eval_har_ood_v2.py
```

---

## Training Pipelines

### HAR training flow

The active HAR pipeline is driven by [train/train_mae_har.py](train/train_mae_har.py).

The script includes:

- data split loading from CSI-Bench task metadata
- MAE or MAEv2 model construction
- optional mask strategy selection
- attention-memory guard before training
- feature extraction at target layers
- KNN / LP / MLP evaluation across official splits
- checkpoint saving under `checkpoints/mae_har/`

### HID training flow

The active HID pipeline is in [train/train_mae_hid.py](train/train_mae_hid.py).

This script extends the HAR pipeline for identity learning and evaluates:

- test_id
- test_cross_device
- test_cross_env
- test_cross_user

It also distinguishes between:

- closed-set classification on known identities
- open-set identity retrieval and verification when the split contains identities unseen during training

---

## Evaluation Pipelines

### KNN evaluation

Implemented in [eval/knn_probe.py](eval/knn_probe.py)

The evaluation workflow typically does the following:

1. extract frozen embeddings at a selected encoder layer
2. normalize embeddings for cosine similarity
3. compare query samples against train-set features
4. report prediction accuracy and F1 when applicable

### Linear probe and MLP probe

These are used to test how linearly or nonlinearly separable the features are after pretraining.

Typical files:

- [eval/eval_linear_probe.py](eval/eval_linear_probe.py)
- [eval/eval_hid_device_closedset.py](eval/eval_hid_device_closedset.py)

### Open-set identity evaluation

Used for HID splits with unseen identities.

Relevant file:

- [eval/eval_hid_openset.py](eval/eval_hid_openset.py)

This includes metrics such as:

- Rank-1 retrieval accuracy
- verification AUC
- EER

---

## Important Notes and Caveats

### Memory safety

The attention implementation in [models/vit.py](models/vit.py) is a naive, non-flash attention block. The attention score tensor scales quadratically with the number of patches and is retained during backward across all layers. This is why the training scripts include `check_attention_memory()` before training starts.

If patch sizes are too small or the batch size is too large, training can run out of memory mid-run. The recommended safer patch sizes are in the 11–21 range for the current encoder design.

### Data conventions

The dataset loader treats each CSI sample as a 2D array and normalizes it before conversion to tensor format. All samples are standardized per sample before they are used for pretraining or evaluation.

### Split semantics

The code differentiates between:

- in-distribution evaluation (`test_id`)
- out-of-distribution evaluation (`test_cross_device`, `test_cross_env`, `test_cross_user`)
- closed-set vs open-set identity evaluation

This distinction matters for interpreting results and should not be ignored when reporting metrics.

### Result files

Results are stored in:

- [results/mae_har](results/mae_har)
- [results/mae_hid](results/mae_hid)
- [results/csibench_official](results/csibench_official)

Checkpoints are saved under:

- [checkpoints/mae_har](checkpoints/mae_har)
- [checkpoints/mae_hid](checkpoints/mae_hid)

Large model artifacts are not intended to be committed to git.

---

## Legacy Experiments

This repo still contains earlier non-core experiments from the fall-detection and SSL research track.

These include:

- [train/train_ijepa.py](train/train_ijepa.py)
- [train/train_ablation.py](train/train_ablation.py)
- [train/train_booyleg_recon.py](train/train_booyleg_recon.py)
- [models/ijepa.py](models/ijepa.py)
- [models/bootleg_with_recon.py](models/bootleg_with_recon.py)
- [finetune_mae_har.py](finetune_mae_har.py)

These are kept for historical and experimental reference, but the current active pipeline is the HAR/HID MAE work.

---

## Useful References in This Repo

- [config.py](config.py)
- [data/dataset.py](data/dataset.py)
- [train/train_mae_har.py](train/train_mae_har.py)
- [train/train_mae_hid.py](train/train_mae_hid.py)
- [eval/eval_hid_device_closedset.py](eval/eval_hid_device_closedset.py)
- [eval/eval_hid_openset.py](eval/eval_hid_openset.py)
- [RESULTS.md](RESULTS.md)

If you are starting from scratch, the best entry points are the HAR training script and the HID training script, followed by the relevant evaluation script for the split and metric you need.

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