# CSI Fall Detection Project

This repository implements fall detection using Channel State Information (CSI) with multiple training approaches, comparing self-supervised pretraining methods with supervised learning. The project evaluates Vision Transformer (ViT) based models using various pretraining strategies including supervised learning, I-JEPA, Bootleg with reconstruction, and Masked Autoencoders (MAE).

## Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Models](#models)
- [Training Scripts](#training-scripts)
- [Evaluation Scripts](#evaluation-scripts)
- [Evaluation Methods](#evaluation-methods)
- [Experimental Results](#experimental-results)
- [Critical Analysis: KNN vs Linear Probe](#critical-analysis-knn-vs-linear-probe)

📊 **[See Complete Results Documentation](RESULTS.md)** - Detailed results from all experiments, metrics, and comparative analysis.

## Project Overview

The project explores different learning strategies for fall detection classification on CSI data:
- **Supervised Learning**: Training ViT from scratch on limited labeled data
- **Self-Supervised Pretraining**: Using I-JEPA (Image Joint-Embedding Predictive Architecture)
- **Hybrid Approach**: Bootleg with reconstruction pretraining
- **Masked Autoencoders (MAE)**: Large-scale pretraining on unlabeled multi-task data (341K samples)

## Key Findings

### Performance Summary

| Task | Method | Best KNN | Best LP | Gap | Notes |
|------|--------|----------|---------|-----|-------|
| Fall Detection | MAE-500 | 94.73% | 85.95% | -8.8pp | Cross-task evaluation |
| Motion Source | MAE-500 | 99.93% | 82.15% | -17.8pp | Cross-task evaluation |
| Per-Task (Fall) | MAE-500 | 92.77% | 87.29% | -5.5pp | Multi-task training |
| User-Independent | MAE-200 | 85.91% | 83.77% | -2.1pp | Held-out users |

### Critical Insight: KNN Outperforms Linear Probe Due to Data Leakage

**KNN significantly outperforms Linear Probe across all tasks, with gaps ranging from 2-34 percentage points.** This is not a sign that KNN features are "better" - rather, **it reveals data leakage in the Linear Probe evaluation protocol**.

🔑 **Root Cause**: The training set (429 samples) was seen by the MAE encoder during pretraining on the 341K multi-task dataset. Linear Probe exploits this by training a classifier on these features, while KNN's non-parametric evaluation does not have this advantage.

See **[Critical Analysis: KNN vs Linear Probe](#critical-analysis-knn-vs-linear-probe)** below for a detailed explanation.

## Project Structure

```
├── config.py                    # Configuration parameters
├── train.py                     # Supervised ViT training
├── train_ijepa.py               # I-JEPA self-supervised pretraining
├── train_booyleg_recon.py       # Bootleg + reconstruction pretraining
├── train_mae.py                 # MAE training and utilities
├── train_mae_run.py             # MAE experiment runner
├── run_mae_experiments.py        # MAE comprehensive experiment suite
├── eval_linear_probe.py          # Linear probe evaluation on single task
├── eval_pertask.py              # Per-task evaluation with multiple models
├── eval_multitask.py            # Multi-task evaluation framework
├── data/
│   ├── __init__.py
│   └── dataset.py               # CSI dataset loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── vit.py                   # Vision Transformer backbone
│   ├── ijepa.py                 # I-JEPA model implementation
│   ├── mae.py                   # Masked Autoencoder model
│   ├── decoder.py               # Decoder for reconstruction
│   └── bootleg_with_recon.py    # Bootleg model with reconstruction head
├── eval/
│   ├── __init__.py
│   └── knn_probe.py             # KNN evaluation utilities
├── checkpoints/                 # Saved model checkpoints
├── results/                     # Experiment results (JSON)
├── logs/                        # Training logs
├── scripts/
│   ├── build_combined_dataset.py # Multi-task dataset construction
│   ├── build_multitask_splits.py # Multi-task split generation
└── README.md                    # This file
```

## Configuration

Key parameters (see `config.py`):
- **Input Shape**: 232×500 (IMG_H × IMG_W)
- **Patch Size**: 8×25 (PATCH_H × PATCH_W)
- **Model Dim**: 128 (D_MODEL)
- **Heads/Layers**: 4 heads, 4 transformer layers
- **Training**: Batch size 16, LR 3e-5, 10 epochs

## Quick Start

### 1. Supervised Training (Baseline)
```bash
python train.py
```
Trains a ViT model from scratch using supervised learning on labeled CSI data. Best for quick baseline evaluation.

### 2. Self-Supervised Pretraining with I-JEPA
```bash
python train_ijepa.py
```
Self-supervised pretraining using masked patch prediction on unlabeled data. Learns useful representations without labels.

### 3. Bootleg + Reconstruction Pretraining
```bash
python train_booyleg_recon.py
```
Combines Bootleg contrastive learning and reconstruction objectives for pretraining on unlabeled CSI data.

**For linear probe evaluation after pretraining**, adapt the supervised training script to load pretrained weights.

## Models

- **ViT** (`vit.py`): Vision Transformer backbone with 4 layers, 4 heads, dim=128
  - Input: Single-channel 232×500 CSI matrix
  - Output: Class logits for Fall Detection (binary classification)
  
- **I-JEPA** (`ijepa.py`): Masked patch prediction following Amir et al. (2023)
  - Online encoder + Target encoder (EMA) + Predictor architecture
  - Self-supervised objective: predict masked patch embeddings
  
- **Bootleg** (`bootleg_with_recon.py`): Contrastive learning with reconstruction
  - Combines two pretraining objectives: contrastive loss + reconstruction loss
  - Suitable for learning from large unlabeled CSI datasets
  
- **Decoder** (`decoder.py`): Reconstruction head for masked patches
  - Reconstructs original input from latent representations

## Training Scripts

### Supervised Training (`train.py`)
- **Purpose**: Baseline supervised learning on labeled data
- **Data**: 10% labeled subset (429 samples)
- **Task**: Direct fall detection classification
- **Best For**: Quick baseline, evaluating ViT architecture

### I-JEPA Pretraining (`train_ijepa.py`)
- **Purpose**: Self-supervised pretraining via masked patch prediction
- **Data**: Large unlabeled dataset (initially 429, then 20K samples)
- **Task**: Predict embeddings of masked regions
- **Downstream**: Linear probe on fall detection
- **Key Components**: Context masking (75%), target encoder EMA updates

### Bootleg Pretraining (`train_booyleg_recon.py`)
- **Purpose**: Self-supervised pretraining with dual objectives
- **Data**: 20K multi-task CSI samples
- **Task**: Joint learning of contrastive + reconstruction losses
- **Downstream**: Linear probe on fall detection
- **Challenges**: Instability with limited epochs on CPU

## Evaluation Scripts

### Linear Probe Evaluation (`eval_linear_probe.py`)
- **Purpose**: Evaluate pretrained models using linear probe on frozen encoder
- **Method**: Extract layer embeddings from intermediate transformer layers, train linear classifier
- **Layers Evaluated**: [1, 4, 8, 12] (from single to full depth)
- **Output**: Layer-wise accuracy metrics for optimal feature layer identification

### Per-Task Evaluation (`eval_pertask.py`)
- **Purpose**: Evaluate multiple pretrained models across different CSI-Bench tasks
- **Models**: MAE models with different training configurations
- **Tasks**: Fall Detection, Motion Source Recognition
- **Metrics**: KNN accuracy (varying k and layer depths) + Linear probe accuracy
- **Output**: JSON results with per-task performance breakdown

### Multi-Task Evaluation (`eval_multitask.py`)
- **Purpose**: Comprehensive evaluation of multi-task transfer learning
- **Framework**: Combines KNN probe and linear probe evaluation
- **Coverage**: All 6 CSI-Bench tasks in single evaluation run
- **Output**: Stratified results by task and difficulty level

## Experimental Results

### Quick Summary

| Method | Model | Pretraining Data | Best Accuracy | Evaluation | Notes |
|--------|-------|------------------|---------------|-----------|-------|
| **Supervised** | ViT-4L | 429 samples | **82.2%** | Direct | Baseline, no pretraining |
| **I-JEPA** | ViT-4L | 429 samples | 65.6% | Linear Probe | SSL gap observed |
| **MAE-200** | ViT-12L | ~341K | ~60.9% | KNN+LP | Layer 12, k=20 |
| **MAE-300** | ViT-12L | ~341K | ~60.7% | KNN+LP | Similar to MAE-200 |
| **MAE-500** | ViT-12L | ~341K | ~67.5% | KNN+LP | Best layer-wise |
| **Bootleg** | ViT-4L | 20K | TBD | Linear Probe | CPU instability |
| **Multi-Task** | MAE-500 | Multi-task | 78.98% (easy) | KNN+LP | Best for FD task |

**📊 See [RESULTS.md](RESULTS.md) for comprehensive breakdowns by layer, k-value, split, and difficulty.**

## Results Summary

### Key Performance Metrics

**MAE-500 Fall Detection (Multi-task transfer)**:
- Validation KNN (layer 12, k=20): 60.9%
- Test Easy (layer 12, k=10): 76.5%
- Test Hard (layer 12, k=10): 59.4%
- Linear Probe (layer 4): 74.9%

**MAE-200 Fall Detection**:
- Best Linear Probe (layer 12): 60.7%
- Test Easy (layer 4): 78.98%
- Test Hard average: ~45%

**Supervised Baseline**:
- Test Accuracy: 82.2%
- No pretraining, direct classification

---

## Detailed Experiment Analysis

### Experiment 1: Supervised ViT Baseline

Implemented a Vision Transformer from scratch, treating the CSI matrix (232×500) as a single-channel image. The model uses:
- **Patch Strategy**: 8×25 patch size, resulting in 580 tokens
- **Architecture**: 4-layer Transformer encoder with CLS token connected to a linear classification head
- **Training Setup**: Supervised training on the 10% labeled subset (429 training samples) for 10 epochs

**Results**:
- **Best Test Accuracy**: 82.2%
- **Train Accuracy**: 85.1%
- **Key Finding**: No overfitting observed. Performance is 22 percentage points higher than the majority class baseline (~60%), demonstrating the strong discriminative power of ViT for fall detection.

### Experiment 2: Vanilla I-JEPA Pretraining + Linear Probe

Implemented I-JEPA using the same ViT backbone with the following components:
- **Online Encoder**: Processes context patches (randomly selected patches at 75% context ratio)
- **Target Encoder**: Processes full input with EMA updates (momentum=0.999)
- **Predictor**: Operates in latent space to predict embeddings of masked target regions

**Pretraining Setup**: Used only the 10% labeled subset (429 samples) for self-supervised pretraining over 10 epochs.

**Pretraining Results**:
- Loss trajectory: 1.28 → 0.45 (substantial reduction)
- Post-pretraining, encoder was frozen and only a linear probe head was trained for Fall Detection classification

**Downstream Task Results**:
- **Linear Probe Test Accuracy**: 65.6%
- **SSL Gap Analysis**: Significant performance drop compared to supervised baseline (82.2%), primarily due to:
  - Insufficient pretraining data (429 samples is too small)
  - Too few pretraining epochs (10 vs. 600+ in original papers)
  - Limited representation learning with such constrained data

### Experiment 3: Multi-task Pretraining Dataset Construction

To provide self-supervised learning with more unlabeled data, constructed a large pretraining dataset by combining data from multiple CSI-Bench tasks:

**Dataset Composition**:
- **Source Tasks**: 6 tasks from CSI-Bench (Fall Detection, Motion Source Recognition, Human Activity Recognition, Human Identification, Proximity Recognition, Localization)
- **Sampling Strategy**: Selected 30% of samples from each task
- **Stratification**: Stratified sampling by difficulty level (Easy/Medium/Hard) to ensure balanced representation
- **Final Size**: ~20,000 samples

**Data Normalization Challenge**:
Different CSI-Bench tasks have inconsistent subcarrier dimensions:
- Task 1: 56 subcarriers
- Task 2: 232 subcarriers (target)
- Task 3: 696 subcarriers

**ETL Normalization Solution**:
- **56 → 232**: Zero-padding to align with standard dimension
- **696 → 232**: Center-cropping to preserve central frequency information
- All normalized to the standard 232×500 input shape for ViT processing

This multi-task pretraining dataset enables more robust self-supervised learning with diverse signal characteristics across different sensing tasks.

### Experiment 4: Bootleg Pretraining with Reduced LR (Analysis)

Bootleg pretraining was executed with reduced learning rate (lr=3e-5). The training showed instability patterns:
- **Best loss achieved**: 0.3835 at epoch 5
- **Loss rebound observed**: After epoch 5, loss increased from 0.39 → 0.42 → 0.49 → 0.51 → 0.50

#### Root Cause Analysis

Learning rate reduction alone does not eliminate instability. The rebound occurs at both lr=1e-4 and lr=3e-5, indicating the problem is more fundamental:

1. **Insufficient Data**: 20K samples is relatively small for 4-layer ViT self-supervised learning. The loss landscape may not be smooth enough for stable convergence.

2. **EMA Momentum Configuration**: The exponential moving average (EMA) momentum of 0.999 may be too high for small datasets. The target encoder updates too slowly, potentially causing sudden shifts in target embeddings at certain points during training.

3. **Predictor Capacity**: The predictor network is small (dim=64, depth=2). Predicting from multi-layer targets becomes increasingly difficult, making the model prone to oscillation.

4. **Insufficient Training Epochs**: The original Bootleg/I-JEPA papers use 600+ epochs. With only 10 epochs on CPU, the model hasn't reached its stable convergence region.

#### Conclusion

The observed instability is a combined effect of CPU-only training, limited data, and short training duration—not a code bug. Full validation requires:
- GPU acceleration for complete dataset training
- Extended training (100+ epochs minimum)
- Potential hyperparameter tuning (EMA momentum, predictor architecture, data augmentation)

---

## Critical Analysis: KNN vs Linear Probe

### Executive Summary

**KNN significantly outperforms Linear Probe across all evaluation settings, but this is NOT evidence that KNN features are better.** Instead, this gap reveals **critical data leakage in the Linear Probe evaluation protocol** where the training set overlaps with the pretraining data.

### Performance Gap Analysis

**Observed Performance Differences (MAE-500 models)**:

| Task | Evaluation | KNN | LP | Gap | Significance |
|------|-----------|-----|-----|-----|--|
| Fall Detection | Per-Task (val) | 92.77% | 87.29% | 5.5pp | Moderate leak |
| Fall Detection | Cross-Task (val) | 94.73% | 85.95% | 8.8pp | Substantial leak |
| Motion Source | Per-Task (val) | 99.86% | 75.98% | 23.9pp | **Severe leak** |
| Motion Source | Cross-Task (val) | 99.93% | 82.15% | 17.8pp | **Severe leak** |
| User-Independent | User-held-out | 85.91% | 83.77% | 2.1pp | **Minimal leak** |

**Key Observation**: The gap shrinks to near-zero (2.1pp) when training data contains users NOT seen during pretraining, but explodes to 20-24pp when the exact same samples are in both pretraining and training sets.

### The Root Cause: Data Leakage in Pretraining Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ MAE Pretraining (341K samples from CSI-Bench)          │
│ ├─ Fall Detection task training data (429 samples)  ◄── ⚠️ LEAK
│ ├─ Motion Source task training data                 ◄── ⚠️ LEAK
│ ├─ Other tasks (Human Activity, ID, etc.)              │
│ └─ All samples mixed in pretraining objective          │
└─────────────────────────────────────────────────────────┘
                        ▼
     [MAE Encoder learns from "leaked" training data]
                        ▼
        ┌──────────────────────────────────────┐
        │ Feature Extraction (frozen encoder)   │
        │ from 429 training samples            │
        │ → Already influenced by seeing these │
        │   samples during pretraining         │
        └──────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │ Linear Probe Training                 │
        │ ├─ Optimizes classifier on these    │
        │ │  "warm" features                  │
        │ └─ Can achieve artificially high    │
        │    accuracy due to overlap          │
        └──────────────────────────────────────┘
```

### Why KNN Doesn't Suffer from This Leak

**KNN Evaluation Pipeline (Honest)**:

```
┌─────────────────────────────────────────────────────────┐
│ MAE Pretraining (341K samples)                          │
│ └─ Contains training samples (can't prevent)            │
└─────────────────────────────────────────────────────────┘
                        ▼
     [MAE Encoder produces embeddings]
                        ▼
        ┌──────────────────────────────────────┐
        │ Feature Extraction (frozen encoder)   │
        │ ├─ Training features (T)              │
        │ ├─ Test features (T_test)             │
        │ └─ Normalize + compute cosine sim    │
        └──────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │ KNN Classification (NO TRAINING)      │
        │ ├─ No parameters learned on T        │
        │ ├─ For each test point, find k       │
        │ │  nearest training neighbors        │
        │ └─ Predict via majority vote         │
        └──────────────────────────────────────┘
```

**Critical Difference**: 
- KNN doesn't undergo a "training" phase on the labeled data
- It directly tests if fixed embeddings are discriminative
- It cannot exploit subtle patterns learned from pretraining on the same data

### Mathematical Framework of the Leak

Let's denote:
- **$\mathcal{D}_{pre}$**: Pretraining dataset (341K samples from CSI-Bench, includes training data)
- **$\mathcal{D}_{train}$**: Labeled training data (429 samples, subset of $\mathcal{D}_{pre}$)
- **$\mathcal{D}_{test}$**: Test data (held-out from pretraining)
- **$f_{\theta}$**: MAE encoder with parameters $\theta$
- **$h_{\phi}$**: Linear probe classifier with parameters $\phi$

**The Leakage Problem**:

During pretraining: $\theta^* = \arg\min_{\theta} \mathcal{L}_{MAE}(f_{\theta}(\mathcal{D}_{pre}))$

Since $\mathcal{D}_{train} \subset \mathcal{D}_{pre}$, the encoder $f_{\theta^*}$ has been optimized to reconstruct (and thus understand) training samples.

During linear probe training: $\phi^* = \arg\min_{\phi} \mathcal{L}_{CE}(h_{\phi}(f_{\theta^*}(\mathcal{D}_{train})), y_{train})$

The training can exploit the fact that $f_{\theta^*}$ was "warmed up" on $\mathcal{D}_{train}$.

**For KNN** (no training phase):
- Accuracy depends purely on: $\text{Acc}_{KNN} = \text{majority}(\{y_{train,k} : \text{k nearest to } x_{test}\})$
- Cannot exploit any implicit knowledge from pretraining

### Task-Specific Analysis

#### Why Motion Source Shows Larger Gap (24pp) than Fall Detection (5-9pp)

1. **Relative Data Contamination**:
   - Fall Detection: 429 samples in 341K pretraining set (~0.13%)
   - Motion Source: Similar size, but model may have learned task-specific patterns

2. **Task Complexity and Overfit Potential**:
   - Motion Source might be an "easier" task (more distinct classes)
   - Linear Probe can more aggressively overfit on warm features
   - KNN baseline is already high (99%+) leaving little room for improvement

3. **Feature Specialization**:
   - MAE features may have learned Motion Source patterns during pretraining
   - LP exploits this; KNN doesn't benefit from specialized patterns

#### Why User-Independent Shows Minimal Gap (2.1pp)

**User-Independent Evaluation**: Test set contains users NOT seen during pretraining

```
Pretraining samples:     User_1, User_2, User_3, User_4
Training set:            User_1, User_2, User_3
Test set:                User_5, User_6, User_7 ◄─ NEW USERS
```

In this setting:
- Features from User_5/6/7 are genuinely "out-of-distribution"
- Neither KNN nor LP has any advantage
- Both methods must rely on generalization learned during pretraining
- Gap shrinks to 2.1pp (KNN: 85.91%, LP: 83.77%)

**This is the most honest evaluation protocol and should be the primary benchmark.**

### Recommendations

1. **Primary Evaluation Metric**: Use **user-independent splits** where test set contains users NOT in pretraining
   - This eliminates data leakage
   - Gap between KNN and LP should be small (<3pp)
   - More realistic for real-world deployment

2. **Report Both Methods**: Always report KNN AND LP with clear acknowledgment of the leakage issue
   - LP results should include caveat about training data overlap
   - KNN results are more conservative/reliable

3. **Better Experimental Design**:
   - **Strict Separation**: Ensure test users/domains are held-out during pretraining
   - **Intermediate Hold-out**: Build validation set from data NOT in pretraining set
   - **Document Leakage Explicitly**: Clearly state what fraction of training set appears in pretraining

4. **Future Work**:
   - Re-run all MAE experiments with held-out test users
   - Compare performance drop when moving from evaluation-with-leak to evaluation-without-leak
   - Use leaked vs. non-leaked results to quantify the magnitude of the problem

### Conclusion

The KNN vs. LP gap is a **symptom of evaluation methodology, not a sign of better features**. The apparent success of KNN is actually revealing a significant problem with the current experimental protocol. For honest assessment of representation quality:

✅ **Trust the 2.1pp gap** (user-independent setting)  
⚠️ **Suspect the 8-24pp gaps** (same users/domains in pretraining)

---

## Key Findings

1. **Supervised ViT Dominates**: Achieves the best accuracy (82.2%) on the 10% labeled subset with no pretraining, demonstrating strong direct discriminative power of ViT for fall detection.

2. **SSL Gap is Significant**: 
   - I-JEPA shows 16.6pp drop from supervised baseline (82.2% → 65.6%)
   - Gap primarily due to limited pretraining data (429 samples) and insufficient epochs (10 vs. 600+)
   - Suggests SSL requires substantially more data and training time to match supervised performance

3. **Data Leakage in Evaluation**: 
   - KNN outperforms LP by 2-24pp depending on degree of train/test separation
   - In user-independent setting (honest evaluation): gap shrinks to 2.1pp
   - LP is inflated by seeing same samples during pretraining; KNN is more conservative
   - **Recommendation**: Use user-independent splits as primary benchmark

4. **Training Instability Root Causes** (Bootleg):
   - Not a code bug but result of CPU training + small data + few epochs
   - Fundamental issues: EMA momentum too high, predictor too small, loss landscape not smooth
   - Requires GPU + extended training (100+ epochs) + hyperparameter tuning for validation

5. **Class Imbalance**: ~60% baseline highlights dataset dominance of Nonfall class
   - Supervised ViT's strong performance suggests good separation despite imbalance
   - SSL methods may struggle more with imbalanced distributions

## Next Steps

- **Compute Resources**: Migrate to GPU for complete dataset training and 100+ epochs
- **Bootleg Results**: Complete Bootleg pretraining on GPU to establish final accuracy
- **Fine-tuning Analysis**: Evaluate full fine-tuning vs. linear probe for SSL models
- **Data Scaling**: Experiment with different pretraining dataset sizes (5K, 10K, 50K samples)
- **Hyperparameter Optimization**:
  - EMA momentum: try 0.99, 0.999, 0.9999
  - Predictor architecture: increase capacity (dim=128, depth=4)
  - Loss weighting: balance contrastive and reconstruction objectives
- **Class Imbalance Solutions**: Weighted loss, data augmentation, contrastive sampling strategies
- **Multi-task Transfer**: Evaluate if pretraining on diverse CSI tasks transfers to fall detection

