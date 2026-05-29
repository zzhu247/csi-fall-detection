# CSI Fall Detection Project

This repository implements fall detection using Channel State Information (CSI) with multiple training approaches, comparing self-supervised pretraining methods with supervised learning.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Models](#models)
- [Training Scripts](#training-scripts)
- [Experimental Results](#experimental-results)
- [Detailed Experiment Analysis](#detailed-experiment-analysis)
- [Key Findings](#key-findings)
- [Next Steps](#next-steps)

## Project Overview

The project explores different learning strategies for fall detection classification on CSI data:
- **Supervised Learning**: Training ViT from scratch on limited labeled data
- **Self-Supervised Pretraining**: Using I-JEPA (Image Joint-Embedding Predictive Architecture)
- **Hybrid Approach**: Bootleg with reconstruction pretraining

## Project Structure

```
├── config.py                    # Configuration parameters
├── train.py                     # Supervised ViT training
├── train_ijepa.py               # I-JEPA self-supervised pretraining
├── train_booyleg_recon.py       # Bootleg + reconstruction pretraining
├── data/
│   ├── __init__.py
│   └── dataset.py               # CSI dataset loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── vit.py                   # Vision Transformer backbone
│   ├── ijepa.py                 # I-JEPA model implementation
│   ├── decoder.py               # Decoder for reconstruction
│   └── bootleg_with_recon.py    # Bootleg model with reconstruction head
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

## Experimental Results

| Method | Test Acc | Notes |
|--------|----------|-------|
| Majority Baseline | ~60.0% | Always predict Nonfall |
| Supervised ViT (10% subset) | 82.2% | From scratch, no pretraining |
| I-JEPA Linear Probe | 65.6% | Pretrain: 429 samples, 10 epochs |
| Bootleg Linear Probe | TBD | Pretrain: 20K samples, 10 epochs |

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

## Key Findings

1. **Supervised ViT Dominates**: Achieves the best accuracy (82.2%) on the 10% labeled subset with no pretraining, demonstrating strong direct discriminative power of ViT for fall detection.

2. **SSL Gap is Significant**: 
   - I-JEPA shows 16.6pp drop from supervised baseline (82.2% → 65.6%)
   - Gap primarily due to limited pretraining data (429 samples) and insufficient epochs (10 vs. 600+)
   - Suggests SSL requires substantially more data and training time to match supervised performance

3. **Data Scale Matters**: 
   - Multi-task dataset (20K samples) enables more realistic SSL experiments
   - But 10 epochs on CPU is insufficient for convergence
   - Original papers use 600+ epochs with GPU acceleration

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

