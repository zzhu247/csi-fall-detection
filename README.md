# CSI Fall Detection Project

This repository implements fall detection using Channel State Information (CSI) with multiple training approaches, comparing self-supervised pretraining methods with supervised learning.

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

## Experimental Results

| Method | Test Acc | Notes |
|--------|----------|-------|
| Majority Baseline | ~60.0% | Always predict Nonfall |
| Supervised ViT (10% subset) | 82.2% | From scratch, no pretraining |
| I-JEPA Linear Probe | 65.6% | Pretrain: 429 samples, 10 epochs |
| Bootleg Linear Probe | TBD | Pretrain: 20K samples, 10 epochs |

### Key Findings

- **Supervised ViT** achieves the best accuracy (82.2%) on the 10% labeled subset
- **I-JEPA** shows lower performance (65.6%) with limited pretraining data
- **Bootleg** pretraining uses substantially more data (20K samples) - results pending
- Baseline of ~60% highlights the class imbalance in the dataset (majority is Nonfall)

## Training Scripts

### Supervised Training
```bash
python train.py
```
Trains a ViT model from scratch using supervised learning on labeled CSI data.

### I-JEPA Pretraining
```bash
python train_ijepa.py
```
Self-supervised pretraining using masked patch prediction. Learns useful representations without labels.

### Bootleg + Reconstruction Pretraining
```bash
python train_booyleg_recon.py
```
Combines Bootleg and reconstruction objectives for pretraining on unlabeled CSI data.

## Models

- **ViT**: Vision Transformer backbone for feature extraction
- **I-JEPA**: Masked patch prediction following Amir et al. (2023)
- **Bootleg**: Contrastive learning approach
- **Decoder**: Reconstruction head for masked patches

## Usage

1. Ensure CSI data is available at the path specified in `config.py`
2. Run the desired training script
3. For linear probe evaluation after pretraining, adapt the supervised training script to load pretrained weights

## Next Steps

- Complete Bootleg pretraining experiments
- Evaluate fine-tuning performance on pretrained models
- Experiment with different pretraining data scales
- Investigate class imbalance solutions (weighted loss, data augmentation)
