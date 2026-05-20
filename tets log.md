# CSI Fall Detection — ViT & I-JEPA Demo

## Overview
A mini demo applying ViT and I-JEPA to WiFi CSI signal classification,
using the CSI-Bench FallDetection dataset as a practice exercise before
the main lab research project.

## Dataset
- **Source**: CSI-Bench (Kaggle)
- **Task**: Fall Detection (binary: Fall / Nonfall)
- **Device filter**: HP only (consistent 232 subcarriers)
- **Split**: Easy split (train/test)
- **Subset used**: 10% (429 train / 90 test)

## Data Format
Each sample is a WiFi CSI measurement:
- Shape: `(232, 500)` → 232 subcarriers × 500 time steps
- Treated as a single-channel image `[1, 232, 500]` for ViT

## Model: ViT (Supervised Baseline)
- Patch size: `8 × 25` → 580 tokens per sample
- d_model: 128, d_ff: 512, heads: 4, layers: 4
- Training: 10 epochs, Adam lr=1e-4, CrossEntropyLoss

### Results
| Metric              | Value |
|---------------------|-------|
| Best Test Accuracy  | 82.2% |
| Final Train Accuracy| 85.1% |
| Majority Baseline   | ~60%  |

No overfitting observed — train and test loss decreased together.

## Model: I-JEPA (Self-Supervised Pretraining + Linear Probe)
- Same ViT backbone as above
- Pretraining: predict masked patch embeddings in latent space (no pixel reconstruction)
- EMA momentum: 0.996
- Predictor: smaller transformer (dim=64, heads=2, depth=2)
- Linear Probe: freeze encoder, train only MLPHead

### Results
| Metric                        | Value |
|-------------------------------|-------|
| Pretrain Loss (epoch 1 → 10)  | 1.28 → 0.45 |
| Linear Probe Test Accuracy    | 65.6% |
| Supervised Baseline           | 82.2% |
| Majority Baseline             | ~60%  |

### Why the gap?
- Only 429 samples for pretraining (SSL needs much more data)
- Only 10 epochs (original I-JEPA uses 600+ epochs on ImageNet)
- Linear probe is the strictest evaluation — encoder completely frozen
- Expected behavior: gap closes with more data and longer pretraining

## Key Learnings
- CSI data naturally maps to a 2D image (subcarrier × time)
- I-JEPA predicts in **latent space**, not pixel space (unlike MAE)
- SSL representation quality depends heavily on pretraining scale
- Linear probe is a standard SSL evaluation benchmark

## Project Structure