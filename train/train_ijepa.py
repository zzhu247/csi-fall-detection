# train_ijepa.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import json
import pandas as pd

import config
from models.ijepa import IJEPA
from data.dataset import CSIFallDataset, load_metadata, get_splits


def sample_masks(num_patches, context_ratio=0.75, num_target_blocks=4, target_block_size=16):
    """
    Sample context and target masks following I-JEPA strategy.

    Context mask: randomly keep context_ratio of patches
    Target mask:  several contiguous blocks (to predict semantically meaningful regions)

    Args:
        num_patches:       total number of patches (e.g. 580 for CSI)
        context_ratio:     fraction of patches used as context
        num_target_blocks: how many target blocks to sample
        target_block_size: size of each target block

    Returns:
        context_mask:   [num_patches] boolean tensor, True = context patch
        target_indices: [num_target_patches] int tensor, indices to predict
    """
    all_indices = torch.randperm(num_patches)

    # Sample target blocks as contiguous regions
    target_set = set()
    for _ in range(num_target_blocks):
        start = torch.randint(0, num_patches - target_block_size, (1,)).item()
        block = list(range(start, start + target_block_size))
        target_set.update(block)
    target_indices = torch.tensor(sorted(target_set), dtype=torch.long)

    # Context = random subset of non-target patches
    non_target = [i for i in range(num_patches) if i not in target_set]
    n_context  = int(len(non_target) * context_ratio)
    context_indices = torch.tensor(non_target[:n_context], dtype=torch.long)

    # Build boolean context mask
    context_mask = torch.zeros(num_patches, dtype=torch.bool)
    context_mask[context_indices] = True

    return context_mask, target_indices


def train_one_epoch_ijepa(model, loader, optimizer, device):
    """
    One epoch of I-JEPA pretraining.
    No labels used — purely self-supervised.
    """
    model.train()
    total_loss = 0

    for csi in loader:           # labels ignored during pretraining
        csi = csi.to(device)
        B   = csi.shape[0]

        # Sample masks (same mask applied to all samples in batch)
        context_mask, target_indices = sample_masks(model.num_patches)
        context_mask   = context_mask.to(device)
        target_indices = target_indices.unsqueeze(0).expand(B, -1).to(device)

        optimizer.zero_grad()
        loss = model(csi, context_mask, target_indices)
        loss.backward()
        optimizer.step()

        # EMA update of target encoder after every step
        model.update_target_encoder()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data (no labels needed for pretraining)
    meta_hp             = load_metadata(config.DATA_ROOT)
    train_df, test_df   = get_splits(config.DATA_ROOT, meta_hp)

    # Filter missing files
    def file_exists(row):
        h5_path = config.DATA_ROOT + "/FallDetection/" + row["file_path"].lstrip("./")
        return os.path.exists(h5_path)

    train_df = train_df[train_df.apply(file_exists, axis=1)].reset_index(drop=True)

    # Use small subset for now
    train_df_small = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"Pretraining on {len(train_df_small)} samples")

    train_loader = DataLoader(
        CSIFallDataset(train_df_small, config.DATA_ROOT),
        batch_size=16, shuffle=True, num_workers=0
    )

    # Initialize I-JEPA model
    model = IJEPA(
        in_channels    = config.IN_CHANNELS,
        img_h          = config.IMG_H,
        img_w          = config.IMG_W,
        patch_h        = config.PATCH_H,
        patch_w        = config.PATCH_W,
        encoder_dim    = config.D_MODEL,
        encoder_ff_dim = config.D_FF,
        encoder_heads  = config.N_HEADS,
        encoder_depth  = config.N_LAYERS,
        predictor_dim  = config.D_MODEL // 2,   # predictor is smaller than encoder
        predictor_heads= 2,
        predictor_depth= 2,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.online_encoder.parameters()) +
        list(model.online_cls.parameters()) +
        list(model.online_pos.parameters()) +
        list(model.online_encoder_blocks.parameters()) +
        list(model.predictor.parameters()),
        lr=config.LR
    )

    # Pretraining loop
    print("\n--- I-JEPA Pretraining ---")
    for epoch in range(config.NUM_EPOCHS):
        loss = train_one_epoch_ijepa(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS} | Pretrain Loss: {loss:.4f}")

    # Save pretrained encoder
    save_path = "/Volumes/csi/default/raw_data/ijepa_pretrained.pth"
    torch.save({
        "online_encoder":        model.online_encoder.state_dict(),
        "online_cls":            model.online_cls.state_dict(),
        "online_pos":            model.online_pos.state_dict(),
        "online_encoder_blocks": model.online_encoder_blocks.state_dict(),
    }, save_path)
    print(f"\nPretrained encoder saved: {save_path}")


if __name__ == "__main__":
    main()