# train_bootleg_recon.py

import torch
from train_ijepa import sample_masks


def train_one_epoch_bootleg_recon(model, loader, optimizer, device):
    """
    One epoch of combined Bootleg + Reconstruction pretraining.
    Prints both loss components separately for monitoring.
    """
    model.train()
    total_loss       = 0
    total_loss_jepa  = 0
    total_loss_recon = 0

    for csi in loader:
        csi = csi.to(device)
        B   = csi.shape[0]

        context_mask, target_indices = sample_masks(model.ijepa.num_patches)
        context_mask   = context_mask.to(device)
        target_indices = target_indices.unsqueeze(0).expand(B, -1).to(device)

        optimizer.zero_grad()
        loss, loss_jepa, loss_recon = model(csi, context_mask, target_indices)
        loss.backward()
        optimizer.step()

        model.update_target_encoder()

        total_loss       += loss.item()
        total_loss_jepa  += loss_jepa
        total_loss_recon += loss_recon

    n = len(loader)
    return total_loss / n, total_loss_jepa / n, total_loss_recon / n