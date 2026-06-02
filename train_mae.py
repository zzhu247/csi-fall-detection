# train_mae.py

import torch
from models.mae import MAE


def train_one_epoch_mae(model, loader, optimizer, device):
    """
    One epoch of MAE pretraining.
    No labels used — purely self-supervised reconstruction.
    """
    model.train()
    total_loss = 0

    for csi in loader:
        csi = csi.to(device)

        optimizer.zero_grad()
        loss, pred, mask = model(csi)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, n_samples=4):
    """
    Visualize MAE reconstruction quality.
    Returns original and reconstructed CSI for a few samples.
    """
    model.eval()
    for csi, _ in loader:
        csi = csi.to(device)
        loss, pred, mask = model(csi)

        # Unpatchify predictions
        B = csi.shape[0]
        img_h, img_w = csi.shape[2], csi.shape[3]
        recon = model.unpatchify(pred, img_h, img_w)

        # Only show masked regions on top of original
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, model.patch_h * model.patch_w)
        recon_patches  = pred.clone()
        orig_patches   = model.patchify(csi)
        recon_patches[~mask_expanded] = orig_patches[~mask_expanded]
        recon_vis = model.unpatchify(recon_patches, img_h, img_w)

        return csi[:n_samples].cpu(), recon_vis[:n_samples].cpu(), loss.item()
