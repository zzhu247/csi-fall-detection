# models/bootleg_with_recon.py

import torch
import torch.nn as nn
from models.ijepa import IJEPA
from models.decoder import CSIDecoder


class BootlegWithReconstruction(nn.Module):
    """
    Combined Bootleg + Masked Reconstruction model.

    Two training objectives:
    1. Bootleg loss:   predict multi-layer target embeddings in latent space
    2. Recon loss:     reconstruct original CSI amplitude for masked patches

    Total loss = alpha * L_bootleg + beta * L_recon
    """
    def __init__(self, in_channels, img_h, img_w,
                 patch_h, patch_w, encoder_dim, encoder_ff_dim,
                 encoder_heads, encoder_depth,
                 predictor_dim, predictor_heads, predictor_depth,
                 decoder_dim, decoder_heads, decoder_depth,
                 target_layers=[1, 2, 3, 4],
                 alpha=1.0, beta=1.0,
                 ema_momentum=0.999):
        super().__init__()

        self.alpha = alpha   # weight for Bootleg loss
        self.beta  = beta    # weight for reconstruction loss

        # Bootleg I-JEPA backbone
        self.ijepa = IJEPA(
            in_channels    = in_channels,
            img_h          = img_h,
            img_w          = img_w,
            patch_h        = patch_h,
            patch_w        = patch_w,
            encoder_dim    = encoder_dim,
            encoder_ff_dim = encoder_ff_dim,
            encoder_heads  = encoder_heads,
            encoder_depth  = encoder_depth,
            predictor_dim  = predictor_dim,
            predictor_heads= predictor_heads,
            predictor_depth= predictor_depth,
            target_layers  = target_layers,
            ema_momentum   = ema_momentum,
        )

        # Reconstruction decoder
        num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.decoder = CSIDecoder(
            encoder_dim = encoder_dim,
            decoder_dim = decoder_dim,
            num_heads   = decoder_heads,
            depth       = decoder_depth,
            patch_h     = patch_h,
            patch_w     = patch_w,
            num_patches = num_patches,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w

    def extract_target_patches(self, x, context_mask):
        """
        Extract original pixel values for masked patches.
        Used as reconstruction target.

        Args:
            x:            [B, 1, H, W]  original CSI
            context_mask: [N] boolean

        Returns:
            [B, N_masked, patch_h * patch_w]
        """
        B, C, H, W = x.shape
        ph, pw     = self.patch_h, self.patch_w
        n_h        = H // ph
        n_w        = W // pw

        # Reshape into patches: [B, N, patch_pixels]
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        # [B, C, n_h, n_w, ph, pw]
        patches = patches.contiguous().view(B, n_h * n_w, ph * pw)
        # [B, N, patch_pixels]

        # Return only masked patches
        return patches[:, ~context_mask, :]   # [B, N_masked, patch_pixels]

    def forward(self, x, context_mask, target_indices):
        """
        Args:
            x:              [B, 1, H, W]
            context_mask:   [N] boolean
            target_indices: [B, N_target]

        Returns:
            loss:       total loss (scalar)
            loss_jepa:  Bootleg component
            loss_recon: reconstruction component
        """
        # ── 1. Bootleg loss ──────────────────────────────────────
        loss_jepa = self.ijepa(x, context_mask, target_indices)

        # ── 2. Reconstruction loss ───────────────────────────────
        # Get context embeddings from online encoder
        context_embeddings = self.ijepa.encode_context(x, context_mask)
        # [B, N_context, encoder_dim]

        # Decode: reconstruct masked patch pixels
        reconstructed = self.decoder(context_embeddings, context_mask)
        # [B, N_masked, patch_pixels]

        # Get ground truth pixel values for masked patches
        target_pixels = self.extract_target_patches(x, context_mask)
        # [B, N_masked, patch_pixels]

        loss_recon = nn.functional.mse_loss(reconstructed, target_pixels)

        # ── 3. Combined loss ─────────────────────────────────────
        loss = self.alpha * loss_jepa + self.beta * loss_recon

        return loss, loss_jepa.item(), loss_recon.item()

    def update_target_encoder(self):
        """Delegate EMA update to IJEPA."""
        self.ijepa.update_target_encoder()