# models/decoder.py

import torch
import torch.nn as nn
from models.vit import Encoder


class CSIDecoder(nn.Module):
    """
    Masked CSI reconstruction decoder (MAE-style).

    Takes encoder output (full sequence including masked positions),
    reconstructs pixel values for masked patches only.

    Architecture:
        Linear projection → small Transformer → Linear projection to patch pixels
    """
    def __init__(self, encoder_dim, decoder_dim, num_heads, depth,
                 patch_h, patch_w, num_patches):
        super().__init__()

        self.patch_h    = patch_h
        self.patch_w    = patch_w
        patch_pixels    = patch_h * patch_w   # pixels per patch

        # Project from encoder dim to decoder dim
        self.input_proj = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask token: placeholder for masked patches
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))

        # Learnable positional embedding for decoder
        self.pos_embed  = nn.Parameter(torch.randn(1, num_patches, decoder_dim))

        # Small transformer decoder
        self.transformer = Encoder(decoder_dim, num_heads,
                                   decoder_dim * 4, depth)

        # Project back to pixel space
        self.output_proj = nn.Linear(decoder_dim, patch_pixels)

    def forward(self, encoder_output, context_mask):
        """
        Args:
            encoder_output: [B, N_context, encoder_dim]  (context patches only)
            context_mask:   [N] boolean, True = context patch

        Returns:
            reconstructed: [B, N_masked, patch_pixels]
        """
        B          = encoder_output.shape[0]
        N          = context_mask.shape[0]          # total patches
        N_context  = context_mask.sum().item()
        N_masked   = N - N_context

        # Project context tokens to decoder dim
        x_context = self.input_proj(encoder_output)  # [B, N_context, decoder_dim]

        # Build full sequence: context tokens + mask tokens
        # mask_token fills in for all masked positions
        mask_tokens = self.mask_token.expand(
            B, N_masked, -1
        )  # [B, N_masked, decoder_dim]

        # Reconstruct full sequence in original patch order
        full_seq = torch.zeros(B, N, self.input_proj.out_features,
                               device=encoder_output.device)
        full_seq[:, context_mask, :]  = x_context
        full_seq[:, ~context_mask, :] = mask_tokens

        # Add positional embedding
        full_seq = full_seq + self.pos_embed   # [B, N, decoder_dim]

        # Run through decoder transformer
        full_seq = self.transformer(full_seq)   # [B, N, decoder_dim]

        # Project to pixel space, return only masked positions
        reconstructed = self.output_proj(full_seq)        # [B, N, patch_pixels]
        return reconstructed[:, ~context_mask, :]         # [B, N_masked, patch_pixels]