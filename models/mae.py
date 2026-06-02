# models/mae.py

import torch
import torch.nn as nn
from models.vit import PatchEmbedding, Encoder


class MAE(nn.Module):
    """
    Masked Autoencoder for CSI signals.

    Key differences from I-JEPA/Bootleg:
    - Encoder only processes visible (unmasked) patches
    - Lightweight decoder reconstructs original signal in pixel space
    - Loss computed ONLY on masked patches
    - Decoder is discarded after pretraining; only encoder is kept

    Reference: He et al. 2022 "Masked Autoencoders Are Scalable Vision Learners"
    """
    def __init__(self, in_channels, img_h, img_w,
                 patch_h, patch_w,
                 encoder_dim, encoder_ff_dim, encoder_heads, encoder_depth,
                 decoder_dim, decoder_heads, decoder_depth,
                 mask_ratio=0.75):
        super().__init__()

        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.patch_h     = patch_h
        self.patch_w     = patch_w
        self.mask_ratio  = mask_ratio
        self.encoder_dim = encoder_dim

        # ── Encoder ──────────────────────────────────────────
        self.patch_embedding  = PatchEmbedding(in_channels, patch_h, patch_w, encoder_dim)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim) * 0.02)
        self.encoder_blocks   = Encoder(encoder_dim, encoder_heads, encoder_ff_dim, encoder_depth)
        self.encoder_norm     = nn.LayerNorm(encoder_dim)

        # ── Decoder ──────────────────────────────────────────
        # Learnable mask token for masked positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)

        # Project encoder dim → decoder dim
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)

        # Decoder positional embedding (full sequence)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_dim) * 0.02)
        self.decoder_blocks    = Encoder(decoder_dim, decoder_heads, decoder_dim * 4, decoder_depth)
        self.decoder_norm      = nn.LayerNorm(decoder_dim)

        # Project back to pixel space
        self.decoder_proj = nn.Linear(decoder_dim, patch_h * patch_w)

    def patchify(self, x):
        """
        Convert CSI image to patch tokens.
        x: [B, 1, H, W]
        Returns: [B, N, patch_h * patch_w]
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        # [B, C, n_h, n_w, ph, pw]
        x = x.contiguous().view(B, -1, ph * pw)
        return x  # [B, N, patch_pixels]

    def unpatchify(self, patches, img_h, img_w):
        """
        Convert patch tokens back to CSI image.
        patches: [B, N, patch_pixels]
        Returns: [B, 1, H, W]
        """
        B = patches.shape[0]
        ph, pw = self.patch_h, self.patch_w
        n_h = img_h // ph
        n_w = img_w // pw
        x = patches.view(B, n_h, n_w, ph, pw)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, 1, img_h, img_w)
        return x

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W]  input CSI
        Returns:
            loss:       reconstruction MSE on masked patches only
            pred:       [B, N, patch_pixels] full reconstructed sequence
            mask:       [B, N] boolean, True = masked patch
        """
        B = x.shape[0]
        N = self.num_patches
        n_mask = int(N * self.mask_ratio)

        # ── 1. Patch embedding + positional encoding ──────────
        tokens = self.patch_embedding(x)            # [B, N, D]
        tokens = tokens + self.encoder_pos_embed    # add pos encoding

        # ── 2. Random masking ─────────────────────────────────
        noise       = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)          # ascending: first n_mask are masked
        ids_restore = ids_shuffle.argsort(dim=1)    # restore original order

        ids_masked  = ids_shuffle[:, :n_mask]       # [B, n_mask]
        ids_visible = ids_shuffle[:, n_mask:]       # [B, N-n_mask]

        # Keep only visible tokens for encoder
        visible_tokens = tokens.gather(
            1, ids_visible.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )  # [B, N_visible, D]

        # ── 3. Encoder: visible patches only ──────────────────
        encoded = self.encoder_blocks(visible_tokens)  # [B, N_visible, D]
        encoded = self.encoder_norm(encoded)

        # ── 4. Project to decoder dim ─────────────────────────
        encoded = self.encoder_to_decoder(encoded)     # [B, N_visible, decoder_dim]

        # ── 5. Restore full sequence with mask tokens ─────────
        mask_tokens = self.mask_token.expand(B, n_mask, -1)  # [B, n_mask, decoder_dim]
        full_seq    = torch.cat([mask_tokens, encoded], dim=1) # [B, N, decoder_dim]

        # Unshuffle to original patch order
        full_seq = full_seq.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, full_seq.shape[-1])
        )  # [B, N, decoder_dim]

        # ── 6. Add decoder positional encoding ────────────────
        full_seq = full_seq + self.decoder_pos_embed

        # ── 7. Decoder ────────────────────────────────────────
        decoded = self.decoder_blocks(full_seq)    # [B, N, decoder_dim]
        decoded = self.decoder_norm(decoded)
        pred    = self.decoder_proj(decoded)       # [B, N, patch_pixels]

        # ── 8. Loss: only on masked patches ───────────────────
        target = self.patchify(x)                  # [B, N, patch_pixels]

        # Build mask tensor: True = masked
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_masked, True)

        # MSE only on masked positions
        loss = ((pred - target) ** 2)              # [B, N, patch_pixels]
        loss = loss[mask].mean()                   # scalar

        return loss, pred, mask

    @torch.no_grad()
    def extract_layer_embeddings(self, x, layers=None):
        """
        Extract hidden embeddings at specified encoder layers.
        Used for KNN probing and layer analysis.

        Args:
            x:      [B, 1, H, W]
            layers: list of layer indices (1-indexed), default = all layers
        Returns:
            dict {layer_idx: [B, encoder_dim]}  (mean-pooled over patches)
        """
        if layers is None:
            layers = list(range(1, len(self.encoder_blocks.layers) + 1))

        # Use ALL patches (no masking for feature extraction)
        tokens = self.patch_embedding(x) + self.encoder_pos_embed

        layer_outputs = {}
        h = tokens
        for i, block in enumerate(self.encoder_blocks.layers):
            h = block(h)
            if (i + 1) in layers:
                # Mean pool over patch tokens → [B, D]
                layer_outputs[i + 1] = self.encoder_norm(h).mean(dim=1)

        return layer_outputs
