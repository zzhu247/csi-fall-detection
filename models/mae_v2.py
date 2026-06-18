# models/mae_v2.py
#
# Upgrades over mae.py:
#   1. Block masking  — mask contiguous time/subcarrier blocks instead of random patches
#   2. Physics losses — spectral diff loss + temporal diff loss on top of MSE
#
# Drop-in replacement: same forward() signature as MAE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit import PatchEmbedding, Encoder


# ── Masking strategies ────────────────────────────────────────────────────────

def random_mask(B, N, mask_ratio, device):
    """Original MAE random masking. Returns ids_masked, ids_visible, ids_restore."""
    n_mask = int(N * mask_ratio)
    noise = torch.rand(B, N, device=device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_masked  = ids_shuffle[:, :n_mask]
    ids_visible = ids_shuffle[:, n_mask:]
    return ids_masked, ids_visible, ids_restore


def block_mask(B, N, mask_ratio, n_h, n_w, device, block_mode="mixed"):
    """
    Block masking on the 2D patch grid.

    block_mode:
        "time"     — mask contiguous columns (time blocks)
        "freq"     — mask contiguous rows    (subcarrier blocks)
        "mixed"    — randomly choose time or freq per sample in the batch
        "2d"       — mask a single 2D rectangular block per sample

    Returns ids_masked [B, n_mask], ids_visible [B, N-n_mask], ids_restore [B, N]
    """
    n_mask = int(N * mask_ratio)
    all_ids_masked  = []
    all_ids_visible = []

    for b in range(B):
        mode = block_mode
        if block_mode == "mixed":
            mode = "time" if torch.rand(1).item() > 0.5 else "freq"

        # Build a 2D boolean mask on the patch grid [n_h, n_w]
        patch_mask = torch.zeros(n_h, n_w, dtype=torch.bool, device=device)

        if mode == "time":
            # Mask contiguous time columns until we hit n_mask patches
            n_cols_to_mask = max(1, n_mask // n_h)
            col_start = torch.randint(0, max(1, n_w - n_cols_to_mask), (1,)).item()
            col_end   = min(n_w, col_start + n_cols_to_mask)
            patch_mask[:, col_start:col_end] = True

        elif mode == "freq":
            # Mask contiguous subcarrier rows
            n_rows_to_mask = max(1, n_mask // n_w)
            row_start = torch.randint(0, max(1, n_h - n_rows_to_mask), (1,)).item()
            row_end   = min(n_h, row_start + n_rows_to_mask)
            patch_mask[row_start:row_end, :] = True

        elif mode == "2d":
            # Mask a 2D rectangular block
            block_h = max(1, int((n_mask / N) ** 0.5 * n_h))
            block_w = max(1, int((n_mask / N) ** 0.5 * n_w))
            r_start = torch.randint(0, max(1, n_h - block_h), (1,)).item()
            c_start = torch.randint(0, max(1, n_w - block_w), (1,)).item()
            patch_mask[r_start:r_start + block_h, c_start:c_start + block_w] = True

        # Flatten to patch indices
        flat_mask = patch_mask.flatten()  # [N]

        # If we over/under-masked due to integer rounding, adjust randomly
        masked_indices   = flat_mask.nonzero(as_tuple=True)[0]
        unmasked_indices = (~flat_mask).nonzero(as_tuple=True)[0]

        # Shuffle both and trim/extend to exactly n_mask
        perm_m = torch.randperm(len(masked_indices),   device=device)
        perm_v = torch.randperm(len(unmasked_indices), device=device)
        masked_indices   = masked_indices[perm_m]
        unmasked_indices = unmasked_indices[perm_v]

        if len(masked_indices) < n_mask:
            # Need more masked: borrow from visible
            deficit = n_mask - len(masked_indices)
            extra   = unmasked_indices[:deficit]
            masked_indices   = torch.cat([masked_indices, extra])
            unmasked_indices = unmasked_indices[deficit:]
        elif len(masked_indices) > n_mask:
            # Too many masked: move excess to visible
            excess           = masked_indices[n_mask:]
            masked_indices   = masked_indices[:n_mask]
            unmasked_indices = torch.cat([unmasked_indices, excess])

        all_ids_masked.append(masked_indices)
        all_ids_visible.append(unmasked_indices)

    ids_masked  = torch.stack(all_ids_masked,  dim=0)  # [B, n_mask]
    ids_visible = torch.stack(all_ids_visible, dim=0)  # [B, N - n_mask]

    # Build ids_restore
    ids_shuffle = torch.cat([ids_masked, ids_visible], dim=1)  # [B, N]
    ids_restore = ids_shuffle.argsort(dim=1)

    return ids_masked, ids_visible, ids_restore


# ── Physics-aware loss ────────────────────────────────────────────────────────

def physics_loss(pred_patches, target_patches, mask,
                 n_h, n_w, patch_h, patch_w,
                 lambda_spec=0.1, lambda_temp=0.1):
    """
    Combines MSE + spectral diff loss + temporal diff loss.

    pred_patches:   [B, N, patch_h * patch_w]
    target_patches: [B, N, patch_h * patch_w]
    mask:           [B, N] bool, True = masked patch

    Spectral diff loss:  penalises difference between adjacent subcarrier patches
                         along the frequency axis (row direction in patch grid)
    Temporal diff loss:  penalises difference between adjacent time patches
                         along the time axis (column direction in patch grid)

    lambda_spec / lambda_temp control contribution of each physics term.
    """
    B, N, P = pred_patches.shape

    # ── MSE on masked patches only ─────────────────────────────
    loss_mse = ((pred_patches - target_patches) ** 2)[mask].mean()

    # ── Reshape to 2D patch grid for diff losses ───────────────
    # pred_img: [B, n_h, n_w, patch_h, patch_w]
    pred_img   = pred_patches.view(B, n_h, n_w, patch_h, patch_w)
    target_img = target_patches.view(B, n_h, n_w, patch_h, patch_w)

    # Full reconstructed image in pixel space: [B, 1, H, W]
    H = n_h * patch_h
    W = n_w * patch_w
    pred_full   = pred_img.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, H, W)
    target_full = target_img.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, H, W)

    # ── Spectral diff loss (along subcarrier / height axis) ────
    # First-order difference along H (subcarrier) dimension
    pred_spec_diff   = pred_full[:, :, 1:, :] - pred_full[:, :, :-1, :]   # [B,1,H-1,W]
    target_spec_diff = target_full[:, :, 1:, :] - target_full[:, :, :-1, :]
    loss_spec = F.mse_loss(pred_spec_diff, target_spec_diff)

    # ── Temporal diff loss (along time / width axis) ───────────
    pred_temp_diff   = pred_full[:, :, :, 1:] - pred_full[:, :, :, :-1]   # [B,1,H,W-1]
    target_temp_diff = target_full[:, :, :, 1:] - target_full[:, :, :, :-1]
    loss_temp = F.mse_loss(pred_temp_diff, target_temp_diff)

    total = loss_mse + lambda_spec * loss_spec + lambda_temp * loss_temp
    return total, loss_mse, loss_spec, loss_temp


# ── MAE v2 ───────────────────────────────────────────────────────────────────

class MAEv2(nn.Module):
    """
    MAE with configurable masking strategy and physics-aware loss.

    mask_strategy: "random" | "time" | "freq" | "mixed" | "2d"
    use_physics_loss: if True, add spectral + temporal diff losses
    lambda_spec / lambda_temp: weights for physics loss terms
    """
    def __init__(self, in_channels, img_h, img_w,
                 patch_h, patch_w,
                 encoder_dim, encoder_ff_dim, encoder_heads, encoder_depth,
                 decoder_dim, decoder_heads, decoder_depth,
                 mask_ratio=0.75,
                 mask_strategy="random",
                 use_physics_loss=False,
                 lambda_spec=0.1,
                 lambda_temp=0.1):
        super().__init__()

        self.img_h      = img_h
        self.img_w      = img_w
        self.patch_h    = patch_h
        self.patch_w    = patch_w
        self.n_h        = img_h // patch_h
        self.n_w        = img_w // patch_w
        self.num_patches = self.n_h * self.n_w
        self.mask_ratio  = mask_ratio
        self.encoder_dim = encoder_dim

        self.mask_strategy     = mask_strategy
        self.use_physics_loss  = use_physics_loss
        self.lambda_spec       = lambda_spec
        self.lambda_temp       = lambda_temp

        # ── Encoder ──────────────────────────────────────────
        self.patch_embedding   = PatchEmbedding(in_channels, patch_h, patch_w, encoder_dim)
        self.encoder_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, encoder_dim) * 0.02
        )
        self.encoder_blocks = Encoder(encoder_dim, encoder_heads, encoder_ff_dim, encoder_depth)
        self.encoder_norm   = nn.LayerNorm(encoder_dim)

        # ── Decoder ──────────────────────────────────────────
        self.mask_token         = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_pos_embed  = nn.Parameter(
            torch.randn(1, self.num_patches, decoder_dim) * 0.02
        )
        self.decoder_blocks = Encoder(decoder_dim, decoder_heads, decoder_dim * 4, decoder_depth)
        self.decoder_norm   = nn.LayerNorm(decoder_dim)
        self.decoder_proj   = nn.Linear(decoder_dim, patch_h * patch_w)

    # ── Patch utilities ───────────────────────────────────────

    def patchify(self, x):
        """x: [B,1,H,W] → [B,N,patch_h*patch_w]"""
        B, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        return x.contiguous().view(B, -1, ph * pw)

    def unpatchify(self, patches, img_h=None, img_w=None):
        """patches: [B,N,patch_pixels] → [B,1,H,W]"""
        img_h = img_h or self.img_h
        img_w = img_w or self.img_w
        B = patches.shape[0]
        ph, pw = self.patch_h, self.patch_w
        n_h, n_w = img_h // ph, img_w // pw
        x = patches.view(B, n_h, n_w, ph, pw)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.view(B, 1, img_h, img_w)

    # ── Forward ───────────────────────────────────────────────

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W]
        Returns:
            loss:      scalar
            pred:      [B, N, patch_pixels]  full reconstructed sequence
            mask:      [B, N] bool, True = masked
            loss_dict: dict with individual loss components
        """
        B = x.shape[0]
        N = self.num_patches

        # ── 1. Patch embedding + pos encoding ─────────────────
        tokens = self.patch_embedding(x) + self.encoder_pos_embed  # [B, N, D]

        # ── 2. Masking ────────────────────────────────────────
        if self.mask_strategy == "random":
            ids_masked, ids_visible, ids_restore = random_mask(
                B, N, self.mask_ratio, x.device
            )
        else:
            ids_masked, ids_visible, ids_restore = block_mask(
                B, N, self.mask_ratio, self.n_h, self.n_w,
                x.device, block_mode=self.mask_strategy
            )

        # Keep only visible tokens for encoder
        visible_tokens = tokens.gather(
            1, ids_visible.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )  # [B, N_visible, D]

        # ── 3. Encode visible patches ─────────────────────────
        encoded = self.encoder_norm(self.encoder_blocks(visible_tokens))
        encoded = self.encoder_to_decoder(encoded)  # [B, N_visible, decoder_dim]

        # ── 4. Restore full sequence with mask tokens ─────────
        mask_tokens = self.mask_token.expand(B, N - encoded.shape[1], -1)
        full_seq    = torch.cat([mask_tokens, encoded], dim=1)
        full_seq    = full_seq.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, full_seq.shape[-1])
        ) + self.decoder_pos_embed  # [B, N, decoder_dim]

        # ── 5. Decode ─────────────────────────────────────────
        decoded = self.decoder_norm(self.decoder_blocks(full_seq))
        pred    = self.decoder_proj(decoded)  # [B, N, patch_pixels]

        # ── 6. Build mask boolean tensor ──────────────────────
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_masked, True)  # True = masked

        # ── 7. Loss ───────────────────────────────────────────
        target = self.patchify(x)  # [B, N, patch_pixels]

        if self.use_physics_loss:
            loss, loss_mse, loss_spec, loss_temp = physics_loss(
                pred, target, mask,
                self.n_h, self.n_w, self.patch_h, self.patch_w,
                self.lambda_spec, self.lambda_temp
            )
            loss_dict = {
                "loss":      loss.item(),
                "loss_mse":  loss_mse.item(),
                "loss_spec": loss_spec.item(),
                "loss_temp": loss_temp.item(),
            }
        else:
            loss = ((pred - target) ** 2)[mask].mean()
            loss_dict = {"loss": loss.item()}

        return loss, pred, mask, loss_dict

    @torch.no_grad()
    def extract_layer_embeddings(self, x, layers=None):
        """
        Extract encoder embeddings for KNN/linear probe evaluation.
        No masking — uses all patches.
        """
        if layers is None:
            layers = list(range(1, len(self.encoder_blocks.layers) + 1))

        tokens = self.patch_embedding(x) + self.encoder_pos_embed
        layer_outputs = {}
        h = tokens
        for i, block in enumerate(self.encoder_blocks.layers):
            h = block(h)
            if (i + 1) in layers:
                layer_outputs[i + 1] = self.encoder_norm(h).mean(dim=1)
        return layer_outputs
