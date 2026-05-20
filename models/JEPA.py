# models/ijepa.py

import copy
import torch
import torch.nn as nn
from models.vit import PatchEmbedding, CLSToken, PositionalEncoding, Encoder


class Predictor(nn.Module):
    """
    Small transformer that predicts target embeddings from context embeddings.
    Intentionally smaller than the main encoder (fewer layers, smaller d_model).
    """
    def __init__(self, encoder_dim, predictor_dim, num_heads, depth, num_patches):
        super().__init__()

        # Project from encoder space to predictor space
        self.input_proj  = nn.Linear(encoder_dim, predictor_dim)

        # Learnable positional embeddings for target positions
        self.target_pos_embed = nn.Embedding(num_patches, predictor_dim)

        # Small transformer encoder as the predictor backbone
        self.transformer = Encoder(predictor_dim, num_heads, predictor_dim * 4, depth)

        # Project back to encoder space for loss computation
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, context_embeddings, target_indices):
        """
        Args:
            context_embeddings: [B, N_context, encoder_dim]
            target_indices:     [B, N_target]  patch indices to predict
        Returns:
            predicted embeddings: [B, N_target, encoder_dim]
        """
        B, N_target = target_indices.shape

        # Project context into predictor space
        x = self.input_proj(context_embeddings)   # [B, N_context, predictor_dim]

        # Build target query tokens from positional embeddings
        target_queries = self.target_pos_embed(target_indices)  # [B, N_target, predictor_dim]

        # Concatenate context + target queries, run through transformer
        x = torch.cat([x, target_queries], dim=1)  # [B, N_context + N_target, predictor_dim]
        x = self.transformer(x)

        # Extract only the target positions (last N_target tokens)
        x = x[:, -N_target:, :]                   # [B, N_target, predictor_dim]

        return self.output_proj(x)                 # [B, N_target, encoder_dim]


class IJEPA(nn.Module):
    """
    I-JEPA: Image Joint-Embedding Predictive Architecture.

    Key components:
    - Online encoder:  processes context patches (with gradient)
    - Target encoder:  processes full input, updated via EMA (no gradient)
    - Predictor:       predicts target embeddings from context (small transformer)

    Loss: MSE between predicted and target embeddings in latent space.
    No pixel/signal reconstruction — everything happens in embedding space.
    """
    def __init__(self, in_channels, img_h, img_w,
                 patch_h, patch_w, encoder_dim, encoder_ff_dim,
                 encoder_heads, encoder_depth,
                 predictor_dim, predictor_heads, predictor_depth,
                 ema_momentum=0.996):
        super().__init__()

        self.num_patches  = (img_h // patch_h) * (img_w // patch_w)
        self.ema_momentum = ema_momentum

        # Online encoder: receives context tokens, gradient flows here
        self.online_encoder = nn.Sequential(
            PatchEmbedding(in_channels, patch_h, patch_w, encoder_dim),
        )
        self.online_cls      = CLSToken(encoder_dim)
        self.online_pos      = PositionalEncoding(self.num_patches, encoder_dim)
        self.online_encoder_blocks = Encoder(encoder_dim, encoder_heads, encoder_ff_dim, encoder_depth)

        # Target encoder: receives full input, NO gradient, updated via EMA
        self.target_encoder = nn.Sequential(
            PatchEmbedding(in_channels, patch_h, patch_w, encoder_dim),
        )
        self.target_cls      = CLSToken(encoder_dim)
        self.target_pos      = PositionalEncoding(self.num_patches, encoder_dim)
        self.target_encoder_blocks = Encoder(encoder_dim, encoder_heads, encoder_ff_dim, encoder_depth)

        # Copy online weights to target encoder, then freeze target
        self._init_target_encoder()

        # Predictor: predicts target embeddings from context
        self.predictor = Predictor(
            encoder_dim, predictor_dim,
            predictor_heads, predictor_depth,
            self.num_patches
        )

    def _init_target_encoder(self):
        """Copy online encoder weights to target encoder and freeze it."""
        for online, target in self._encoder_param_pairs():
            target.data.copy_(online.data)
            target.requires_grad = False

    def _encoder_param_pairs(self):
        """Yield (online_param, target_param) pairs for EMA update."""
        online_params = (
            list(self.online_encoder.parameters()) +
            list(self.online_cls.parameters()) +
            list(self.online_pos.parameters()) +
            list(self.online_encoder_blocks.parameters())
        )
        target_params = (
            list(self.target_encoder.parameters()) +
            list(self.target_cls.parameters()) +
            list(self.target_pos.parameters()) +
            list(self.target_encoder_blocks.parameters())
        )
        return zip(online_params, target_params)

    @torch.no_grad()
    def update_target_encoder(self):
        """
        EMA update: θ_target ← τ * θ_target + (1 - τ) * θ_online
        Called after every training step.
        """
        for online, target in self._encoder_param_pairs():
            target.data = (self.ema_momentum * target.data +
                           (1 - self.ema_momentum) * online.data)

    def encode_target(self, x):
        """Full forward pass through target encoder (no gradient)."""
        x = self.target_encoder(x)   # [B, N, encoder_dim]
        x = self.target_cls(x)       # [B, N+1, encoder_dim]
        x = self.target_pos(x)       # [B, N+1, encoder_dim]
        x = self.target_encoder_blocks(x)
        return x[:, 1:, :]           # drop CLS, return patch tokens [B, N, encoder_dim]

    def encode_context(self, x, context_mask):
        """
        Forward pass through online encoder using only context patches.
        context_mask: boolean tensor [N], True = keep this patch
        """
        x = self.online_encoder(x)              # [B, N, encoder_dim]
        x = x[:, context_mask, :]               # keep context patches only
        x = self.online_encoder_blocks(x)       # [B, N_context, encoder_dim]
        return x

    def forward(self, x, context_mask, target_indices):
        """
        Args:
            x:              [B, C, H, W]  input CSI
            context_mask:   [N] boolean   which patches are context
            target_indices: [B, N_target] which patch indices to predict
        Returns:
            loss: scalar MSE in embedding space
        """
        # 1. Target encoder: encode full input, stop gradient
        with torch.no_grad():
            target_embeddings = self.encode_target(x)           # [B, N, D]
            target_embeddings = target_embeddings[:, target_indices[0], :]  # [B, N_target, D]

        # 2. Online encoder: encode context patches only
        context_embeddings = self.encode_context(x, context_mask)  # [B, N_context, D]

        # 3. Predictor: predict target embeddings from context
        predicted = self.predictor(context_embeddings, target_indices)  # [B, N_target, D]

        # 4. Loss: MSE in latent space (NOT pixel space)
        loss = nn.functional.mse_loss(predicted, target_embeddings.detach())

        return loss