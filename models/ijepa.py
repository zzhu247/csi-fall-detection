# models/ijepa.py

import copy
import torch
import torch.nn as nn
from models.vit import PatchEmbedding, CLSToken, PositionalEncoding, Encoder


class Predictor(nn.Module):
    """
    Small transformer that predicts target embeddings from context embeddings.
    Intentionally smaller than the main encoder (fewer layers, smaller d_model).
    Output dimension is encoder_dim * num_target_layers to match concatenated
    multi-layer targets (Bootleg style).
    """
    def __init__(self, encoder_dim, predictor_dim, num_heads, depth,
                 num_patches, num_target_layers=4):
        super().__init__()

        self.num_target_layers = num_target_layers

        # Project from encoder space to predictor space
        self.input_proj       = nn.Linear(encoder_dim, predictor_dim)

        # Learnable positional embeddings for target positions
        self.target_pos_embed = nn.Embedding(num_patches, predictor_dim)

        # Small transformer encoder as the predictor backbone
        self.transformer      = Encoder(predictor_dim, num_heads,
                                        predictor_dim * 4, depth)

        # Project to encoder_dim * num_target_layers (concatenated multi-layer target)
        self.output_proj      = nn.Linear(predictor_dim,
                                          encoder_dim * num_target_layers)

    def forward(self, context_embeddings, target_indices):
        """
        Args:
            context_embeddings: [B, N_context, encoder_dim]
            target_indices:     [B, N_target]
        Returns:
            predicted embeddings: [B, N_target, encoder_dim * num_target_layers]
        """
        B, N_target = target_indices.shape

        # Project context into predictor space
        x = self.input_proj(context_embeddings)              # [B, N_context, predictor_dim]

        # Build target query tokens from positional embeddings
        target_queries = self.target_pos_embed(target_indices)  # [B, N_target, predictor_dim]

        # Concatenate context + target queries, run through transformer
        x = torch.cat([x, target_queries], dim=1)            # [B, N_context+N_target, predictor_dim]
        x = self.transformer(x)

        # Extract only the target positions (last N_target tokens)
        x = x[:, -N_target:, :]                              # [B, N_target, predictor_dim]

        return self.output_proj(x)                           # [B, N_target, encoder_dim * num_target_layers]


class IJEPA(nn.Module):
    """
    Bootleg-style I-JEPA: Image Joint-Embedding Predictive Architecture
    with multi-layer hidden self-distillation.

    Key difference from vanilla I-JEPA:
    - Instead of predicting only the final layer embedding,
      we predict embeddings from multiple hidden layers (e.g. 1, 4, 8, 12)
      concatenated together as the target.
    - This forces the model to capture features at varying levels of
      abstraction simultaneously (Bootleg, Lowe et al. 2026).

    Components:
    - Online encoder:  context patches only, gradient flows here
    - Target encoder:  full input, EMA update only, no gradient
    - Predictor:       predicts concatenated multi-layer target embeddings
    """
    def __init__(self, in_channels, img_h, img_w,
                 patch_h, patch_w, encoder_dim, encoder_ff_dim,
                 encoder_heads, encoder_depth,
                 predictor_dim, predictor_heads, predictor_depth,
                 target_layers=[1, 4, 8, 12],
                 ema_momentum=0.999):
        super().__init__()

        self.num_patches   = (img_h // patch_h) * (img_w // patch_w)
        self.ema_momentum  = ema_momentum
        self.target_layers = target_layers

        # Online encoder: receives context tokens, gradient flows here
        self.online_encoder        = nn.Sequential(
            PatchEmbedding(in_channels, patch_h, patch_w, encoder_dim),
        )
        self.online_cls            = CLSToken(encoder_dim)
        self.online_pos            = PositionalEncoding(self.num_patches, encoder_dim)
        self.online_encoder_blocks = Encoder(encoder_dim, encoder_heads,
                                             encoder_ff_dim, encoder_depth)

        # Target encoder: full input, NO gradient, EMA updated
        self.target_encoder        = nn.Sequential(
            PatchEmbedding(in_channels, patch_h, patch_w, encoder_dim),
        )
        self.target_cls            = CLSToken(encoder_dim)
        self.target_pos            = PositionalEncoding(self.num_patches, encoder_dim)
        self.target_encoder_blocks = Encoder(encoder_dim, encoder_heads,
                                             encoder_ff_dim, encoder_depth)

        self._init_target_encoder()

        # Predictor: outputs encoder_dim * num_target_layers
        self.predictor = Predictor(
            encoder_dim, predictor_dim,
            predictor_heads, predictor_depth,
            self.num_patches,
            num_target_layers=len(target_layers)
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

    def encode_target_multilayer(self, x):
        """
        Run full input through target encoder block by block.
        Collect patch token embeddings at each layer in self.target_layers.

        Returns:
            concatenated embeddings [B, N, encoder_dim * num_target_layers]
        """
        with torch.no_grad():
            x = self.target_encoder(x)   # [B, N, D]
            x = self.target_cls(x)       # [B, N+1, D]
            x = self.target_pos(x)       # [B, N+1, D]

            layer_outputs = {}
            for i, block in enumerate(self.target_encoder_blocks.layers):
                x = block(x)
                layer_idx = i + 1        # 1-indexed
                if layer_idx in self.target_layers:
                    # Drop CLS token, keep patch tokens only
                    layer_outputs[layer_idx] = x[:, 1:, :]  # [B, N, D]

        # Concatenate along embedding dim in layer order
        return torch.cat(
            [layer_outputs[l] for l in self.target_layers], dim=-1
        )  # [B, N, D * num_target_layers]

    def encode_context(self, x, context_mask):
        """
        Forward pass through online encoder using only context patches.
        context_mask: boolean tensor [N], True = keep this patch
        """
        x = self.online_encoder(x)       # [B, N, D]
        x = x[:, context_mask, :]        # keep context patches only
        x = self.online_encoder_blocks(x)
        return x                         # [B, N_context, D]

    def forward(self, x, context_mask, target_indices):
        """
        Bootleg forward pass.

        Args:
            x:              [B, C, H, W]  input CSI
            context_mask:   [N] boolean   which patches are context
            target_indices: [B, N_target] which patch indices to predict
        Returns:
            loss: scalar MSE over concatenated multi-layer targets
        """
        B = x.shape[0]

        # 1. Target encoder: collect multi-layer embeddings, stop gradient
        target_embeddings = self.encode_target_multilayer(x)
        # [B, N, D * num_target_layers]

        # Extract only target patch positions
        target_embeddings = target_embeddings[:, target_indices[0], :]
        # [B, N_target, D * num_target_layers]

        # 2. Online encoder: context patches only
        context_embeddings = self.encode_context(x, context_mask)
        # [B, N_context, D]

        # 3. Predictor: predict concatenated multi-layer target
        predicted = self.predictor(context_embeddings, target_indices)
        # [B, N_target, D * num_target_layers]

        # 4. MSE loss in latent space across all target layers
        loss = nn.functional.mse_loss(predicted, target_embeddings.detach())

        return loss