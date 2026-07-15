import torch
import torch.nn as nn

class MAEv2ForDownstream(nn.Module):
    """
    Downstream fine-tuning wrapper for MAEv2.
    
    Supports:
    1. Freezing/Unfreezing the pretrained backbone.
    2. A 2-layer MLP head for classification or regression tasks.
    """
    def __init__(self, pretrained_mae_v2, num_classes, hidden_dim=256, freeze_backbone=True):
        super().__init__()
        
        # 1. Extract core Encoder components from pretrained MAEv2
        self.patch_embedding   = pretrained_mae_v2.patch_embedding
        self.encoder_pos_embed = pretrained_mae_v2.encoder_pos_embed
        self.encoder_blocks    = pretrained_mae_v2.encoder_blocks
        self.encoder_norm      = pretrained_mae_v2.encoder_norm
        
        encoder_dim = pretrained_mae_v2.encoder_dim

        # 2. Add 2-layer MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        # 3. Initialize backbone status based on config
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool):
        """
        Dynamically toggle whether the backbone parameters require gradients.
        trainable=False -> Linear Probing
        trainable=True  -> Full Fine-tuning
        """
        backbone_modules = [self.patch_embedding, self.encoder_blocks, self.encoder_norm]
        
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = trainable
                
        # Ensure positional embedding status matches the backbone
        self.encoder_pos_embed.requires_grad = trainable
        
        status = "Unfrozen" if trainable else "Frozen"
        print(f"[*] Backbone status updated to: {status}")

    def forward(self, x):
        """
        Downstream fine-tuning processes the FULL sequence without any masking.
        x: [B, 1, H, W]
        """
        # 1. Forward through the unmasked encoder
        tokens = self.patch_embedding(x) + self.encoder_pos_embed
        h = self.encoder_blocks(tokens)
        h = self.encoder_norm(h)
        
        # 2. Global Average Pooling over all patch tokens
        # [B, N, encoder_dim] -> [B, encoder_dim]
        global_features = h.mean(dim=1)
        
        # 3. Map to final logits using the 2-layer MLP head
        logits = self.mlp_head(global_features)
        return logits