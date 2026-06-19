# models/baselines.py
#
# All CSI-Bench paper baseline models (Appendix B):
#   - MLP
#   - LSTM (Bidirectional)
#   - Transformer (global average pooling, no CLS)
#   - ViTPaper (6 layers, dropout=0.1, matches paper exactly)
#   - PatchTST
#   - TimeSformer1D

import math
import torch
import torch.nn as nn


# ── MLP ───────────────────────────────────────────────────────────────────────
# Input → Linear(512) → ReLU → Dropout(0.5) → Linear(128) → ReLU
#       → Dropout(0.3) → Linear(num_classes)
# Input is flattened CSI, capped at 10000 dims.

class MLP(nn.Module):
    def __init__(self, img_h=232, img_w=500, num_classes=2, max_input=10000):
        super().__init__()
        input_dim = min(img_h * img_w, max_input)
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, H, W]
        B = x.shape[0]
        x = x.view(B, -1)                        # [B, H*W]
        x = x[:, :self.input_dim]                # cap at max_input
        return self.net(x)


# ── LSTM ──────────────────────────────────────────────────────────────────────
# Bidirectional LSTM, 2 layers, 256 hidden units
# Input → BiLSTM(256, 2 layers, dropout=0.3) → Linear(256) → ReLU
#       → Dropout(0.3) → Linear(num_classes)
# Treats subcarrier dimension as feature dim, time as sequence.

class LSTMModel(nn.Module):
    def __init__(self, img_h=232, img_w=500, num_classes=2):
        super().__init__()
        # input: [B, T, H] where T=time steps, H=subcarriers
        self.lstm = nn.LSTM(
            input_size  = img_h,
            hidden_size = 256,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 256),   # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, H, W]  H=subcarriers, W=time
        B, C, H, W = x.shape
        x = x.squeeze(1).permute(0, 2, 1)   # [B, T, H]
        out, _ = self.lstm(x)                # [B, T, 512]
        out = out[:, -1, :]                  # last time step [B, 512]
        return self.head(out)


# ── Transformer ───────────────────────────────────────────────────────────────
# 4 layers, d_model=256, 8 heads, dropout=0.1
# Linear projection → positional encoding → Transformer encoder
# → global average pooling → classifier

class TransformerModel(nn.Module):
    def __init__(self, img_h=232, img_w=500, num_classes=2,
                 d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        # Project each time step (subcarrier vector) to d_model
        self.input_proj = nn.Linear(img_h, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(1, img_w, d_model) * 0.02)
        self.dropout    = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model     = d_model,
            nhead       = nhead,
            dim_feedforward = d_model * 4,
            dropout     = dropout,
            batch_first = True,
            norm_first  = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        B, C, H, W = x.shape
        x = x.squeeze(1).permute(0, 2, 1)   # [B, T, H]
        x = self.input_proj(x)               # [B, T, d_model]
        x = self.dropout(x + self.pos_embed[:, :x.shape[1], :])
        x = self.encoder(x)                  # [B, T, d_model]
        x = self.norm(x)
        x = x.mean(dim=1)                    # global average pooling [B, d_model]
        return self.head(x)


# ── ViT (Paper-exact) ─────────────────────────────────────────────────────────
# 6 layers, d_model=128, 4 heads, dropout=0.1, CLS token

class ViTPaper(nn.Module):
    def __init__(self, img_h=232, img_w=500, patch_h=8, patch_w=25,
                 num_classes=2, d_model=128, nhead=4, num_layers=6, dropout=0.1):
        super().__init__()
        n_h = img_h // patch_h
        n_w = img_w // patch_w
        num_patches = n_h * n_w

        # Conv2d patch embedding
        self.patch_embed = nn.Conv2d(1, d_model,
                                     kernel_size=(patch_h, patch_w),
                                     stride=(patch_h, patch_w))
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)
        self.dropout    = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)              # [B, d_model, n_h, n_w]
        x = x.flatten(2).transpose(1, 2)     # [B, N, d_model]

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)     # [B, N+1, d_model]
        x   = self.dropout(x + self.pos_embed)

        x   = self.encoder(x)
        x   = self.norm(x)
        return self.head(x[:, 0])            # CLS token


# ── PatchTST ──────────────────────────────────────────────────────────────────
# Temporal patch embeddings: patch_length=16, stride=8
# 4 layers, d_model=128, 4 heads, dropout=0.1
# Treats each subcarrier as an independent channel,
# patches along the time dimension.

class PatchTST(nn.Module):
    def __init__(self, img_h=232, img_w=500, num_classes=2,
                 patch_len=16, stride=8,
                 d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.img_h     = img_h
        self.patch_len = patch_len
        self.stride    = stride

        # Number of patches along time axis
        n_patches = (img_w - patch_len) // stride + 1
        self.n_patches = n_patches

        # Each patch: patch_len values per subcarrier channel
        # Project each (subcarrier, patch) to d_model
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed   = nn.Parameter(
            torch.randn(1, n_patches, d_model) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout   = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        # Mean pool over subcarriers then classify
        self.head    = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        B, C, H, W = x.shape
        x = x.squeeze(1)   # [B, H, W]

        # Extract patches along time: [B, H, n_patches, patch_len]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: [B, H, n_patches, patch_len]

        B, H, NP, PL = patches.shape
        patches = patches.reshape(B * H, NP, PL)          # [B*H, NP, PL]
        patches = self.patch_embed(patches)                # [B*H, NP, d_model]

        cls = self.cls_token.expand(B * H, -1, -1)
        patches = torch.cat([cls, patches], dim=1)        # [B*H, NP+1, d_model]
        patches = self.dropout(patches + torch.cat([
            torch.zeros(B * H, 1, patches.shape[-1], device=x.device),
            self.pos_embed.expand(B * H, -1, -1)
        ], dim=1))

        out = self.encoder(patches)                        # [B*H, NP+1, d_model]
        out = self.norm(out[:, 0])                         # CLS token [B*H, d_model]
        out = out.view(B, H, -1).mean(dim=1)              # mean over subcarriers [B, d_model]
        return self.head(out)


# ── TimeSformer-1D ────────────────────────────────────────────────────────────
# patch_size=4 along time, 4 layers, d_model=128, 8 heads, dropout
# Separate temporal and feature attention within each block.

class TimeSformer1D(nn.Module):
    def __init__(self, img_h=232, img_w=500, num_classes=2,
                 patch_size=4, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        n_patches = img_w // patch_size
        self.patch_embed = nn.Conv1d(img_h, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        self.dropout    = nn.Dropout(dropout)

        # Each block: temporal attention + feature attention
        self.temporal_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.feature_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        B, C, H, W = x.shape
        x = x.squeeze(1)                         # [B, H, W]
        x = self.patch_embed(x)                  # [B, d_model, n_patches]
        x = x.transpose(1, 2)                    # [B, n_patches, d_model]

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)         # [B, n_patches+1, d_model]
        x   = self.dropout(x + self.pos_embed)

        for t_attn, f_attn in zip(self.temporal_attn, self.feature_attn):
            x = t_attn(x)   # temporal attention
            x = f_attn(x)   # feature attention

        x = self.norm(x)
        return self.head(x[:, 0])               # CLS token


# ── Factory ───────────────────────────────────────────────────────────────────

def build_baseline(model_name, num_classes, img_h=232, img_w=500):
    """
    model_name: one of
        mlp, lstm, transformer, vit, vit_paper,
        patchtst, timesformer1d, resnet18
    """
    kwargs = dict(img_h=img_h, img_w=img_w, num_classes=num_classes)

    if model_name == "mlp":
        return MLP(**kwargs)
    elif model_name == "lstm":
        return LSTMModel(**kwargs)
    elif model_name == "transformer":
        return TransformerModel(**kwargs)
    elif model_name == "vit_paper":
        return ViTPaper(**kwargs)
    elif model_name == "patchtst":
        return PatchTST(**kwargs)
    elif model_name == "timesformer1d":
        return TimeSformer1D(**kwargs)
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        return ResNet18(in_channels=1, num_classes=num_classes)
    elif model_name == "vit":
        from models.vit import ViT
        return ViT(in_channels=1, img_h=img_h, img_w=img_w,
                   patch_h=8, patch_w=25, d_model=128, d_ff=512,
                   h=4, N=12, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
