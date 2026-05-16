# models/vit.py

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_h, patch_w, d_model):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))

    def forward(self, x):
        x = self.proj(x)       # [B, d_model, H', W']
        x = x.flatten(2)       # [B, d_model, N]
        x = x.transpose(1, 2)  # [B, N, d_model]
        return x


class CLSToken(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        cls = self.cls_token.expand(x.shape[0], 1, self.cls_token.shape[-1])
        return torch.cat((cls, x), dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

    def forward(self, x):
        return x + self.pos_embedding


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_out):
        return self.norm(x + sublayer_out)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.h   = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B = q.shape[0]
        Q = self.w_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        out = (torch.softmax(scores, dim=-1) @ V).transpose(1, 2).contiguous()
        return self.W_o(out.view(B, -1, self.h * self.d_k))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, h)
        self.ff        = FeedForward(d_model, d_ff)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, mask=None):
        x = self.add_norm1(x, self.attention(x, x, x, mask))
        x = self.add_norm2(x, self.ff(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, h, d_ff, N):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, h, d_ff) for _ in range(N)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MLPHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.linear(x[:, 0, :])  # CLS token only


class ViT(nn.Module):
    def __init__(self, in_channels, img_h, img_w,
                 patch_h, patch_w, d_model, d_ff, h, N, num_classes):
        super().__init__()
        num_patches = (img_h // patch_h) * (img_w // patch_w)

        self.patch_embedding = PatchEmbedding(in_channels, patch_h, patch_w, d_model)
        self.cls_token       = CLSToken(d_model)
        self.pos_embedding   = PositionalEncoding(num_patches, d_model)
        self.encoder         = Encoder(d_model, h, d_ff, N)
        self.mlp_head        = MLPHead(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.cls_token(x)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        return self.mlp_head(x)