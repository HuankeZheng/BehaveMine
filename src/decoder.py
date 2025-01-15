import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, share_dim):
        super(DecoderLayer, self).__init__()
        self.norm = nn.LayerNorm(share_dim[0])
        self.attn = nn.MultiheadAttention(embed_dim=share_dim[0], num_heads=share_dim[1])
        self.feedforward = nn.Sequential(
            nn.Linear(share_dim[0], share_dim[0] * 2),
            nn.ReLU(),
            nn.Linear(share_dim[0] * 2, share_dim[0])
        )

    def forward(self, x, mask=None):
        # Multi
        y = self.norm(x)
        y = y.permute(1, 0, 2)
        y = self.attn(y, y, y, attn_mask=mask)[0]
        y = y.permute(1, 0, 2)
        x = x + y
        # Feedforward
        y = self.norm(x)
        y = self.feedforward(y)
        x = x + y
        return x


class Decoder(nn.Module):
    def __init__(self, share_dim):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(share_dim[:-1]) for _ in range(share_dim[-1])])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
