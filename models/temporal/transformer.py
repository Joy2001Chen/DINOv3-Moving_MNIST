# models/temporal/transformer.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerEncoder(nn.Module):
    """
    Shared per-patch temporal encoder (Transformer).
    Input:  (B', T, C_in)
    Output: (B', D)  -- mean pooled Transformer outputs
    """
    def __init__(self, c_in: int, d_model: int = 512, num_layers: int = 4, nhead: int = 8, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(c_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B', T, C_in)
        x = self.in_proj(x)
        x = self.pos(x)
        x = self.encoder(x)     # (B', T, D)
        h = x.mean(dim=1)       # global average over time
        return h                # (B', D)
