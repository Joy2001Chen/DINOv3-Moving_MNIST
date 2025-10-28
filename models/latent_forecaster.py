
# models/latent_forecaster.py
from typing import Literal
import torch
import torch.nn as nn

from models.backbones.dino_v2 import DinoV2Features
from models.temporal.lstm import LSTMEncoder

# robust import for various class names
from importlib import import_module
_tx = import_module("models.temporal.transformer")
TransformerEncoder = getattr(_tx, "TransformerEncoder", None) \
    or getattr(_tx, "Transformer", None) \
    or getattr(_tx, "TemporalTransformer", None)
if TransformerEncoder is None:
    raise ImportError("Cannot find a transformer temporal encoder class. "
                      "Tried TransformerEncoder / Transformer / TemporalTransformer in models.temporal.transformer")

from models.decoders.conv_decoder import TokenGridDecoder

class LatentDynViTForecaster(nn.Module):
    """
    DINOv2 (frozen) → per-patch temporal encoder (shared) → predict future patch tokens → decode to frames.
    """
    def __init__(self,
                 backbone_name: str = "facebook/dinov2-small",
                 temporal: Literal["lstm", "transformer"] = "lstm",
                 token_dim: int = 384,     # dinov2-small hidden size
                 d_model: int = 512,
                 enc_layers: int = 2,
                 transformer_heads: int = 8,
                 transformer_ff: int = 1024,
                 dropout: float = 0.1,
                 cond_len: int = 10,
                 pred_len: int = 10):
        super().__init__()
        self.backbone = DinoV2Features(model_name=backbone_name, trainable=False)
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.token_dim = token_dim

        # per-patch temporal encoder
        if temporal == "lstm":
            self.temporal = LSTMEncoder(c_in=token_dim, d_model=d_model, num_layers=enc_layers, dropout=dropout)
        elif temporal == "transformer":
            self.temporal = TransformerEncoder(c_in=token_dim, d_model=d_model, num_layers=enc_layers,
                                               nhead=transformer_heads, dim_feedforward=transformer_ff, dropout=dropout)
        else:
            raise ValueError(f"Unknown temporal type: {temporal}")

        # MLP head to generate pred_len * token_dim per patch
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, pred_len * token_dim)
        )

        # decoder will be initialized after first forward (needs Hp/Wp)
        self.decoder = None

    def forward(self, cond_frames: torch.Tensor):
        """
        Args:
            cond_frames: (B, Tin, 1, 64, 64), Tin == cond_len
        Returns:
            pred_frames: (B, Tout, 1, 64, 64)
        """
        assert cond_frames.size(1) == self.cond_len
        with torch.no_grad():
            tokens, Hp, Wp = self.backbone.forward_patches(cond_frames)  # (B, Tin, N, C)
        B, Tin, N, C = tokens.shape
        # shared per-patch encoder: treat each patch location independently
        x = tokens.permute(0, 2, 1, 3).contiguous().view(B * N, Tin, C)  # (B*N, Tin, C)
        ctx = self.temporal(x)                                           # (B*N, D)
        y = self.mlp(ctx)                                                # (B*N, Tout*C)
        y = y.view(B, N, self.pred_len, C).permute(0, 2, 1, 3).contiguous()  # (B, Tout, N, C)
        y = y.view(B, self.pred_len, Hp, Wp, C)                          # grid tokens

        if self.decoder is None:
            self.decoder = TokenGridDecoder(c_token=C, hp=Hp, wp=Wp)

        pred_frames = self.decoder(y)                                    # (B, Tout, 1, 64, 64)
        return pred_frames
