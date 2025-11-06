
# models/latent_forecaster.py
import math
from typing import Literal
import torch
import torch.nn as nn

from models.backbones.dino_v2 import DinoV2Features
from models.temporal.lstm import LSTMEncoder
from models.backbones.dino_v3 import DinoV3Features

# robust import for various class names
from importlib import import_module
_tx = import_module("models.temporal.transformer")
TransformerEncoder = getattr(_tx, "TransformerEncoder", None) \
    or getattr(_tx, "Transformer", None) \
    or getattr(_tx, "TemporalTransformer", None)
# if TransformerEncoder is None:
#     raise ImportError("Cannot find a transformer temporal encoder class. "
#                       "Tried TransformerEncoder / Transformer / TemporalTransformer in models.temporal.transformer")

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
                 pred_len: int = 10,
                 use_dino3: bool = False,
                 dino3_arch: str = "vits16",
                 dino3_weights: str = "",
                 debug_checks: bool = False):
        super().__init__()
        if use_dino3:
            # 延迟导入，避免在未使用 DINOv3 时触发额外依赖
            self.backbone = DinoV3Features(arch=dino3_arch, weights_path=dino3_weights, trainable=False)
            backbone_dim = getattr(self.backbone, "token_dim", None)
        else:
            self.backbone = DinoV2Features(model_name=backbone_name, trainable=False)
            backbone_dim = getattr(getattr(self.backbone, "model", None), "config", None)
            backbone_dim = getattr(backbone_dim, "hidden_size", None)

        # ensure backbone stays frozen regardless of upstream defaults
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        if token_dim is None and backbone_dim is None:
            raise ValueError("Cannot infer token_dim; please provide --token-dim explicitly.")

        if backbone_dim is not None:
            if token_dim is None:
                token_dim = backbone_dim
            elif token_dim != backbone_dim:
                raise ValueError(f"token_dim ({token_dim}) does not match backbone output dim ({backbone_dim}).")

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

        # lightweight spatial mixer so temporal encoder sees neighbouring context
        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(token_dim, token_dim, kernel_size=1)
        )
        self.debug_checks = debug_checks

        # decoder maps predicted patch tokens back to image space
        self.decoder = TokenGridDecoder(c_token=token_dim, debug=debug_checks)

    def forward(self, cond_frames: torch.Tensor, cached_tokens: torch.Tensor | None = None):
        """
        Args:
            cond_frames: (B, Tin, 1, 64, 64), Tin == cond_len
            cached_tokens: optional precomputed patch tokens (B, Tin, N, C)
        Returns:
            pred_frames: (B, Tout, 1, 64, 64)
        """
        assert cond_frames.size(1) == self.cond_len
        if cached_tokens is not None:
            tokens = cached_tokens.to(cond_frames.device).float()
            B, Tin, N, C = tokens.shape
            Hp = int(math.sqrt(N))
            Wp = Hp
        else:
            with torch.no_grad():
                tokens, Hp, Wp = self.backbone.forward_patches(cond_frames)  # (B, Tin, N, C)
            tokens = tokens.float()
            B, Tin, N, C = tokens.shape
        B, Tin, N, C = tokens.shape

        # mix neighbouring patches within each frame to carry motion cues spatially
        mixed = tokens.view(B * Tin, Hp, Wp, C).permute(0, 3, 1, 2)            # (B*Tin, C, Hp, Wp)
        mixed = self.spatial_mixer(mixed)
        mixed = mixed.permute(0, 2, 3, 1).contiguous().view(B, Tin, N, C)      # (B, Tin, N, C)
        tokens = tokens + mixed

        if self.debug_checks:
            with torch.no_grad():
                token_stats = {
                    "min": tokens.min().item(),
                    "max": tokens.max().item(),
                    "mean": tokens.mean().item()
                }
                print(f"[Backbone] token stats: {token_stats}")
                if torch.isnan(tokens).any():
                    print("[Backbone] Warning: NaNs detected in tokens")

        # shared per-patch encoder: treat each patch location independently
        x = tokens.permute(0, 2, 1, 3).contiguous().view(B * N, Tin, C)  # (B*N, Tin, C)
        ctx = self.temporal(x)                                           # (B*N, D)
        y = self.mlp(ctx)                                                # (B*N, Tout*C)
        y = y.view(B, N, self.pred_len, C).permute(0, 2, 1, 3).contiguous()  # (B, Tout, N, C)
        y = y.view(B, self.pred_len, Hp, Wp, C)                          # grid tokens

        logits = self.decoder(y)                                         # (B, Tout, 1, 64, 64)
        return logits
