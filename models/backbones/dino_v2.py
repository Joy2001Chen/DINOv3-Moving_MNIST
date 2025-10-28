# models/backbones/dino_v2.py
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class DinoV2Features(nn.Module):
    """
    Frozen DINOv2 ViT as a patch-token feature extractor.
    - Input frames: (B, T, 1, 64, 64) in [0,1]
    - Output patch tokens: (B, T, N, C), where N = H_p * W_p (default 16*16 for 224/14)
    - Also provides CLS tokens for sequence classification: (B, T, C)
    """
    def __init__(self, model_name: str = "facebook/dinov2-small", trainable: bool = False):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.image_size = 224  # Dinov2 default
        self.patch_grid = None # inferred at first forward

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

    @torch.no_grad()
    def forward_patches(self, frames: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            frames: (B, T, 1, 64, 64) in [0,1]
        Returns:
            patch_tokens: (B, T, N, C)
            Hp, Wp: patch grid size (e.g., 16,16)
        """
        assert frames.dim() == 5 and frames.size(2) == 1, "Expect (B,T,1,H,W)"
        B, T, _, H, W = frames.shape
        x = frames.repeat(1, 1, 3, 1, 1)          # to 3 channels
        x = x.reshape(B*T, 3, H, W)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x)          # last_hidden_state: (B*T, 1+N, C)
        tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
        N, C = tokens.shape[1], tokens.shape[2]
        Hp = int(math.sqrt(N)); Wp = Hp
        tokens = tokens.reshape(B, T, N, C)
        return tokens, Hp, Wp

    @torch.no_grad()
    def forward_cls(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Returns CLS embedding per frame: (B, T, C)
        """
        assert frames.dim() == 5 and frames.size(2) == 1, "Expect (B,T,1,H,W)"
        B, T, _, H, W = frames.shape
        x = frames.repeat(1, 1, 3, 1, 1)
        x = x.reshape(B*T, 3, H, W)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x)
        cls = out.last_hidden_state[:, 0, :]      # (B*T, C)
        return cls.reshape(B, T, -1)
