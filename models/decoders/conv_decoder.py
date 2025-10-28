# models/decoders/conv_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenGridDecoder(nn.Module):
    """
    Convert predicted patch-token grid to 64x64 grayscale images.
    Input:  tokens_pred: (B, Tout, Hp, Wp, C_token)
    Output: frames_pred: (B, Tout, 1, 64, 64) in [0,1]
    """
    def __init__(self, c_token: int, hp: int = 16, wp: int = 16, hidden: int = 256):
        super().__init__()
        # project token dim to conv channels
        self.proj = nn.Conv2d(c_token, hidden, kernel_size=1)
        # Up: 16 -> 32 -> 64 (assumes Hp=Wp=16; works with typical DINOv2 patch14 @224)
        self.deconv1 = nn.ConvTranspose2d(hidden, hidden//2, kernel_size=4, stride=2, padding=1)  # 16->32
        self.deconv2 = nn.ConvTranspose2d(hidden//2, hidden//4, kernel_size=4, stride=2, padding=1)  # 32->64
        self.out = nn.Conv2d(hidden//4, 1, kernel_size=3, padding=1)

    def forward(self, tokens_pred: torch.Tensor) -> torch.Tensor:
        B, Tout, Hp, Wp, C = tokens_pred.shape
        x = tokens_pred.view(B*Tout, Hp, Wp, C).permute(0, 3, 1, 2)  # (B*Tout, C, Hp, Wp)
        x = self.proj(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.out(x))
        x = x.view(B, Tout, 1, 64, 64)
        return x
