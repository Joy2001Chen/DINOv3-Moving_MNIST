# models/st_baselines/convlstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4*hid_ch, kernel_size, padding=padding)
        self.hid_ch = hid_ch

    def forward(self, x, h, c):
        # x: (B, in_ch, H, W), h/c: (B, hid_ch, H, W)
        out = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(out, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTMPredictor(nn.Module):
    """
    Simple 2-layer ConvLSTM that predicts future frames from past frames.
    """
    def __init__(self, in_ch: int = 1, hid_ch: int = 64, num_layers: int = 2, pred_len: int = 10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.pred_len = pred_len

        ch_in = in_ch
        for _ in range(num_layers):
            self.layers.append(ConvLSTMCell(ch_in, hid_ch, kernel_size=3))
            ch_in = hid_ch
        self.out = nn.Conv2d(hid_ch, 1, kernel_size=1)

    def forward(self, cond_frames):
        # cond_frames: (B, Tin, 1, 64, 64)
        B, Tin, C, H, W = cond_frames.shape
        device = cond_frames.device
        hs = [torch.zeros(B, self.layers[0].hid_ch, H, W, device=device) for _ in range(self.num_layers)]
        cs = [torch.zeros_like(hs[0]) for _ in range(self.num_layers)]

        # encode past
        for t in range(Tin):
            x = cond_frames[:, t]
            for l, cell in enumerate(self.layers):
                h, c = hs[l], cs[l]
                h, c = cell(x, h, c)
                hs[l], cs[l] = h, c
                x = h

        # predict future, autoregressive
        preds = []
        x = cond_frames[:, -1]  # last frame as input seed
        for t in range(self.pred_len):
            xx = x
            for l, cell in enumerate(self.layers):
                h, c = hs[l], cs[l]
                h, c = cell(xx, h, c)
                hs[l], cs[l] = h, c
                xx = h
            y = torch.sigmoid(self.out(xx))
            preds.append(y)
            x = y  # feed back
        return torch.stack(preds, dim=1)  # (B, Tout, 1, H, W)
