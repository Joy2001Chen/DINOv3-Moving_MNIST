# models/temporal/lstm.py
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    Shared per-patch temporal encoder.
    Input:  (B', T, C_in)
    Output: (B', D)  -- last hidden state (or mean pool)
    """
    def __init__(self, c_in: int, d_model: int = 512, num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=c_in, hidden_size=d_model//(2 if bidirectional else 1),
                            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional)
        self.d_model = d_model
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B', T, C_in)
        out, (hn, cn) = self.lstm(x)  # hn: (num_layers * num_dir, B', hidden)
        if self.bidirectional:
            # concat last layer's forward & backward
            h = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            h = hn[-1]
        return h  # (B', D)
