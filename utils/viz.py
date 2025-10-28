# utils/viz.py
import os
from typing import Optional
import torch
from PIL import Image

def save_comparison_grid(cond: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, path: str, scale: int = 2):
    """
    Make a single image: row1=cond, row2=pred, row3=target
    cond:   (Tin,1,H,W)   pred: (Tout,1,H,W)   target: (Tout,1,H,W)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cond = cond.detach().cpu().clamp(0,1)
    pred = pred.detach().cpu().clamp(0,1)
    target = target.detach().cpu().clamp(0,1)

    Tin = cond.size(0); Tout = pred.size(0)
    H, W = cond.size(2), cond.size(3)

    # to PIL tiles
    def to_img(t):
        arr = (t.squeeze(0).numpy() * 255).astype('uint8')  # (H,W)
        return Image.fromarray(arr, mode='L').resize((W*scale, H*scale), Image.NEAREST)

    tiles = [to_img(cond[i]) for i in range(Tin)] + \
            [to_img(pred[i]) for i in range(Tout)] + \
            [to_img(target[i]) for i in range(Tout)]
    cols = max(Tin, Tout)
    rows = 2 + 1  # cond row + pred row + gt row
    cell_w, cell_h = W*scale, H*scale
    grid = Image.new('L', (cols*cell_w, rows*cell_h), color=0)

    # paste cond
    for i in range(Tin):
        grid.paste(tiles[i], (i*cell_w, 0))
    # pred row
    offset = Tin
    for i in range(Tout):
        grid.paste(tiles[offset+i], (i*cell_w, cell_h))
    # target row
    offset += Tout
    for i in range(Tout):
        grid.paste(tiles[offset+i], (i*cell_w, 2*cell_h))

    grid.save(path)
