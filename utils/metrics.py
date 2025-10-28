# utils/metrics.py
import torch
import torch.nn.functional as F

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # per-sample
    psnr = 10.0 * torch.log10((max_val ** 2) / (mse + 1e-8))
    return psnr.mean()

# Lightweight SSIM (grayscale, [0,1])
def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    pred/target: (B, T, 1, H, W) or (B, 1, H, W)
    Returns mean SSIM over batch/time.
    """
    if pred.dim() == 5:
        B, T = pred.size(0), pred.size(1)
        pred = pred.view(B*T, 1, pred.size(3), pred.size(4))
        target = target.view(B*T, 1, target.size(3), target.size(4))
    else:
        B = pred.size(0)

    # create gaussian window
    import math
    device = pred.device
    def gauss(kernel_size, sigma):
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx = (ax ** 2)
        kernel = torch.exp(-(xx) / (2*sigma*sigma))
        kernel = kernel / kernel.sum()
        return kernel

    sigma = 1.5
    w = gauss(window_size, sigma).unsqueeze(0)
    window = (w.t() @ w).unsqueeze(0).unsqueeze(0)  # (1,1,ks,ks)
    window = window.to(device)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=1) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)

    return ssim_map.mean()
