# eval_predict.py
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data.moving_mnist import MovingMNISTDataset
from models.latent_forecaster import LatentDynViTForecaster
from utils.metrics import mse_loss, psnr, ssim
from utils.viz import save_comparison_grid

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--cond-len", type=int, default=10)
    p.add_argument("--num-digits", type=int, default=2)
    p.add_argument("--test-seqs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--outdir", type=str, default="outputs/test_samples")
    return p.parse_args()

def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]
    device = torch.device(args.device)

    model = LatentDynViTForecaster(backbone_name=cfg.get("backbone", "facebook/dinov2-small"),
                                   temporal=cfg["temporal"],
                                   token_dim=cfg["token_dim"],
                                   d_model=cfg["d_model"],
                                   enc_layers=cfg["layers"],
                                   transformer_heads=cfg["transformer_heads"],
                                   transformer_ff=cfg["transformer_ff"],
                                   dropout=cfg["dropout"],
                                   cond_len=cfg["cond_len"],
                                   pred_len=cfg["seq_len"] - cfg["cond_len"],
                                   use_dino3=cfg.get("use_dino3", False),
                                   dino3_arch=cfg.get("dino3_arch", "vits16"),
                                   dino3_weights=cfg.get("dino3_weights", "")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds_test = MovingMNISTDataset(root=args.data_root, split="test",
                                 seq_len=args.seq_len, cond_len=args.cond_len,
                                 num_digits=args.num_digits, num_sequences=args.test_seqs,
                                 deterministic=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    mse_sum, psnr_sum, ssim_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for seq, cond, target in tqdm(dl_test, desc="Eval"):
            cond = cond.to(device).float()
            target = target.to(device).float()
            pred = model(cond)
            mse_sum += mse_loss(pred, target).item() * cond.size(0)
            psnr_sum += psnr(pred, target).item() * cond.size(0)
            ssim_sum += ssim(pred, target).item() * cond.size(0)
            n += cond.size(0)

    print(f"[Test] MSE: {mse_sum/n:.6f} | PSNR: {psnr_sum/n:.3f} | SSIM: {ssim_sum/n:.4f}")

    # save qualitative samples
    with torch.no_grad():
        seq, cond, target = next(iter(dl_test))
        cond = cond[:args.num_samples].to(device).float()
        target = target[:args.num_samples].to(device).float()
        pred = model(cond)
        for i in range(cond.size(0)):
            save_comparison_grid(cond[i].cpu(), pred[i].cpu(), target[i].cpu(),
                                 path=os.path.join(args.outdir, f"sample{i}.png"))

if __name__ == "__main__":
    main()
