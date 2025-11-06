# train_predict.py
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import amp
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from data.moving_mnist import MovingMNISTDataset
from models.latent_forecaster import LatentDynViTForecaster
from utils.metrics import mse_loss, psnr, ssim
from utils.seed import set_seed
from utils.viz import save_comparison_grid


def resolve_device(device_arg: str) -> torch.device:
    requested = (device_arg or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    try:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return device
    except (RuntimeError, ValueError) as exc:
        print(f"[Device] Unable to use '{device_arg}': {exc}. Falling back to CPU.")
        return torch.device("cpu")


def unpack_batch(batch):
    if isinstance(batch, dict):
        raise TypeError("Expected tuple batch from DataLoader, got dict. Please update collate_fn.")
    seq, cond, target = batch[0], batch[1], batch[2]
    feature_tuple = None
    collision = None
    idx = 3
    if len(batch) > idx:
        maybe_feat = batch[idx]
        if isinstance(maybe_feat, (tuple, list)) and len(maybe_feat) == 4:
            feature_tuple = maybe_feat
            idx += 1
    if len(batch) > idx:
        collision = batch[idx]
    return seq, cond, target, feature_tuple, collision

def get_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--cond-len", type=int, default=10)
    p.add_argument("--num-digits", type=int, default=2)
    p.add_argument("--train-seqs", type=int, default=50000)
    p.add_argument("--val-seqs", type=int, default=10000)
    # train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="auto",
                   help="Device to run on. 'auto' prefers CUDA when available, otherwise CPU.")
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--backbone", type=str, default="facebook/dinov2-small")
    p.add_argument("--temporal", type=str, choices=["lstm","transformer"], default="transformer")
    p.add_argument("--token-dim", type=int, default=384)  # dinov2-small=384, base=768
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--transformer-heads", type=int, default=8)
    p.add_argument("--transformer-ff", type=int, default=1024)
    # backbone variants
    p.add_argument("--use-dino3", action="store_true",
                   help="Use the offline DINOv3 backbone instead of HuggingFace DINOv2")
    p.add_argument("--dino3-arch", type=str, default="vits16", choices=["vits16", "vitb16"],
                   help="DINOv3 architecture to use when --use-dino3 is set")
    p.add_argument("--dino3-weights", type=str, default="",
                   help="Path to local DINOv3 weights .pth (optional if auto-detectable)")
    p.add_argument("--feature-cache-root", type=str, default="",
                   help="Path to cached backbone tokens. Pass directory containing split subfolders (train/val).")
    p.add_argument("--use-feature-cache", action="store_true",
                   help="Enable cached backbone tokens (requires deterministic dataset ordering).")
    p.add_argument("--cache-dtype", type=str, default="float16", choices=["float16", "float32"],
                   help="Expected dtype of cached tokens; used when loading to torch tensors.")
    p.add_argument("--debug-checks", action="store_true",
                   help="Print per-forward statistics to help debug collapsed predictions.")
    # misc
    p.add_argument("--save", type=str, default="checkpoints/latent_dino_lstm.pt")
    p.add_argument("--samples-out", type=str, default="outputs/train_samples")
    return p.parse_args()

def make_loaders(args):
    if args.use_feature_cache and not args.feature_cache_root:
        raise ValueError("--use-feature-cache requires --feature-cache-root to be set")

    cache_train = cache_val = None
    if args.use_feature_cache:
        cache_train = os.path.join(args.feature_cache_root, "train")
        cache_val = os.path.join(args.feature_cache_root, "val")

    ds_train = MovingMNISTDataset(root=args.data_root, split="train",
                                  seq_len=args.seq_len, cond_len=args.cond_len,
                                  num_digits=args.num_digits, num_sequences=args.train_seqs,
                                  deterministic=args.use_feature_cache,
                                  feature_cache_root=cache_train)
    ds_val   = MovingMNISTDataset(root=args.data_root, split="val",
                                  seq_len=args.seq_len, cond_len=args.cond_len,
                                  num_digits=args.num_digits, num_sequences=args.val_seqs,
                                  deterministic=True,
                                  feature_cache_root=cache_val)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val
    

def validate(model, dl, device, use_amp: bool, cache_dtype: torch.dtype):
    model.eval()
    mse_sum, psnr_sum, ssim_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in dl:
            seq, cond, target, feature_tuple, _ = unpack_batch(batch)
            non_blocking = device.type == "cuda"
            cond = cond.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            target = target.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            cond_tokens = None
            if feature_tuple is not None:
                cond_tokens = feature_tuple[0].to(device=device, dtype=cache_dtype, non_blocking=non_blocking).float()
            with amp.autocast('cuda', enabled=use_amp):
                pred_logits = model(cond, cached_tokens=cond_tokens)
            pred_logits = pred_logits.float()
            pred_probs = torch.sigmoid(pred_logits)
            print("pred sigmoid min/max/mean:", pred_probs.min().item(), pred_probs.max().item(), pred_probs.mean().item())
            mse_sum += mse_loss(pred_probs, target).item() * cond.size(0)
            psnr_sum += psnr(pred_probs, target).item() * cond.size(0)
            ssim_sum += ssim(pred_probs, target).item() * cond.size(0)
            n += cond.size(0)
    return mse_sum/n, psnr_sum/n, ssim_sum/n

def main():
    args = get_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[Config] requested_device={args.device}, using {device}")
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(args.samples_out, exist_ok=True)

    model_kwargs = dict(backbone_name=args.backbone,
                        temporal=args.temporal,
                        token_dim=args.token_dim,
                        d_model=args.d_model,
                        enc_layers=args.layers,
                        transformer_heads=args.transformer_heads,
                        transformer_ff=args.transformer_ff,
                        dropout=args.dropout,
                        cond_len=args.cond_len,
                        pred_len=args.seq_len - args.cond_len,
                        use_dino3=args.use_dino3,
                        dino3_arch=args.dino3_arch,
                        dino3_weights=args.dino3_weights,
                        debug_checks=args.debug_checks)
    try:
        model = LatentDynViTForecaster(**model_kwargs).to(device)
    except RuntimeError as exc:
        if device.type == "cuda":
            print(f"[Device] Failed to place model on CUDA ({exc}). Falling back to CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            device = torch.device("cpu")
            model = LatentDynViTForecaster(**model_kwargs).to(device)
        else:
            raise
    use_amp = device.type == "cuda"
    cache_dtype = getattr(torch, args.cache_dtype)

    trainable_params = (p for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(trainable_params, lr=args.lr)
    #scaler = GradScaler(enabled=use_amp)
    scaler = amp.GradScaler('cuda' if use_amp else 'cpu', enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    dl_train, dl_val = make_loaders(args)

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        for it, batch in enumerate(pbar):
            seq, cond, target, feature_tuple, _ = unpack_batch(batch)
            non_blocking = device.type == "cuda"
            cond = cond.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            target = target.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            cond_tokens = None
            if feature_tuple is not None:
                cond_tokens = feature_tuple[0].to(device=device, dtype=cache_dtype, non_blocking=non_blocking).float()
            opt.zero_grad(set_to_none=True)
            #with autocast(enabled=use_amp):
            with amp.autocast('cuda', enabled=use_amp):
                pred_logits = model(cond, cached_tokens=cond_tokens)
                loss = criterion(pred_logits, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if it % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # validation
        val_mse, val_psnr, val_ssim = validate(model, dl_val, device, use_amp, cache_dtype)
        print(f"[Val] MSE: {val_mse:.6f} | PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.4f}")

        # save best (by MSE)
        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(args)
            }, args.save)
            print(f"Saved checkpoint to {args.save}")

        # save a few visual samples
        with torch.no_grad():
            sample_batch = next(iter(dl_val))
            seq, cond, target, feature_tuple, _ = unpack_batch(sample_batch)
            non_blocking = device.type == "cuda"
            cond = cond[:4].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            target = target[:4].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            cond_tokens = None
            if feature_tuple is not None:
                cond_tokens = feature_tuple[0][:cond.size(0)].to(device=device, dtype=cache_dtype, non_blocking=non_blocking).float()
            with amp.autocast('cuda', enabled=use_amp):
                pred_logits = model(cond, cached_tokens=cond_tokens)
            pred_probs = torch.sigmoid(pred_logits.float())
            for i in range(cond.size(0)):
                save_comparison_grid(cond[i].cpu(), pred_probs[i].cpu(), target[i].cpu(),
                                     path=os.path.join(args.samples_out, f"epoch{epoch:02d}_sample{i}.png"))

if __name__ == "__main__":
    main()
