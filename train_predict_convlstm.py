# train_predict_convlstm.py
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data.moving_mnist import MovingMNISTDataset
from models.st_baselines.convlstm import ConvLSTMPredictor
from utils.metrics import mse_loss, psnr, ssim
from utils.seed import set_seed
from utils.viz import save_comparison_grid

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--cond-len", type=int, default=10)
    p.add_argument("--num-digits", type=int, default=2)
    p.add_argument("--train-seqs", type=int, default=50000)
    p.add_argument("--val-seqs", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--hid-ch", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="checkpoints/convlstm_baseline.pt")
    p.add_argument("--samples-out", type=str, default="outputs/convlstm_samples")
    return p.parse_args()

def make_loaders(args):
    ds_train = MovingMNISTDataset(root=args.data_root, split="train",
                                  seq_len=args.seq_len, cond_len=args.cond_len,
                                  num_digits=args.num_digits, num_sequences=args.train_seqs,
                                  deterministic=False)
    ds_val   = MovingMNISTDataset(root=args.data_root, split="val",
                                  seq_len=args.seq_len, cond_len=args.cond_len,
                                  num_digits=args.num_digits, num_sequences=args.val_seqs,
                                  deterministic=True)
    #新增：persistent_workers=True, prefetch_factor=2：解决“数据没及时喂到 GPU → 训练卡顿”的问题
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True, prefetch_factor=2)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return dl_train, dl_val

def validate(model, dl, device):
    model.eval()
    mse_sum, psnr_sum, ssim_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for seq, cond, target in dl:
            cond = cond.to(device).float()
            target = target.to(device).float()
            pred = model(cond)
            mse_sum += mse_loss(pred, target).item() * cond.size(0)
            psnr_sum += psnr(pred, target).item() * cond.size(0)
            ssim_sum += ssim(pred, target).item() * cond.size(0)
            n += cond.size(0)
    return mse_sum/n, psnr_sum/n, ssim_sum/n

# 打印模型参数统计与每个参数的形状
def print_model_stats(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Model params: total={total:,}, trainable={trainable:,}")
    for name, p in m.named_parameters():
        print(f"{name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}, numel={p.numel()}")


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(args.samples_out, exist_ok=True)

    device = torch.device(args.device)
    model = ConvLSTMPredictor(in_ch=1, hid_ch=args.hid_ch, num_layers=args.layers,
                              pred_len=args.seq_len - args.cond_len).to(device)
    # print_model_stats(model)
    # Model params: total=445,249, trainable=445,249
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    dl_train, dl_val = make_loaders(args)

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        for it, (seq, cond, target) in enumerate(pbar):
            cond = cond.to(device).float()
            target = target.to(device).float()
            target = target[:, :args.seq_len - args.cond_len]  # ensure correct length
            pred = model(cond) # (B, Tout, 1, H, W)
            # print(pred.shape, target.shape)
            print(pred.min().item(), pred.max().item(), target.min().item(), target.max().item())
            loss = mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_mse, val_psnr, val_ssim = validate(model, dl_val, device)
        print(f"[Val] MSE: {val_mse:.6f} | PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save({"model": model.state_dict(), "cfg": vars(args)}, args.save)
            print(f"Saved baseline checkpoint to {args.save}")

        # qualitative
        with torch.no_grad():
            seq, cond, target = next(iter(dl_val))
            cond = cond[:4].to(device).float()
            target = target[:4].to(device).float()
            pred = model(cond)
            for i in range(cond.size(0)):
                save_comparison_grid(cond[i].cpu(), pred[i].cpu(), target[i].cpu(),
                                     path=os.path.join(args.samples_out, f"epoch{epoch:02d}_sample{i}.png"))

if __name__ == "__main__":
    main()
