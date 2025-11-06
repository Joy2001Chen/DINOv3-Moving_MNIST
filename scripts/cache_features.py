import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from data.moving_mnist import MovingMNISTDataset
from models.backbones.dino_v2 import DinoV2Features
from models.backbones.dino_v3 import DinoV3Features
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Precompute and cache backbone tokens for MovingMNIST")
    p.add_argument("--output", type=str, required=True,
                   help="Directory to write cached features (creates split-specific subfolders)")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                   help="Dataset split to cache")
    p.add_argument("--data-root", type=str, default="./data", help="MovingMNIST data root")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--cond-len", type=int, default=10)
    p.add_argument("--num-digits", type=int, default=2)
    p.add_argument("--num-seqs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", type=str, default="dinov3",
                   choices=["dinov3", "dinov2"], help="Backbone to use for token extraction")
    p.add_argument("--dino3-arch", type=str, default="vits16", choices=["vits16", "vitb16"])
    p.add_argument("--dino3-weights", type=str, default="")
    p.add_argument("--dino2-model", type=str, default="facebook/dinov2-small")
    p.add_argument("--torch-dtype", type=str, default="float16", choices=["float16", "float32"],
                   help="Precision for cached tokens")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cache directory")
    return p.parse_args()


def ensure_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Cache directory {path} already exists. Use --overwrite to replace.")
        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_root = Path(args.output) / args.split
    ensure_dir(cache_root, overwrite=args.overwrite)

    dataset = MovingMNISTDataset(root=args.data_root,
                                 split=args.split,
                                 seq_len=args.seq_len,
                                 cond_len=args.cond_len,
                                 num_digits=args.num_digits,
                                 num_sequences=args.num_seqs,
                                 deterministic=True,
                                 seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.backbone == "dinov3":
        backbone = DinoV3Features(arch=args.dino3_arch,
                                  weights_path=args.dino3_weights,
                                  trainable=False)
    else:
        backbone = DinoV2Features(model_name=args.dino2_model, trainable=False)
    backbone = backbone.to(device)

    torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.float32
    meta: Dict[str, Any] = {
        "split": args.split,
        "seq_len": args.seq_len,
        "cond_len": args.cond_len,
        "num_digits": args.num_digits,
        "num_sequences": args.num_seqs,
        "backbone": args.backbone,
        "torch_dtype": args.torch_dtype,
    }

    with torch.no_grad():
        sample_idx = 0
        for batch in loader:
            _, cond, target = batch
            cond = cond.to(device).float()
            target = target.to(device).float()
            whole = torch.cat([cond, target], dim=1)
            tokens, Hp, Wp = backbone.forward_patches(whole)
            tokens = tokens.cpu().to(torch_dtype)

            B = tokens.size(0)
            for b in range(B):
                sample_dir = cache_root / f"seq_{sample_idx:06d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                seq_tokens = tokens[b]
                cond_tokens = seq_tokens[:args.cond_len]
                target_tokens = seq_tokens[args.cond_len:]
                torch.save({
                    "cond_tokens": cond_tokens,
                    "target_tokens": target_tokens,
                    "Hp": Hp,
                    "Wp": Wp
                }, sample_dir / "tokens.pt")
                sample_idx += 1

    with open(cache_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
