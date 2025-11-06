# train_classify_collision.py
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.moving_mnist import MovingMNISTDataset
from models.backbones.dino_v2 import DinoV2Features
from models.temporal.lstm import LSTMEncoder
from models.temporal.transformer import TransformerEncoder
from utils.seed import set_seed

def get_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--cond-len", type=int, default=10)
    p.add_argument("--num-digits", type=int, default=2)
    p.add_argument("--train-seqs", type=int, default=40000)
    p.add_argument("--val-seqs", type=int, default=10000)
    # train
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--backbone", type=str, default="facebook/dinov2-small")
    p.add_argument("--temporal", type=str, choices=["lstm","transformer"], default="lstm")
    p.add_argument("--token-dim", type=int, default=384)  # dinov2-small
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--transformer-heads", type=int, default=8)
    p.add_argument("--transformer-ff", type=int, default=1024)
    p.add_argument("--save", type=str, default="checkpoints/collision_classifier.pt")
    return p.parse_args()

def make_loaders(args):
    ds_train = MovingMNISTDataset(root=args.data_root, split="train",
                                  seq_len=args.seq_len, cond_len=args.cond_len,
                                  num_digits=args.num_digits, num_sequences=args.train_seqs,
                                  deterministic=False, return_collision_label=True)
    ds_val   = MovingMNISTDataset(root=args.data_root, split="val",
                                  seq_len=args.seq_len, cond_len=args.cond_len,
                                  num_digits=args.num_digits, num_sequences=args.val_seqs,
                                  deterministic=True, return_collision_label=True)
    dl_tr = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_tr, dl_va

class LatentSequenceClassifier(nn.Module):
    def __init__(self, backbone_name: str, temporal: str, token_dim: int, d_model: int, layers: int,
                 transformer_heads: int, transformer_ff: int, dropout: float, cond_len: int):
        super().__init__()
        self.backbone = DinoV2Features(model_name=backbone_name, trainable=False)
        self.cond_len = cond_len
        if temporal == "lstm":
            self.temporal = LSTMEncoder(c_in=token_dim, d_model=d_model, num_layers=layers, dropout=dropout)
        else:
            self.temporal = TransformerEncoder(c_in=token_dim, d_model=d_model, num_layers=layers,
                                               nhead=transformer_heads, dim_feedforward=transformer_ff, dropout=dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2)
        )

    def forward(self, cond_frames):
        # use CLS token per frame; then temporal model; then classify
        with torch.no_grad():
            x = self.backbone.forward_cls(cond_frames)   # (B, Tin, C)
        h = self.temporal(x)                              # (B, D)
        logits = self.cls_head(h)                         # (B, 2)
        return logits

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    dl_tr, dl_va = make_loaders(args)
    device = torch.device(args.device)
    model = LatentSequenceClassifier(backbone_name=args.backbone,
                                     temporal=args.temporal,
                                     token_dim=args.token_dim,
                                     d_model=args.d_model,
                                     layers=args.layers,
                                     transformer_heads=args.transformer_heads,
                                     transformer_ff=args.transformer_ff,
                                     dropout=args.dropout,
                                     cond_len=args.cond_len).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        # train
        model.train()
        total, correct = 0, 0
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs}")
        for seq, cond, target, label in pbar:
            cond = cond.to(device).float()
            label = label.to(device)
            print(label.shape)
            logits = model(cond)
            print(logits.shape)
            pr
            loss = criterion(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

        # val
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for seq, cond, target, label in dl_va:
                cond = cond.to(device).float()
                label = label.to(device)
                logits = model(cond)
                pred = logits.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        acc = correct / total
        print(f"[Val] Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "cfg": vars(args)}, args.save)
            print(f"Saved classifier to {args.save}")

if __name__ == "__main__":
    main()
