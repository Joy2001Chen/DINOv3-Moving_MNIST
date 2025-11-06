import argparse
import datetime
import json
import math
import random
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.cuda.amp import GradScaler, autocast
from torch import amp


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DigitRenderer:
    def __init__(self, size: int = 28) -> None:
        self.size = size
        font = ImageFont.load_default()
        digits = []
        for d in range(10):
            canvas = Image.new("L", (16, 16), 0)
            draw = ImageDraw.Draw(canvas)
            w, h = draw.textsize(str(d), font=font)
            draw.text(((16 - w) / 2, (16 - h) / 2), str(d), fill=255, font=font)
            digits.append(np.array(canvas.resize((size, size), Image.NEAREST), dtype=np.float32) / 255.0)
        self.digits = np.stack(digits, axis=0)

    def get(self, digit: int) -> np.ndarray:
        return self.digits[digit]


class MovingMNISTGenerator:
    def __init__(
        self,
        seed: int,
        t_in: int,
        t_out: int,
        canvas_size: int = 64,
        digit_size: int = 28,
        num_digits_range: Tuple[int, int] = (1, 2),
        max_speed: float = 3.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.t_in = t_in
        self.t_out = t_out
        self.seq_len = t_in + t_out
        self.canvas_size = canvas_size
        self.digit_size = digit_size
        self.num_digits_range = num_digits_range
        self.max_speed = max_speed
        self.max_pos = canvas_size - digit_size
        self.renderer = DigitRenderer(size=digit_size)

    def _sample_velocities(self, n: int) -> np.ndarray:
        vel = self.rng.uniform(-self.max_speed, self.max_speed, size=(n, 2)).astype(np.float32)
        mask = np.abs(vel) < 0.5
        while mask.any():
            vel[mask] = self.rng.uniform(-self.max_speed, self.max_speed, size=mask.sum())
            mask = np.abs(vel) < 0.5
        return vel

    def _bounce(self, pos: np.ndarray, vel: np.ndarray) -> None:
        for axis in (0, 1):
            low = pos[:, axis] < 0
            if low.any():
                pos[low, axis] = -pos[low, axis]
                vel[low, axis] *= -1
            high = pos[:, axis] > self.max_pos
            if high.any():
                pos[high, axis] = 2 * self.max_pos - pos[high, axis]
                vel[high, axis] *= -1

    def _sample_sequence(self) -> np.ndarray:
        num_digits = int(self.rng.integers(self.num_digits_range[0], self.num_digits_range[1] + 1))
        digit_ids = self.rng.integers(0, 10, size=num_digits)
        positions = self.rng.uniform(0, self.max_pos, size=(num_digits, 2)).astype(np.float32)
        velocities = self._sample_velocities(num_digits)
        frames = np.zeros((self.seq_len, self.canvas_size, self.canvas_size), dtype=np.float32)
        for t in range(self.seq_len):
            canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
            for idx, digit_id in enumerate(digit_ids):
                xi = min(max(int(round(float(positions[idx, 0]))), 0), self.max_pos)
                yi = min(max(int(round(float(positions[idx, 1]))), 0), self.max_pos)
                digit_img = self.renderer.get(int(digit_id))
                patch = canvas[yi : yi + self.digit_size, xi : xi + self.digit_size]
                canvas[yi : yi + self.digit_size, xi : xi + self.digit_size] = np.maximum(patch, digit_img)
            frames[t] = canvas
            positions += velocities
            self._bounce(positions, velocities)
        return frames

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        sequences = np.stack([self._sample_sequence() for _ in range(batch_size)], axis=0).astype(np.float32)
        inputs = sequences[:, : self.t_in, None, :, :]
        targets = sequences[:, self.t_in :, None, :, :]
        return inputs, targets


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        cells = []
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            cells.append(ConvLSTMCell(in_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def init_state(self, batch_size: int, spatial_size: Tuple[int, int], device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        h, w = spatial_size
        states = []
        for _ in range(self.num_layers):
            h_state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
            c_state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
            states.append((h_state, c_state))
        return states

    def forward(self, x: torch.Tensor, states: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        inputs = x
        new_states = []
        for idx, cell in enumerate(self.cells):
            h, c = states[idx]
            h_next, c_next = cell(inputs, (h, c))
            inputs = h_next
            new_states.append((h_next, c_next))
        return new_states


class ConvLSTMForecaster(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convlstm = ConvLSTM(hidden_dim, hidden_dim, num_layers)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor, future_steps: int) -> torch.Tensor:
        b, t_in, _, h, w = inputs.shape
        device = inputs.device
        states = self.convlstm.init_state(b, (h, w), device)

        prev_frame = inputs[:, -1]
        for t in range(t_in):
            enc = self.encoder(inputs[:, t])
            states = self.convlstm(enc, states)

        preds: List[torch.Tensor] = []
        current = prev_frame
        for _ in range(future_steps):
            enc = self.encoder(current)
            states = self.convlstm(enc, states)
            out = self.decoder(states[-1][0])
            preds.append(out)
            current = out
        return torch.stack(preds, dim=1)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema = self._clone_model(model)

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        ema_model = copy.deepcopy(model).to(next(model.parameters()).device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            ema_params = dict(self.ema.named_parameters())
            model_params = dict(model.named_parameters())
            for name, param in model_params.items():
                ema_params[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)
            ema_buffers = dict(self.ema.named_buffers())
            model_buffers = dict(model.named_buffers())
            for name, buf in model_buffers.items():
                if name in ema_buffers:
                    ema_buffers[name].copy_(buf)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.ema.state_dict()

    def to(self, device: torch.device) -> None:
        self.ema.to(device)


class WarmupCosineLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.step_idx = 0

    def _compute_lr(self, step: int) -> float:
        if self.total_steps <= 0:
            return self.base_lr
        if step < self.warmup_steps and self.warmup_steps > 0:
            scale = float(step + 1) / float(self.warmup_steps)
            return self.base_lr * scale
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        step = min(self.step_idx, self.total_steps - 1)
        lr = self._compute_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.step_idx += 1
        return lr


def create_gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss /= gauss.sum()
    window = gauss[:, None] * gauss[None, :]
    return window.view(1, 1, window_size, window_size)


def ssim_per_sequence(pred: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    b, t, _, h, w = pred.shape
    pred = pred.reshape(b * t, 1, h, w)
    target = target.reshape(b * t, 1, h, w)
    window = window.to(pred.device, dtype=pred.dtype)
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(pred, window, padding=padding, groups=1)
    mu_y = F.conv2d(target, window, padding=padding, groups=1)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(pred * pred, window, padding=padding, groups=1) - mu_x2
    sigma_y = F.conv2d(target * target, window, padding=padding, groups=1) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=1) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
    ssim_frame = ssim_map.mean(dim=(1, 2, 3))
    return ssim_frame.view(b, t).mean(dim=1)


def sequence_metrics(pred: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse_frames = (pred - target) ** 2
    mse_seq = mse_frames.mean(dim=(2, 3, 4)).mean(dim=1)
    psnr_seq = 10.0 * torch.log10(torch.clamp(1.0 / torch.clamp(mse_seq, min=1e-8), min=1e-8))
    ssim_seq = ssim_per_sequence(pred, target, window)
    return mse_seq, psnr_seq, ssim_seq


def save_sample_grid(
    input_seq: torch.Tensor,
    target_seq: torch.Tensor,
    pred_seq: torch.Tensor,
    path: Path,
) -> None:
    t_in = input_seq.shape[0]
    t_out = target_seq.shape[0]
    frames_top = [input_seq[i, 0].cpu().numpy() for i in range(t_in)]
    frames_mid = [target_seq[i, 0].cpu().numpy() for i in range(t_out)]
    frames_bot = [pred_seq[i, 0].cpu().numpy() for i in range(t_out)]
    cols = max(t_in, t_out)

    def assemble_row(frames: List[np.ndarray], cols: int) -> np.ndarray:
        row = []
        blank = np.zeros_like(frames[0])
        for idx in range(cols):
            if idx < len(frames):
                row.append(frames[idx])
            else:
                row.append(blank)
        return np.concatenate(row, axis=1)

    top = assemble_row(frames_top, cols)
    mid = assemble_row(frames_mid, cols)
    bot = assemble_row(frames_bot, cols)
    grid = np.concatenate([top, mid, bot], axis=0)
    grid = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(grid).save(path)


def format_progress(epoch: int, train_loss: float, metrics: Dict[str, float]) -> str:
    return (
        f"[Epoch {epoch}] train_loss={train_loss:.4f}, "
        f"val: MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}"
    )


def parse_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_str)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def evaluate_model(
    model: nn.Module,
    generator: MovingMNISTGenerator,
    steps: int,
    batch_size: int,
    device: torch.device,
    window: torch.Tensor,
    samples_dir: Optional[Path] = None,
    samples_per_eval: int = 0,
    epoch_tag: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, List[torch.Tensor]] = {"mse": [], "psnr": [], "ssim": []}
    saved = 0
    with torch.no_grad():
        for step in range(steps):
            inputs_np, targets_np = generator.sample_batch(batch_size)
            inputs = torch.from_numpy(inputs_np).to(device)
            targets = torch.from_numpy(targets_np).to(device)
            preds = model(inputs, targets.shape[1])
            mse_seq, psnr_seq, ssim_seq = sequence_metrics(preds, targets, window)
            metrics["mse"].append(mse_seq)
            metrics["psnr"].append(psnr_seq)
            metrics["ssim"].append(ssim_seq)
            if samples_dir is not None and saved < samples_per_eval:
                bs = inputs.shape[0]
                for b in range(min(bs, samples_per_eval - saved)):
                    fname = f"{epoch_tag or 'eval'}_sample{saved:02d}.png"
                    save_sample_grid(
                        inputs[b].cpu(),
                        targets[b].cpu(),
                        preds[b].cpu(),
                        samples_dir / fname,
                    )
                    saved += 1
    aggregated = {k: torch.cat(v).mean().item() if v else 0.0 for k, v in metrics.items()}
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvLSTM baseline for Moving-MNIST.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tin", type=int, default=10)
    parser.add_argument("--tout", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=20)
    parser.add_argument("--test_steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = parse_device(args.device)

    train_gen = MovingMNISTGenerator(seed=args.seed, t_in=args.tin, t_out=args.tout)
    val_gen = MovingMNISTGenerator(seed=args.seed + 1, t_in=args.tin, t_out=args.tout)
    test_gen = MovingMNISTGenerator(seed=args.seed + 2, t_in=args.tin, t_out=args.tout)

    model = ConvLSTMForecaster(hidden_dim=args.hidden, num_layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, args.epochs * args.train_steps)
    scheduler = WarmupCosineLRScheduler(
        optimizer,
        base_lr=args.lr,
        warmup_steps=args.train_steps,
        total_steps=total_steps,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    ema_helper = ModelEMA(model, decay=args.ema) if args.ema and args.ema > 0 else None
    eval_model = ema_helper.ema if ema_helper is not None else model

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path("checkpoints") / timestamp
    sample_dir = Path("outputs") / timestamp / "samples"
    eval_dir = Path("outputs") / timestamp / "eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    window = create_gaussian_window()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for _ in range(args.train_steps):
            scheduler.step()
            inputs_np, targets_np = train_gen.sample_batch(args.batch_size)
            inputs = torch.from_numpy(inputs_np).to(device)
            targets = torch.from_numpy(targets_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                preds = model(inputs, args.tout)
                loss = F.l1_loss(preds, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if ema_helper is not None:
                ema_helper.update(model)
            running_loss += loss.item()

        average_loss = running_loss / max(1, args.train_steps)
        val_metrics = evaluate_model(
            eval_model,
            val_gen,
            args.val_steps,
            args.batch_size,
            device,
            window,
            samples_dir=sample_dir,
            samples_per_eval=args.samples,
            epoch_tag=f"epoch{epoch:02d}",
        )
        print(format_progress(epoch, average_loss, val_metrics))

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": args.epochs,
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "ema": ema_helper.state_dict() if ema_helper is not None else None,
        "args": vars(args),
    }
    torch.save(checkpoint, ckpt_dir / "model.pt")

    test_metrics = evaluate_model(
        eval_model,
        test_gen,
        args.test_steps,
        args.batch_size,
        device,
        window,
    )
    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Test: MSE={test_metrics['mse']:.6f}, PSNR={test_metrics['psnr']:.2f}, SSIM={test_metrics['ssim']:.4f}")


if __name__ == "__main__":
    main()
