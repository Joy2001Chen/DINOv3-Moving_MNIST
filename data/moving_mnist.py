# data/moving_mnist.py
import math
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import datasets

class MovingMNISTDataset(Dataset):
    """
    Generate Moving-MNIST on the fly.

    Returns:
        - seq:    (T, 1, H, W) in [0,1]
        - cond:   (Tin, 1, H, W)
        - target: (Tout, 1, H, W)
        - (optional) collision_label: int 0/1 (if return_collision_label=True)
    """
    def __init__(self,
                 root: str,
                 split: str = "train",      # "train" | "val" | "test"
                 seq_len: int = 20,
                 cond_len: int = 10,
                 num_digits: int = 2,
                 image_size: int = 64,
                 step_range: Tuple[float, float] = (2.0, 4.0),
                 num_sequences: int = 10000,
                 deterministic: bool = False,
                 seed: int = 42,
                 return_collision_label: bool = False):
        super().__init__()
        assert split in {"train", "val", "test"}
        assert 0 < cond_len < seq_len
        self.seq_len, self.cond_len = seq_len, cond_len
        self.num_digits, self.image_size = num_digits, image_size
        self.step_range = step_range
        self.num_sequences = num_sequences
        self.deterministic = deterministic
        self.seed = seed
        self.return_collision_label = return_collision_label

        # MNIST pools: train/val use MNIST train; test uses MNIST test
        mnist_train = datasets.MNIST(root=root, train=True, download=True)
        mnist_test  = datasets.MNIST(root=root, train=False, download=True)
        if split in {"train", "val"}:
            self.digit_pool = mnist_train.data.numpy().astype(np.float32) / 255.0
        else:
            self.digit_pool = mnist_test.data.numpy().astype(np.float32) / 255.0
        self.digit_size = 28

        # for deterministic generation per-index
        self._base_seed = seed + {"train": 0, "val": 1, "test": 2}[split]

    def __len__(self):
        return self.num_sequences

    @staticmethod
    def _rect_overlap(x1, y1, ds, x2, y2):
        # axis-aligned rectangle overlap test (strict overlap > 0)
        return not (x1+ds <= x2 or x2+ds <= x1 or y1+ds <= y2 or y2+ds <= y1)

    def __getitem__(self, idx):
        # per-sample RNG (deterministic if enabled)
        if self.deterministic:
            rng = np.random.RandomState(self._base_seed + idx)
        else:
            rng = np.random

        H = W = self.image_size
        T = self.seq_len
        D = self.num_digits
        ds = self.digit_size

        # sample digits
        digits_idx = rng.randint(0, self.digit_pool.shape[0], size=D)
        digits = [self.digit_pool[i] for i in digits_idx]  # each (28,28)

        # initial positions (top-left) and velocities
        positions = []
        velocities = []
        for _ in range(D):
            x = rng.randint(0, W - ds)
            y = rng.randint(0, H - ds)
            theta = rng.rand() * 2 * math.pi
            speed = rng.uniform(*self.step_range)
            vx, vy = speed * math.cos(theta), speed * math.sin(theta)
            positions.append([x, y])
            velocities.append([vx, vy])

        # generate frames
        seq = np.zeros((T, 1, H, W), dtype=np.float32)
        collision = 0
        for t in range(T):
            canvas = np.zeros((H, W), dtype=np.float32)
            # check collision by bbox overlap
            if D >= 2:
                coll = False
                for i in range(D):
                    for j in range(i+1, D):
                        xi, yi = positions[i]
                        xj, yj = positions[j]
                        if self._rect_overlap(int(xi), int(yi), ds, int(xj), int(yj)):
                            coll = True
                            break
                    if coll:
                        break
                collision = 1 if (collision or coll) else 0

            for i in range(D):
                x, y = positions[i]
                x_i, y_i = int(round(x)), int(round(y))
                patch = digits[i]
                canvas[y_i:y_i+ds, x_i:x_i+ds] = np.maximum(
                    canvas[y_i:y_i+ds, x_i:x_i+ds], patch
                )
                # update pos with bounce at borders (mirror trick)
                vx, vy = velocities[i]
                x_new, y_new = x + vx, y + vy
                if x_new <= 0:
                    x_new = -x_new; vx = -vx
                if x_new >= W - ds:
                    x_new = 2*(W - ds) - x_new; vx = -vx
                if y_new <= 0:
                    y_new = -y_new; vy = -vy
                if y_new >= H - ds:
                    y_new = 2*(H - ds) - y_new; vy = -vy
                positions[i] = [x_new, y_new]
                velocities[i] = [vx, vy]
            seq[t, 0] = canvas

        cond = seq[:self.cond_len]                 # (Tin,1,H,W)
        target = seq[self.cond_len:]               # (Tout,1,H,W)

        if self.return_collision_label:
            return (torch.from_numpy(seq),
                    torch.from_numpy(cond),
                    torch.from_numpy(target),
                    torch.tensor(collision, dtype=torch.long))
        else:
            return (torch.from_numpy(seq),
                    torch.from_numpy(cond),
                    torch.from_numpy(target))
