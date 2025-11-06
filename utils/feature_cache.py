from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class CachedFeatureBatch:
    cond_tokens: torch.Tensor
    target_tokens: torch.Tensor
    Hp: int
    Wp: int


class FeatureCache:
    """Simple filesystem-backed cache for patch tokens."""

    def __init__(self, root: str, readonly: bool = False):
        self.root = Path(root)
        self.readonly = readonly
        if readonly and not self.root.exists():
            raise FileNotFoundError(f"Cache root {self.root} does not exist")
        if not self.readonly:
            self.root.mkdir(parents=True, exist_ok=True)

    def _sample_dir(self, key: str) -> Path:
        return self.root / key

    def has(self, key: str) -> bool:
        return (self._sample_dir(key) / "tokens.pt").is_file()

    def load(self, key: str, map_location="cpu") -> Optional[CachedFeatureBatch]:
        path = self._sample_dir(key) / "tokens.pt"
        if not path.is_file():
            return None
        data = torch.load(path, map_location=map_location)
        return CachedFeatureBatch(cond_tokens=data["cond_tokens"],
                                  target_tokens=data["target_tokens"],
                                  Hp=data["Hp"],
                                  Wp=data["Wp"])

    def save(self, key: str, batch: CachedFeatureBatch):
        if self.readonly:
            raise RuntimeError("Cache opened in read-only mode")
        sample_dir = self._sample_dir(key)
        sample_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "cond_tokens": batch.cond_tokens,
            "target_tokens": batch.target_tokens,
            "Hp": batch.Hp,
            "Wp": batch.Wp,
        }, sample_dir / "tokens.pt")
