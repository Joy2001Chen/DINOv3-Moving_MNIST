# models/backbones/dino_v3.py
import math, os, glob
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 若 dinov3 仓库不在 sys.path，会 ImportError；这时给出更清晰提示
try:
    from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16
except ImportError as e:
    raise ImportError(
        "Cannot import 'dinov3'. Please ensure the DINOv3 repo is cloned and in PYTHONPATH.\n"
        "Example:\n"
        "  git clone https://github.com/facebookresearch/dinov3.git ~/dinov3\n"
        "  export PYTHONPATH=$PYTHONPATH:~/dinov3\n"
        f"Original error: {e}"
    )

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_ARCH2CTOR = {
    "vits16": dinov3_vits16,
    "vitb16": dinov3_vitb16,
}

# DINOv3 常用 hidden 维度
_ARCH2DIM = {
    "vits16": 384,
    "vitb16": 768,
}

# 在本地优先查找的候选文件名片段（可按需扩展）
_CANDIDATE_NAMES = {
    "vits16": [
        "dinov3_vits16_pretrain",          # 通用前缀
        "dinov3_vits16_pretrain_lvd",      # 你的文件名中常见片段
        "vits16"                            # 兜底片段
    ],
    "vitb16": [
        "dinov3_vitb16_pretrain",
        "vitb16"
    ],
}

def _resolve_weights_path(arch: str, given: Optional[str]) -> str:
    """
    解析权重路径：优先使用显式给定的路径；否则自动在本地常见位置搜寻。
    搜索顺序：
      1) 显式参数 given
      2) 环境变量 DINO3_WEIGHTS
      3) ./checkpoints/ 下按常见命名匹配 *.pth
      4) 当前目录及子目录的 ./checkpoints/**/*.pth
    找到多个时按修改时间(新->旧)排序取最新。
    """
    # 1) 显式路径
    if given:
        if os.path.isfile(given):
            return given
        # 若是目录则在其中匹配
        if os.path.isdir(given):
            found = _search_in_dir(given, arch)
            if found:
                return found
        raise FileNotFoundError(f"weights not found: {given}")

    # 2) 环境变量
    env_path = os.environ.get("DINO3_WEIGHTS", "")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        if os.path.isdir(env_path):
            found = _search_in_dir(env_path, arch)
            if found:
                return found
        raise FileNotFoundError(f"DINO3_WEIGHTS set but not a valid file/dir: {env_path}")

    # 3) ./checkpoints/
    if os.path.isdir("./checkpoints"):
        found = _search_in_dir("./checkpoints", arch)
        if found:
            return found

    # 4) 递归兜底搜索（只在 ./checkpoints/** 下）
    matches = []
    for pat in ["./checkpoints/**/*.pth", "./checkpoints/*.pth"]:
        matches += glob.glob(pat, recursive=True)
    matches = _filter_by_arch(matches, arch)
    if matches:
        # 选最近修改的
        matches.sort(key=lambda p: os.stat(p).st_mtime, reverse=True)
        return matches[0]

    raise FileNotFoundError(
        f"Cannot find local DINOv3 weights for arch={arch}.\n"
        f"Tried: given path, $DINO3_WEIGHTS, ./checkpoints, ./checkpoints/**.\n"
        f"Place your .pth under ./checkpoints/ (e.g., dinov3_{arch}_pretrain*.pth) or set --dino3-weights /path/to/file.pth"
    )

def _filter_by_arch(paths, arch: str):
    """根据文件名包含片段筛选匹配 arch 的 .pth"""
    cand = _CANDIDATE_NAMES.get(arch, [])
    out = []
    for p in paths:
        name = os.path.basename(p).lower()
        if any(k in name for k in cand):
            out.append(p)
    return out

def _search_in_dir(d: str, arch: str) -> Optional[str]:
    """在指定目录中按候选片段匹配 *.pth，返回最近修改的那个"""
    if not os.path.isdir(d):
        return None
    globs = [os.path.join(d, "*.pth"), os.path.join(d, "**/*.pth")]
    matches = []
    for pat in globs:
        matches += glob.glob(pat, recursive=True)
    matches = _filter_by_arch(matches, arch)
    if not matches:
        return None
    matches.sort(key=lambda p: os.stat(p).st_mtime, reverse=True)
    return matches[0]

class DinoV3Features(nn.Module):
    """
    DINOv3 ViT 特征抽取（从本地 .pth 加载，完全离线）。
    - 输入:  (B, T, 1, 64, 64) in [0,1]
    - 输出:  patch tokens (B, T, N, C) 以及 CLS (B, T, C)
    """
    def __init__(self, arch: str, weights_path: str = "", trainable: bool = False):
        super().__init__()
        arch = arch.lower()
        assert arch in _ARCH2CTOR, f"Unsupported arch: {arch} (use vits16 or vitb16)"

        # —— 关键：优先解析本地权重路径 —— #
        weights_path = _resolve_weights_path(arch, weights_path)

        # 构造网络并从本地权重加载（绝不联网）
        self.model = _ARCH2CTOR[arch](weights=weights_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = trainable

        self.token_dim = _ARCH2DIM[arch]
        self.image_size = 224

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

    @torch.no_grad()
    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        # (B,T,1,64,64) -> (B*T,3,224,224) with ImageNet norm
        B, T, _, H, W = frames.shape
        x = frames.repeat(1, 1, 3, 1, 1).reshape(B*T, 3, H, W)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def forward_patches(self, frames: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        返回:
            tokens: (B, T, N, C)
            Hp, Wp: patch 网格大小 (例如 14x14 或 16x16，取决于模型)
        """
        x = self._preprocess(frames)             # (B*T,3,224,224)
        out = self.model.forward_features(x)     # repo 返回 dict
        #tokens = out.get("x_norm_patchtokens", None) or out.get("x_patchtokens", None)
        tokens = out.get("x_norm_patchtokens")
        if tokens is None:
            tokens = out.get("x_patchtokens")

        if tokens is None:
            raise KeyError(
                "Neither 'x_norm_patchtokens' nor 'x_patchtokens' found in backbone output keys: "
                f"{list(out.keys())}"
            )

        assert tokens is not None, "Cannot find patch tokens in DINOv3 forward_features output."
        BT, N, C = tokens.shape
        Hp = int(math.sqrt(N)); Wp = Hp
        return tokens.view(frames.size(0), frames.size(1), N, C), Hp, Wp

    @torch.no_grad()
    def forward_cls(self, frames: torch.Tensor) -> torch.Tensor:
        """
        返回每帧的 CLS: (B, T, C)
        """
        x = self._preprocess(frames)
        out = self.model.forward_features(x)
        cls = out.get("x_norm_clstoken", None) or out.get("x_clstoken", None)
        assert cls is not None, "Cannot find CLS token in DINOv3 forward_features output."
        B, T = frames.size(0), frames.size(1)
        return cls.view(B, T, -1)
