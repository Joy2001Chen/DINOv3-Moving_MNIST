# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Callable, Tuple, Union

from torch import Tensor, nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Adapt pre-trained weights when switching channel count (e.g., RGB → grayscale).
        def _convert_pretrained_weights(*hook_args):
            """
            Handle both legacy (PyTorch <1.13) and newer hook signatures.
            Legacy: (state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys)
            Newer:  (+ error_msgs) and optionally prepends module when with_module=True.
            """
            if len(hook_args) == 7:
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs = hook_args
                module = self
            elif len(hook_args) == 6:
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys = hook_args
                error_msgs = []
                module = self
            elif len(hook_args) == 8:
                module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs = hook_args
            else:
                raise RuntimeError(f"Unexpected pre-hook signature with {len(hook_args)} arguments.")

            # Older PyTorch versions do not pass error_msgs; normalise to a list so we can append.
            if error_msgs is None:
                error_msgs = []
            weight_key = prefix + "proj.weight"
            if weight_key not in state_dict:
                return
            weight = state_dict[weight_key]
            in_channels_loaded = weight.shape[1]
            if in_channels_loaded == module.in_chans:
                return  # already matches
            if module.in_chans == 1 and in_channels_loaded == 3:
                # average RGB filters to a single channel
                state_dict[weight_key] = weight.mean(dim=1, keepdim=True)
            elif module.in_chans == 3 and in_channels_loaded == 1:
                # replicate grayscale filters to RGB (unlikely but safe)
                state_dict[weight_key] = weight.repeat(1, 3, 1, 1)
            else:
                error_msgs.append(
                    f"{weight_key} has incompatible channel dims: "
                    f"loaded={in_channels_loaded}, expected={module.in_chans}"
                )

        self._register_load_state_dict_pre_hook(_convert_pretrained_weights)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
