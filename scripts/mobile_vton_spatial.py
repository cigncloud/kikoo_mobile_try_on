"""Tile local convolutions while keeping every GroupNorm on the full image."""

from __future__ import annotations

import torch


def _check_conv(conv: torch.nn.Conv2d) -> None:
    if (
        conv.stride != (1, 1)
        or conv.dilation != (1, 1)
        or conv.padding_mode != "zeros"
        or not isinstance(conv.padding, tuple)
        or any(2 * p != k - 1 for p, k in zip(conv.padding, conv.kernel_size))
    ):
        raise ValueError(
            "Spatial tiling requires odd, same-size, stride-one convolutions"
        )


class RowTiledConv(torch.nn.Module):
    """Compute convolution rows with exact input halos and no blending."""

    def __init__(self, conv: torch.nn.Conv2d, rows: int, minimum_bytes: int) -> None:
        super().__init__()
        _check_conv(conv)
        if rows <= 0:
            raise ValueError("Tile rows must be positive")
        self.core = conv
        self.rows = rows
        self.minimum_bytes = minimum_bytes

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.numel() * sample.element_size() < self.minimum_bytes:
            return self.core(sample)
        if torch.is_grad_enabled():
            raise ValueError("Spatial tiling is inference-only")
        height = sample.shape[-2]
        output = sample.new_empty(
            (sample.shape[0], self.core.out_channels, height, sample.shape[-1])
        )
        halo = self.core.padding[0]
        for start in range(0, height, self.rows):
            end = min(start + self.rows, height)
            first, last = max(0, start - halo), min(height, end + halo)
            tile = self.core(sample[:, :, first:last, :])
            output[:, :, start:end, :].copy_(tile[:, :, start - first : end - first, :])
            del tile
        return output


class RowTiledSeparable(torch.nn.Module):
    """Fuse each depthwise/pointwise tile to avoid a full depthwise intermediate."""

    def __init__(self, conv: torch.nn.Module, rows: int, minimum_bytes: int) -> None:
        super().__init__()
        _check_conv(conv.dw_conv)
        _check_conv(conv.pw_conv)
        if conv.pw_conv.kernel_size != (1, 1) or rows <= 0:
            raise ValueError("Separable tiling requires a pointwise second convolution")
        self.core = conv
        self.rows = rows
        self.minimum_bytes = minimum_bytes

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.numel() * sample.element_size() < self.minimum_bytes:
            return self.core(sample)
        if torch.is_grad_enabled():
            raise ValueError("Spatial tiling is inference-only")
        height = sample.shape[-2]
        output = sample.new_empty(
            (sample.shape[0], self.core.pw_conv.out_channels, height, sample.shape[-1])
        )
        halo = self.core.dw_conv.padding[0]
        for start in range(0, height, self.rows):
            end = min(start + self.rows, height)
            first, last = max(0, start - halo), min(height, end + halo)
            depthwise = self.core.dw_conv(sample[:, :, first:last, :])
            tile = self.core.pw_conv(depthwise[:, :, start - first : end - first, :])
            output[:, :, start:end, :].copy_(tile)
            del depthwise, tile
        return output


def install_spatial_tiling(
    module: torch.nn.Module,
    rows: int,
    *,
    separable: bool,
    minimum_bytes: int = 32 * 1024**2,
) -> list[str]:
    """Replace only local operators, never normalization or entire VAE blocks."""
    from Mobile_VTON.models.convolutions import SepConv2d

    changed = []

    def visit(parent, prefix):
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            replacement = None
            if separable and isinstance(child, SepConv2d):
                try:
                    _check_conv(child.dw_conv)
                    _check_conv(child.pw_conv)
                except ValueError:
                    continue
                replacement = RowTiledSeparable(child, rows, minimum_bytes)
            elif not separable and isinstance(child, torch.nn.Conv2d):
                try:
                    _check_conv(child)
                except ValueError:
                    continue
                replacement = RowTiledConv(child, rows, minimum_bytes)
            if replacement is not None:
                setattr(parent, name, replacement)
                changed.append(qualified)
            else:
                visit(child, qualified)

    visit(module, "")
    if not changed:
        raise ValueError("No compatible spatial convolution modules found")
    return changed
