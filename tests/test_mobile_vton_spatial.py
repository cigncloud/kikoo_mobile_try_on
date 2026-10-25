"""Check convolution halos and full-image normalization across tile seams."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models/mobile_vton/upstream"))
from mobile_vton_spatial import RowTiledConv, RowTiledSeparable, install_spatial_tiling


@pytest.mark.parametrize("kernel,groups", ((1, 1), (3, 1), (3, 4)))
def test_row_tiles_match_full_convolution_including_image_borders(kernel, groups):
    torch.manual_seed(7)
    conv = torch.nn.Conv2d(4, 4, kernel, padding=kernel // 2, groups=groups)
    sample = torch.randn(2, 4, 13, 11)
    with torch.inference_mode():
        expected = conv(sample)
        actual = RowTiledConv(conv, rows=4, minimum_bytes=0)(sample)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_separable_tiles_keep_global_groupnorm_and_residual_input():
    torch.manual_seed(7)
    dw = torch.nn.Conv2d(4, 4, 3, padding=1, groups=4)
    pw = torch.nn.Conv2d(4, 8, 1)

    class Separable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dw_conv = dw
            self.pw_conv = pw

        def forward(self, sample):
            return self.pw_conv(self.dw_conv(sample))

    conv = Separable()
    norm = torch.nn.GroupNorm(2, 8)
    sample = torch.randn(2, 4, 13, 11)
    saved = sample.clone()
    with torch.inference_mode():
        expected = norm(conv(sample))
        actual = norm(RowTiledSeparable(conv, rows=4, minimum_bytes=0)(sample))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sample, saved, rtol=0, atol=0)


def test_spatial_tiling_rejects_strided_convolution():
    with pytest.raises(ValueError, match="stride-one"):
        RowTiledConv(torch.nn.Conv2d(4, 4, 3, stride=2, padding=1), 4, 0)


def test_install_keeps_downsampling_untouched():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 4, 3, stride=2, padding=1),
        torch.nn.Conv2d(4, 4, 3, padding=1),
    )
    original = model[0]
    assert install_spatial_tiling(model, 4, separable=False) == ["1"]
    assert model[0] is original
    assert isinstance(model[1], RowTiledConv)
