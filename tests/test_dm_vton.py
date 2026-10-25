"""Check the exported cost volume against direct pixel/channel arithmetic."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_dm_vton_mnn import torch_correlation


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_correlation_channel_order_padding_and_subsampling(stride):
    generator = torch.Generator().manual_seed(42)
    first = torch.randn(2, 3, 5, 4, generator=generator)
    second = torch.randn(2, 3, 5, 4, generator=generator)
    actual = torch_correlation(tenFirst=first, tenSecond=second, intStride=stride)
    expected = torch.zeros(
        2, 49, (5 + stride - 1) // stride, (4 + stride - 1) // stride
    )
    for n in range(2):
        for y in range(0, 5, stride):
            for x in range(0, 4, stride):
                for channel in range(49):
                    other_y = y + (channel // 7 - 3) * stride
                    other_x = x + (channel % 7 - 3) * stride
                    if 0 <= other_y < 5 and 0 <= other_x < 4:
                        expected[n, channel, y // stride, x // stride] = (
                            sum(
                                first[n, c, y, x] * second[n, c, other_y, other_x]
                                for c in range(3)
                            )
                            / 3
                        )
    torch.testing.assert_close(actual, expected)
