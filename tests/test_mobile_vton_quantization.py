"""Verify packed INT4 execution, storage accounting, and safe state reload."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mobile_vton_quantization import (
    load_quantized_state,
    quantize_linear_weights,
    save_quantized_state,
    storage_bytes,
)


def test_int4_storage_and_roundtrip(tmp_path):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(256, 256, dtype=torch.bfloat16),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256, dtype=torch.bfloat16),
        )
        .eval()
        .requires_grad_(False)
    )
    inputs = torch.randn(1, 16, 256, dtype=torch.bfloat16)
    reference = model(inputs)
    info = quantize_linear_weights(model, torch.device("cpu"))
    assert info["linear_layers_quantized"] == 2
    assert info["storage_bytes_after"] < info["storage_bytes_before"] * 0.3
    output = model(inputs)
    assert output.shape == reference.shape
    assert torch.isfinite(output).all()
    # Quantization must actually affect the arithmetic, not just checkpoint metadata.
    assert not torch.equal(output, reference)
    path = tmp_path / "int4.pt"
    saved = save_quantized_state(model, path)
    assert saved["bytes"] == path.stat().st_size
    restored = (
        torch.nn.Sequential(
            torch.nn.Linear(256, 256, dtype=torch.bfloat16),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256, dtype=torch.bfloat16),
        )
        .eval()
        .requires_grad_(False)
    )
    loaded = load_quantized_state(restored, path, torch.device("cpu"))
    assert loaded["linear_layers_quantized"] == 2
    assert storage_bytes(restored) == storage_bytes(model)
    torch.testing.assert_close(restored(inputs), output, rtol=0, atol=0)


def test_incompatible_linear_is_reported():
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(256, 256, bias=False, dtype=torch.bfloat16),
            torch.nn.Linear(256, 24, bias=False, dtype=torch.bfloat16),
            torch.nn.Linear(24, 8, bias=False, dtype=torch.bfloat16),
        )
        .eval()
        .requires_grad_(False)
    )
    info = quantize_linear_weights(model, torch.device("cpu"))
    assert info["linear_layers_total"] == 3
    assert info["linear_layers_quantized"] == 1
    assert info["skipped_linear_layers"] == {"1": 256 * 24, "2": 24 * 8}
