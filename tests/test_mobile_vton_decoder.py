"""Check decoder precision boundaries, matching latents and failure reporting."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mobile_vton_decoder import PrecisionDecoder


def test_decoder_cast_keeps_original_latents_and_float_output(tmp_path):
    torch.manual_seed(42)
    decoder = torch.nn.Conv2d(2, 3, 1).half().eval().requires_grad_(False)
    inputs = torch.randn(1, 2, 4, 4)
    copy = inputs.clone()
    events = []
    path = tmp_path / "latents.pt"
    adapter = PrecisionDecoder(decoder, events, path)
    with torch.inference_mode():
        expected = decoder(inputs.half()).float()
        actual = adapter(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs, copy, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.load(path, weights_only=True), inputs, rtol=0, atol=0
    )
    assert actual.dtype == torch.float32
    assert events[0]["model_dtype"] == "torch.float16"
    assert events[0]["finite"]
    events32 = []
    with torch.inference_mode():
        PrecisionDecoder(decoder.float(), events32)(inputs)
    assert events32[0]["latent_sha256_fp32"] == events[0]["latent_sha256_fp32"]


def test_decoder_rejects_nonfinite_output():
    decoder = torch.nn.Conv2d(2, 3, 1)
    events = []
    with torch.no_grad():
        decoder.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="NaN or Inf"):
        PrecisionDecoder(decoder, events)(torch.ones(1, 2, 2, 2))
    assert not events[0]["finite"]


def test_decoder_can_disable_cpu_latent_auditing(monkeypatch):
    def no_cpu(*args, **kwargs):
        raise AssertionError("Unexpected latent copy to CPU")

    decoder = torch.nn.Conv2d(2, 3, 1).eval()
    events = []
    adapter = PrecisionDecoder(decoder, events, audit_latents=False)
    monkeypatch.setattr(torch.Tensor, "cpu", no_cpu)
    with torch.inference_mode():
        adapter(torch.ones(1, 2, 4, 4))
    assert events[0]["latent_sha256_fp32"] is None
