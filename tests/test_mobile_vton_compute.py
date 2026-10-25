"""Check native GQA against the pinned processor, including normalization/RoPE."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models/mobile_vton/upstream"))
from Mobile_VTON.models.attention_processor import Attention
from mobile_vton_compute import install_native_gqa


@pytest.mark.parametrize("kv_heads", (1, 2, 4))
@pytest.mark.parametrize("cross_attention", (False, True))
def test_native_gqa_preserves_normalized_rotary_attention(
    monkeypatch, kv_heads, cross_attention
):
    torch.manual_seed(42)
    attn = Attention(
        query_dim=32,
        heads=4,
        kv_heads=kv_heads,
        dim_head=8,
        cross_attention_dim=24 if cross_attention else None,
        qk_norm="layer_norm",
        residual_connection=True,
    ).eval()
    sample = torch.randn(2, 8, 32)
    context = torch.randn(2, 6, 24) if cross_attention else None
    angles = torch.arange(4).float()[:, None].expand(4, 8) / 10
    kwargs = {
        "encoder_hidden_states": context,
        "image_rotary_emb": (angles.cos(), angles.sin()),
    }
    with torch.inference_mode():
        expected = attn(sample, **kwargs)
    assert install_native_gqa(attn) == [""]
    calls = []
    original = F.scaled_dot_product_attention

    def checked(query, key, value, **options):
        calls.append(
            (query.shape[1], key.shape[1], value.shape[1], options["enable_gqa"])
        )
        return original(query, key, value, **options)

    monkeypatch.setattr(F, "scaled_dot_product_attention", checked)
    with torch.inference_mode():
        actual = attn(sample, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert calls == [(4, kv_heads, kv_heads, kv_heads != 4)]
