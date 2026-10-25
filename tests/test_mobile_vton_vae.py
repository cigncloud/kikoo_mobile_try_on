"""Check exact encoder extraction and real overlapping tiling on a small VAE."""

import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mobile_vton_vae import EncoderOnlyVAE, configure_vae_tiling


def test_encoder_extraction_and_tiling_match_original(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    reference = (
        AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(32, 32),
            layers_per_block=1,
            sample_size=64,
            use_quant_conv=False,
            use_post_quant_conv=False,
        )
        .eval()
        .requires_grad_(False)
    )
    reference.save_pretrained(tmp_path)
    extracted = EncoderOnlyVAE.from_checkpoint(tmp_path).eval().requires_grad_(False)
    assert not hasattr(extracted, "decoder")
    assert extracted.state_dict().keys() == {
        key for key in reference.state_dict() if key.startswith("encoder.")
    }
    image = torch.randn(1, 3, 64, 48)
    with torch.inference_mode():
        expected = reference.encode(image).latent_dist.parameters
        actual = extracted.encode(image).latent_dist.parameters
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    ref_events, actual_events = [], []
    configure_vae_tiling(reference, 32, ref_events)
    configure_vae_tiling(extracted, 32, actual_events)
    with torch.inference_mode():
        tiled_reference = reference.encode(image).latent_dist.parameters
        tiled_extracted = extracted.encode(image).latent_dist.parameters
    torch.testing.assert_close(tiled_extracted, tiled_reference, rtol=0, atol=0)
    assert tiled_extracted.shape == actual.shape
    assert not torch.equal(tiled_extracted, actual)
    assert actual_events == ref_events
    assert actual_events[0]["encoder_forward_count"] == 6
    assert all(max(shape[-2:]) <= 32 for shape in actual_events[0]["tile_shapes"])
