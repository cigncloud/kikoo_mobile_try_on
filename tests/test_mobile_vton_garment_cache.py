"""Check cached distributions, input identity and pipeline cache substitutions."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mobile_vton_garment_cache import install_garment_cache, load_garment_cache
from mobile_vton_lifetimes import release_working_tensors


def test_cached_posterior_preserves_samples_and_rng(tmp_path):
    params = torch.randn(1, 32, 4, 4).bfloat16()
    cache = {
        "identity": {"height": 32, "width": 32, "guidance_scale": 2.5},
        "posterior_parameters": params,
        "projected_image_embeds": torch.randn(2, 3, 8).bfloat16(),
    }
    path = tmp_path / "cache.pt"
    torch.save(cache, path)
    loaded = load_garment_cache(path, cache["identity"], torch.device("cpu"))
    pipe = SimpleNamespace()
    install_garment_cache(pipe, loaded)
    reference = DiagonalGaussianDistribution(params)
    torch.manual_seed(42)
    expected = reference.sample()
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    actual = pipe._garment_posterior.sample()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="identity mismatch"):
        load_garment_cache(
            path,
            {**cache["identity"], "garment_sha256": "changed"},
            torch.device("cpu"),
        )


class CachePipeline:
    do_classifier_free_guidance = True

    def _record_working_tensors(self, *args):
        pass

    def __call__(self, cloth, ip_adapter_image):
        gen_condition = torch.ones(1, 16, 4, 4)
        prompt_embeds = pooled_prompt_embeds = torch.ones(1)
        cloth = self.vae.encode(cloth).latent_dist.sample()
        if ip_adapter_image is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(ip_adapter_image)
        latents = torch.ones_like(cloth)
        for i in range(1):
            latent_model_input = gen_condition
            garment_features = [cloth]
            added_cond_kwargs = {"image_embeds": image_embeds}
            timestep = timestep_garment = torch.ones(1)
            _ = torch.ones(1)
            noise_pred = (
                garment_features[0]
                + latent_model_input
                + added_cond_kwargs["image_embeds"].mean()
                + prompt_embeds
                + pooled_prompt_embeds
                + timestep
                + timestep_garment
            )
            noise_pred_uncond, noise_pred_text = noise_pred, noise_pred
            noise_pred = noise_pred_uncond + 2.5 * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, i, latents, return_dict=False)[0]
        image = self.vae_decoder(latents)
        return image


def test_cache_substitution_skips_both_garment_encoders_without_changing_result():
    posterior = DiagonalGaussianDistribution(torch.zeros(1, 32, 4, 4))
    image_embeds = torch.ones(2, 3, 8)
    calls = []
    original = CachePipeline()
    original.vae = SimpleNamespace(
        encode=lambda x: calls.append("vae") or SimpleNamespace(latent_dist=posterior)
    )
    original.prepare_ip_adapter_image_embeds = (
        lambda x: calls.append("dino") or image_embeds
    )
    original.scheduler = SimpleNamespace(step=lambda noise, *args, **kwargs: (noise,))
    original.vae_decoder = lambda x: x * 2
    adapted = type(
        "Cached",
        (CachePipeline,),
        {
            "__call__": release_working_tensors(
                CachePipeline.__call__, garment_cached=True
            )
        },
    )()
    adapted.scheduler = original.scheduler
    adapted.vae_decoder = original.vae_decoder
    adapted._garment_posterior = posterior
    adapted._garment_image_embeds = image_embeds
    torch.manual_seed(42)
    expected = original(torch.ones(1), torch.ones(1))
    torch.manual_seed(42)
    actual = adapted(torch.empty(0), None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == ["vae", "dino"]
