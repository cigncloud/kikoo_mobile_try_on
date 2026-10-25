"""Verify lifetime changes preserve guidance and actually release tensor refs."""

import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mobile_vton_lifetimes import release_working_tensors


class ExamplePipeline:
    def __init__(self, guidance):
        self.do_classifier_free_guidance = guidance
        self.scheduler = SimpleNamespace(
            step=lambda noise, *args, **kwargs: (noise + 1,)
        )
        self.vae_decoder = lambda x: x * 2
        self.refs = []
        self.condition_refs = []
        self.recorded = []

    def _record_working_tensors(self, stage, step):
        self.recorded.append(
            (
                stage,
                step,
                sum(ref() is not None for ref in self.refs),
                sum(ref() is not None for ref in self.condition_refs),
            )
        )

    def __call__(self):
        latents = torch.ones(1, 2)
        gen_condition = torch.ones(2, 2)
        cloth = torch.ones(1, 2)
        image_embeds = torch.ones(1, 2)
        prompt_embeds = torch.ones(2, 2)
        pooled_prompt_embeds = torch.ones(2, 2)
        self.condition_refs = [
            weakref.ref(tensor)
            for tensor in (
                gen_condition,
                cloth,
                image_embeds,
                prompt_embeds,
                pooled_prompt_embeds,
            )
        ]
        for i in range(2):
            latent_model_input = gen_condition + pooled_prompt_embeds
            garment_features = [cloth + prompt_embeds[:1]]
            _ = torch.ones(1, 2)
            added_cond_kwargs = {"image_embeds": image_embeds}
            timestep = torch.ones(2)
            timestep_garment = torch.ones(1)
            noise_pred = (
                latent_model_input
                + garment_features[0]
                + added_cond_kwargs["image_embeds"]
                + timestep[:, None]
                + timestep_garment[:, None]
            )
            self.refs += [
                weakref.ref(garment_features[0]),
                weakref.ref(latent_model_input),
            ]
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 2.5 * (
                    noise_pred_text - noise_pred_uncond
                )
            latents = self.scheduler.step(noise_pred, i, latents, return_dict=False)[0]
        image = self.vae_decoder(latents)
        return image


@pytest.mark.parametrize("guidance", (False, True))
def test_release_keeps_results_and_releases_garment_tensors(guidance):
    reference = ExamplePipeline(guidance)
    optimized_type = type(
        "Optimized",
        (ExamplePipeline,),
        {
            "__call__": release_working_tensors(ExamplePipeline.__call__),
        },
    )
    optimized = optimized_type(guidance)
    with torch.inference_mode():
        expected = reference()
        actual = optimized()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    after_events = [
        event for event in optimized.recorded if event[0] == "step_after_release"
    ]
    assert len(after_events) == 2
    assert all(event[2] == 0 for event in after_events)
    decode_events = [
        event for event in optimized.recorded if event[0].startswith("decode_")
    ]
    assert [event[3] for event in decode_events] == [5, 0]


def test_release_rejects_changed_pipeline_structure():
    with pytest.raises(ValueError, match="Unexpected upstream"):
        release_working_tensors(test_release_rejects_changed_pipeline_structure)
