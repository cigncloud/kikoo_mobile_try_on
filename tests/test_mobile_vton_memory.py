"""Check CFG branch alignment and actual model release independently of CUDA."""

import sys
import weakref
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mobile_vton_memory import PhaseModel, sequential_cfg_forward


def test_sequential_cfg_keeps_nested_conditioning_aligned():
    batches = []

    def forward(
        sample,
        timestep,
        encoder_hidden_states,
        garment_features,
        added_cond_kwargs,
        return_dict,
    ):
        batches.append(sample.shape[0])
        result = (
            sample * 2
            + timestep[:, None]
            + encoder_hidden_states
            + garment_features[0]
            + added_cond_kwargs["image_embeds"]
        )
        return (result,)

    values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    kwargs = {
        "sample": values,
        "timestep": torch.arange(4),
        "encoder_hidden_states": values.flip(0),
        "garment_features": [values * 3],
        "added_cond_kwargs": {"image_embeds": values * 5},
        "return_dict": False,
    }
    expected = forward(**kwargs)[0]
    actual = sequential_cfg_forward(forward, lambda: True)(**kwargs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert batches == [4, 2, 2]
    unconditional, conditional = actual.chunk(2)
    reference_unconditional, reference_conditional = expected.chunk(2)
    torch.testing.assert_close(
        unconditional + 2.5 * (conditional - unconditional),
        reference_unconditional
        + 2.5 * (reference_conditional - reference_unconditional),
        rtol=0,
        atol=0,
    )


def test_phase_model_releases_weights_and_reloads_for_next_image():
    references = []
    events = []

    def factory():
        model = torch.nn.Linear(2, 2, bias=False)
        references.append(weakref.ref(model))
        return model

    phase = PhaseModel("encoder", factory, 2, events, torch.device("cpu"))
    assert not references
    with torch.inference_mode():
        first = phase(torch.ones(1, 2))
        assert references[0]() is not None
        second = phase(torch.ones(1, 2))
    torch.testing.assert_close(first, second)
    assert references[0]() is None
    assert phase._active is None
    assert [event["action"] for event in events] == ["load", "call", "call", "release"]
    assert all(tensor.numel() == 0 for tensor in phase.state_dict().values())
    with torch.inference_mode():
        phase(torch.ones(1, 2))
    assert len(references) == 2
    assert references[1]() is not None


def test_loading_phase_models_preserves_sampling_rng():
    phase = PhaseModel(
        "decoder", lambda: torch.nn.Linear(2, 2), 1, [], torch.device("cpu")
    )
    torch.manual_seed(42)
    state = torch.random.get_rng_state()
    with torch.inference_mode():
        phase(torch.ones(1, 2))
    assert torch.equal(torch.random.get_rng_state(), state)


def test_sequential_cfg_rejects_incomplete_batch():
    forward = sequential_cfg_forward(lambda **kwargs: None, lambda: True)
    with pytest.raises(ValueError, match="equal unconditional"):
        forward(sample=torch.zeros(3, 2), return_dict=False)
