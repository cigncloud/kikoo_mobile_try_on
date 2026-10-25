"""Validate boundary blending, posterior sampling, schedule and native FP16 IO."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_pixel4_generation import blend_tiles, posterior_sample


def test_boundary_blending_matches_diffusers():
    class FakeEncoder:
        tile_sample_min_size = 512
        tile_latent_min_size = 64
        tile_overlap_factor = 0.25
        blend_v = AutoencoderKL.blend_v
        blend_h = AutoencoderKL.blend_h

        class config:
            use_quant_conv = False

        def encoder(self, image):
            # Tile-specific offsets exercise overlap blending, including corners.
            return torch.nn.functional.avg_pool2d(image, 8) + image.mean()

    torch.set_num_threads(2)
    torch.manual_seed(7)
    image = torch.randn(1, 2, 1024, 768)
    encoder = FakeEncoder()
    expected = AutoencoderKL._tiled_encode(encoder, image).numpy()
    tiles = [
        [
            encoder.encoder(image[:, :, y : y + 512, x : x + 512]).numpy().copy()
            for x in (0, 384)
        ]
        for y in (0, 384, 768)
    ]
    actual = blend_tiles(tiles)
    assert actual.shape == (1, 2, 128, 96)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_posterior_sampling_clamps_log_variance():
    params = np.array([1, 2, -100, 100], dtype=np.float32).reshape(1, 4, 1, 1)
    epsilon = np.array([-1, 1], dtype=np.float32).reshape(1, 2, 1, 1)
    distribution = DiagonalGaussianDistribution(torch.from_numpy(params))
    expected = distribution.mean + distribution.std * torch.from_numpy(epsilon)
    np.testing.assert_allclose(posterior_sample(params, epsilon), expected.numpy())


def test_float32_cfg_euler_matches_diffusers_all_steps():
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    scheduler.set_timesteps(28)
    generator = np.random.default_rng(42)
    actual = generator.normal(size=(1, 16, 8, 6)).astype(np.float32)
    expected = torch.from_numpy(actual.copy())
    for step, timestep in enumerate(scheduler.timesteps):
        negative = generator.normal(size=actual.shape).astype(np.float32)
        positive = generator.normal(size=actual.shape).astype(np.float32)
        guided = negative + np.float32(2.5) * (positive - negative)
        delta = np.float32(scheduler.sigmas[step + 1]) - np.float32(
            scheduler.sigmas[step]
        )
        actual = actual + delta * guided
        expected = scheduler.step(
            torch.from_numpy(guided), timestep, expected, return_dict=False
        )[0]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=0)
    assert scheduler.sigmas[-1] == 0


def test_native_half_reader_all_bit_patterns(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Native FP16 reader validation requires a C++ compiler")
    source = (ROOT / "experiments/2026-10-24-pixel4/mnn_generation.cpp").read_text()
    function = source.split("float halfToFloat(", 1)[1].split("double elapsed(", 1)[0]
    program = tmp_path / "half.cpp"
    program.write_text(
        "#include <cstdint>\n#include <cstring>\n#include <cstdio>\nfloat halfToFloat("
        + function
        + "int main() { for (uint32_t i=0;i<65536;++i) { float f=halfToFloat(i); fwrite(&f,4,1,stdout); } }\n"
    )
    executable = tmp_path / "half"
    subprocess.run(
        [compiler, "-std=c++17", str(program), "-o", str(executable)], check=True
    )
    actual = np.frombuffer(subprocess.check_output([str(executable)]), np.float32)
    expected = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[mask].view(np.uint32), expected[mask].view(np.uint32)
    )
