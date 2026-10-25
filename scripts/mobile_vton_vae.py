"""Load only the original VAE encoder and explicitly configure native tiling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, Encoder
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from safetensors import safe_open


class EncoderOnlyVAE(torch.nn.Module):
    """Keep the exact encoder operations/weights and Diffusers' tiling algorithm."""

    _encode = AutoencoderKL._encode
    _tiled_encode = AutoencoderKL._tiled_encode
    blend_v = AutoencoderKL.blend_v
    blend_h = AutoencoderKL.blend_h
    enable_tiling = AutoencoderKL.enable_tiling

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = SimpleNamespace(**config)
        self.encoder = Encoder(
            in_channels=config["in_channels"],
            out_channels=config["latent_channels"],
            down_block_types=config["down_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            act_fn=config["act_fn"],
            norm_num_groups=config["norm_num_groups"],
            double_z=True,
            mid_block_add_attention=config["mid_block_add_attention"],
        )
        channels = 2 * config["latent_channels"]
        self.quant_conv = (
            torch.nn.Conv2d(channels, channels, 1) if config["use_quant_conv"] else None
        )
        self.use_tiling = False
        self.use_slicing = False
        self.tile_sample_min_size = config["sample_size"]
        self.tile_latent_min_size = config["sample_size"] // (
            2 ** (len(config["block_out_channels"]) - 1)
        )
        self.tile_overlap_factor = 0.25

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> EncoderOnlyVAE:
        model = cls(json.loads((checkpoint / "config.json").read_text()))
        # Read only the encoder keys: the unused decoder is never instantiated
        # and its tensors are never copied out of the safetensors file.
        with safe_open(
            checkpoint / "diffusion_pytorch_model.safetensors",
            framework="pt",
            device="cpu",
        ) as handle:
            tensor_keys = handle.keys()
            state = {
                key: handle.get_tensor(key)
                for key in tensor_keys
                if key.startswith(("encoder.", "quant_conv."))
            }
        model.load_state_dict(state, strict=True)
        return model

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        moments = self._encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        return (
            AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)
        )


def configure_vae_tiling(
    vae: torch.nn.Module, tile_size: int, events: list[dict]
) -> None:
    """Record real tile forwards, not just whether enable_tiling was called."""
    stride = 2 ** (len(vae.config.block_out_channels) - 1)
    if tile_size:
        if tile_size < stride * 4 or tile_size % (stride * 4):
            raise ValueError("VAE tiles must be positive multiples of 4 * VAE stride")
        vae.tile_sample_min_size = tile_size
        vae.tile_latent_min_size = tile_size // stride
        vae.tile_overlap_factor = 0.25
        vae.enable_tiling()
    tile_shapes = []

    def record_tile(module, inputs):
        tile_shapes.append(list(inputs[0].shape))

    vae.encoder.register_forward_pre_hook(record_tile)
    original_encode = vae.encode

    def encode(x, *args, **kwargs):
        tile_shapes.clear()
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        result = original_encode(x, *args, **kwargs)
        posterior = result[0] if isinstance(result, tuple) else result.latent_dist
        event = {
            "input_shape": list(x.shape),
            "tile_shapes": tile_shapes.copy(),
            "encoder_forward_count": len(tile_shapes),
            "latent_mean_shape": list(posterior.mean.shape),
        }
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
            event["cuda_peak_allocated_so_far_bytes"] = torch.cuda.max_memory_allocated(
                x.device
            )
        events.append(event)
        return result

    vae.encode = encode
