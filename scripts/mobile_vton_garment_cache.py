"""GPU garment precomputation with revision/conditioning validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cache_identity(config: dict, garment_hash: str, text_hash: str, root: Path) -> dict:
    """Bind preprocessing and projected embeddings to exact checkpoints."""
    keys = (
        "upstream_revision",
        "model_revision",
        "width",
        "height",
        "dtype",
        "guidance_scale",
        "garment_description",
        "quantization",
        "group_size",
        "vae_encoder_only",
        "vae_tile_size",
    )
    result = {key: config.get(key) for key in keys}
    result.update(
        schema=1,
        garment_sha256=garment_hash,
        text_cache_sha256=text_hash,
        posterior="unscaled Gaussian parameters; sample online to preserve RNG",
    )
    checkpoint = Path(config["checkpoint"])
    files = {
        "vae": checkpoint / "vae/diffusion_pytorch_model.safetensors",
        "dino": checkpoint / "image_encoder/model.safetensors",
        "processor": checkpoint / "image_encoder/preprocessor_config.json",
        "projector": root / config["quantized_state_dir"] / "denoiser_int4.pt",
    }
    result["checkpoint_sha256"] = {name: digest(path) for name, path in files.items()}
    return result


def load_garment_cache(path: Path, expected: dict, device: torch.device) -> dict:
    cache = torch.load(path, map_location=device, weights_only=True)
    if cache.get("identity") != expected:
        raise ValueError(
            "Garment cache preprocessing/checkpoint/input identity mismatch"
        )
    params, image = cache["posterior_parameters"], cache["projected_image_embeds"]
    if params.shape != (1, 32, expected["height"] // 8, expected["width"] // 8):
        raise ValueError("Garment cache latent dimensions mismatch")
    if params.dtype != torch.bfloat16 or image.dtype != torch.bfloat16:
        raise ValueError("Garment cache requires the recorded BF16 representation")
    if image.shape[0] != (2 if expected["guidance_scale"] > 1 else 1):
        raise ValueError("Garment cache CFG dimensions mismatch")
    if not torch.isfinite(params).all() or not torch.isfinite(image).all():
        raise ValueError("Garment cache contains nonfinite values")
    return cache


def install_garment_cache(pipe, cache: dict) -> dict:
    """Keep the chosen garment's cached tensors on the GPU for all requests."""
    params = cache["posterior_parameters"]
    pipe._garment_posterior = DiagonalGaussianDistribution(params)
    pipe._garment_image_embeds = cache["projected_image_embeds"]
    tensors = {
        "posterior_parameters": params,
        "posterior_mean": pipe._garment_posterior.mean,
        "posterior_std": pipe._garment_posterior.std,
        "projected_image_embeds": pipe._garment_image_embeds,
    }
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "logical_bytes": value.numel() * value.element_size(),
        }
        for name, value in tensors.items()
    }


def build_garment_cache(
    pipe, cloth, ip_image, identity: dict, path: Path, guidance_scale: float
) -> dict:
    """Encode on CUDA once; artifact serialization is outside online timing."""
    if cloth.device.type != "cuda" or ip_image.device.type != "cuda":
        raise ValueError("Garment precomputation requires CUDA inputs")
    pipe._guidance_scale = guidance_scale
    with torch.inference_mode():
        parameters = pipe.vae.encode(cloth).latent_dist.parameters
        image = pipe.prepare_ip_adapter_image_embeds(ip_image, cloth.device, 1)
        projected = pipe.denoiser.encoder_hid_proj(image).to(dtype=cloth.dtype)
    artifact = {
        "identity": identity,
        "posterior_parameters": parameters.cpu(),
        "projected_image_embeds": projected.cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    torch.save(artifact, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "tensor_bytes": sum(
            v.numel() * v.element_size()
            for v in artifact.values()
            if isinstance(v, torch.Tensor)
        ),
    }
