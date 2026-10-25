"""Export original Mobile-VTON components for an Android runtime smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=["decoder", "vae_encoder", "vae_encoder_portable", "garment", "tryon"],
        default="decoder",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/tmp/mobile_tryon/weights/mobile_vton/checkpoint"),
    )
    parser.add_argument(
        "--latents",
        type=Path,
        default=ROOT
        / "experiments/2026-10-23-mobile_vton/outputs_compute_decoder32_cfg/decoder_latents.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/2026-10-24-pixel4/exports",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / "models/mobile_vton/upstream"))
    if args.component == "decoder":
        from Mobile_VTON.models.autoencoders.vae import Decoder

        weight_path = args.checkpoint / "vae_decoder/decoder.pt"
        config = json.loads((args.checkpoint / "vae_decoder/decoder.json").read_text())
        model = Decoder(**config).eval().requires_grad_(False)
        model.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True)
        )
        latent = (
            torch.load(args.latents, map_location="cpu", weights_only=True)
            .float()
            .contiguous()
        )
        assert tuple(latent.shape) == (1, 16, 128, 96)
        inputs, input_names, source = (latent,), ["latent"], str(args.latents)
    elif args.component.startswith("vae_encoder"):
        from mobile_vton_vae import EncoderOnlyVAE
        from PIL import Image

        class RawEncoder(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, image):
                return self.encoder._encode(image)

        weight_path = args.checkpoint / "vae/diffusion_pytorch_model.safetensors"
        model = RawEncoder(EncoderOnlyVAE.from_checkpoint(args.checkpoint / "vae"))
        image_path = ROOT / "experiments/2026-10-23-mobile_vton/inputs/person.jpg"
        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize((768, 1024), Image.Resampling.LANCZOS)
        )
        image = (
            torch.from_numpy(np.array(image).copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 127.5
            - 1
        )
        inputs, input_names, source = (
            (image[:, :, : args.tile_size, : args.tile_size].contiguous(),),
            ["image"],
            str(image_path),
        )
    elif args.component == "garment":
        from Mobile_VTON.models.unets.unet_2d_condition_garment import (
            UNet2DConditionModel,
        )

        class GarmentFeatures(torch.nn.Module):
            def __init__(self, denoiser):
                super().__init__()
                self.denoiser = denoiser

            def forward(self, cloth, timestep, text):
                return tuple(
                    self.denoiser(
                        sample=cloth,
                        timestep=timestep,
                        encoder_hidden_states=text,
                        return_dict=False,
                    )[1]
                )

        weight_path = (
            args.checkpoint / "denoiser_garment/diffusion_pytorch_model.safetensors"
        )
        model = GarmentFeatures(
            UNet2DConditionModel.from_pretrained(
                args.checkpoint / "denoiser_garment", local_files_only=True
            )
        )
        text_path = (
            ROOT
            / "experiments/2026-10-23-mobile_vton/outputs_bf16_cached/embeddings.pt"
        )
        text = torch.load(text_path, map_location="cpu", weights_only=True)[
            "garment_prompt"
        ][0].float()
        cache_path = (
            ROOT
            / "experiments/2026-10-23-mobile_vton/outputs_garment_precompute/garment_cache.pt"
        )
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        parameters = cache["posterior_parameters"].float()
        cloth = (parameters[:, :16] - 0.0609) * 1.5305
        inputs, input_names, source = (
            (cloth, torch.tensor([1000.0]), text),
            ["cloth", "timestep", "text"],
            str(cache_path),
        )
    else:
        from Mobile_VTON.models.unets.unet_2d_condition_tryon import (
            UNet2DConditionModel,
        )

        class TryonExport(torch.nn.Module):
            def __init__(self, denoiser):
                super().__init__()
                self.denoiser = denoiser

            def forward(self, sample, timestep, text, image_embeds, *features):
                return self.denoiser(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=text,
                    added_cond_kwargs={"image_embeds": image_embeds},
                    garment_features=features,
                    return_dict=False,
                )[0]

        weight_path = args.checkpoint / "denoiser/diffusion_pytorch_model.safetensors"
        denoiser = UNet2DConditionModel.from_pretrained(
            args.checkpoint / "denoiser", local_files_only=True
        )
        denoiser.set_sample_size_rsd(256, 96)
        model = TryonExport(denoiser)
        fixture_path = args.output / "garment_fixtures.pt"
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
        features = tuple(torch.cat([x, x], dim=1) for x in fixture["features"])
        cache_path = (
            ROOT
            / "experiments/2026-10-23-mobile_vton/outputs_garment_precompute/garment_cache.pt"
        )
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        image_embeds = cache["projected_image_embeds"][1:].float()
        text_path = (
            ROOT
            / "experiments/2026-10-23-mobile_vton/outputs_bf16_cached/embeddings.pt"
        )
        text = torch.load(text_path, map_location="cpu", weights_only=True)["prompt"][
            0
        ].float()
        sample = torch.randn(
            1, 32, 256, 96, generator=torch.Generator().manual_seed(42)
        )
        inputs = (sample, torch.tensor([1000.0]), text, image_embeds, *features)
        input_names = [
            "sample",
            "timestep",
            "text",
            "image_embeds",
            *[f"feature_{i}" for i in range(len(features))],
        ]
        source = str(fixture_path)
    model.eval().requires_grad_(False).float()
    if args.component == "vae_encoder_portable":

        class PortableGroupNorm(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.num_groups = original.num_groups
                self.eps = original.eps
                self.weight = original.weight
                self.bias = original.bias

            def forward(self, image):
                grouped = image.reshape(image.shape[0], self.num_groups, -1)
                mean = grouped.mean(dim=-1, keepdim=True)
                centered = grouped - mean
                variance = (centered * centered).mean(dim=-1, keepdim=True)
                normalized = (centered / torch.sqrt(variance + self.eps)).reshape(
                    image.shape
                )
                affine_shape = (1, -1) + (1,) * (image.ndim - 2)
                return normalized * self.weight.reshape(
                    affine_shape
                ) + self.bias.reshape(affine_shape)

        def replace_norms(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.GroupNorm):
                    setattr(parent, name, PortableGroupNorm(child))
                else:
                    replace_norms(child)

        replace_norms(model)
    graph = args.output / f"{args.component}.onnx"
    started = time.perf_counter()
    with torch.inference_mode():
        reference = model(*inputs)
        outputs = reference if isinstance(reference, tuple) else (reference,)
        output_names = (
            ["image"]
            if args.component == "decoder"
            else [f"output_{i}" for i in range(len(outputs))]
        )
        torch.onnx.export(
            model,
            inputs,
            graph,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(str(graph))
    for name, value in zip(input_names, inputs):
        suffix = "input" if args.component == "decoder" else f"input_{name}"
        value.float().numpy().tofile(args.output / f"{args.component}_{suffix}.f32")
    for name, value in zip(output_names, outputs):
        suffix = "reference" if args.component == "decoder" else f"reference_{name}"
        value.float().numpy().tofile(args.output / f"{args.component}_{suffix}.f32")
    if args.component == "garment":
        torch.save(
            {"inputs": inputs, "features": outputs}, args.output / "garment_fixtures.pt"
        )
    report = {
        "status": "exported",
        "component": args.component,
        "weights": str(weight_path),
        "input_source": source,
        "inputs": {name: list(value.shape) for name, value in zip(input_names, inputs)},
        "outputs": {
            name: list(value.shape) for name, value in zip(output_names, outputs)
        },
        "export_reference_precision": "fp32",
        "export_seconds": time.perf_counter() - started,
        "onnx_bytes": graph.stat().st_size,
        "onnx_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
        "ops": sorted({node.op_type for node in onnx.load(str(graph)).graph.node}),
        "note": "Component validation with original FP32 weights; not full phone generation. Garment fixture uses posterior mean for export only.",
    }
    (args.output / f"{args.component}_export.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
