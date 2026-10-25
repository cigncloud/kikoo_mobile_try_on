"""Run a pinned Mobile-VTON checkpoint on one pair and record real measurements."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key, value in {
    "HF_HOME": "/tmp/mobile_tryon/hf_cache",
    "MPLCONFIGDIR": "/tmp/mobile_tryon/matplotlib",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "USE_TF": "0",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(key, value)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def run(config: dict, output: Path, report: dict) -> None:
    import numpy as np
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from PIL import Image, ImageDraw, ImageOps
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

    upstream = (ROOT / config["upstream"]).resolve()
    git = shutil.which("git")
    if git:
        actual_revision = subprocess.check_output(
            [git, "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
    else:
        # Minimal CUDA/vLLM images omit git; validate the real checkout's HEAD.
        git_dir = upstream / ".git"
        head = (git_dir / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head.removeprefix("ref: ")
            loose_ref = git_dir / ref
            if loose_ref.is_file():
                actual_revision = loose_ref.read_text().strip()
            else:
                refs = (git_dir / "packed-refs").read_text().splitlines()
                actual_revision = next(
                    line.split()[0] for line in refs if line.endswith(" " + ref)
                )
        else:
            actual_revision = head
        report["upstream_revision_check"] = "git_head_files"
    if actual_revision != config["upstream_revision"]:
        raise ValueError(f"Unexpected source revision: {actual_revision}")
    report["upstream_revision_actual"] = actual_revision
    sys.path.insert(0, str(upstream))
    from Mobile_VTON.models.autoencoders.vae import Decoder
    from Mobile_VTON.models.unets.unet_2d_condition_garment import (
        UNet2DConditionModel as Garment,
    )
    from Mobile_VTON.models.unets.unet_2d_condition_tryon import (
        UNet2DConditionModel as Tryon,
    )
    from Mobile_VTON.pipelines.tryon_pipeline_full_cat import (
        T2IMobilePipelineV1_3_NotLoadingT5_Decoder as Pipeline,
    )

    torch.set_num_threads(config["threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; choose --device cpu explicitly for a smoke test"
        )
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        config["dtype"]
    ]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("Use fp32 or bf16 for CPU inference")
    if device.type == "cuda":
        report["cuda_allocator_config"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        report["gpu"] = torch.cuda.get_device_name(device)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(
            device
        ).total_memory
        torch.cuda.reset_peak_memory_stats(device)
    report["packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "accelerate",
            "safetensors",
            "numpy",
            "pillow",
        )
    }
    quantization = config.get("quantization", "none")
    if quantization not in ("none", "int4"):
        raise ValueError(f"Unknown quantization: {quantization}")
    if quantization == "int4":
        report["packages"]["torchao"] = importlib.metadata.version("torchao")
    cache_path = (
        ROOT / config["embedding_cache"] if config.get("embedding_cache") else None
    )
    cached = (
        torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_path and cache_path.exists()
        else None
    )
    garment_cache_path = (
        ROOT / config["garment_cache"] if config.get("garment_cache") else None
    )
    use_garment_cache = garment_cache_path is not None and not config.get(
        "build_garment_cache"
    )
    if garment_cache_path is not None and (
        device.type != "cuda" or not cached or not config.get("release_working_tensors")
    ):
        raise ValueError(
            "Garment caching requires CUDA, text cache and lifetime adapter"
        )
    if use_garment_cache and not garment_cache_path.is_file():
        raise FileNotFoundError(garment_cache_path)
    if cached and (
        cached["description"] != config["garment_description"]
        or cached["model_revision"] != config["model_revision"]
    ):
        raise ValueError("Embedding cache description/model revision does not match")

    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    report["checkpoint_file_bytes"] = sum(
        p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()
    )
    source_images = {}
    for key in ("person", "garment"):
        path = ROOT / config[key]
        with Image.open(path) as image:
            source_images[key] = ImageOps.exif_transpose(image).convert("RGB")
        report.setdefault("inputs", {})[key] = {
            "path": config[key],
            "sha256": sha256(path),
            "size": list(source_images[key].size),
        }

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    load_start = time.perf_counter()
    from mobile_vton_compute import (
        ModuleMemoryProfile,
        chunk_feed_forward,
        install_native_gqa,
    )

    profile_events: list[dict] = []
    profiler = (
        ModuleMemoryProfile(device, profile_events)
        if config.get("profile_modules")
        else None
    )
    report["module_memory_profile"] = {
        "enabled": profiler is not None,
        "events": profile_events,
    }
    report["compute_optimization"] = {
        "ffn_chunk_size": config.get("ffn_chunk_size", 0),
        "native_gqa": config.get("native_gqa", False),
        "spatial_conv_rows": config.get("spatial_conv_rows", 0),
        "spatial_decoder_only": config.get("spatial_decoder_only", False),
        "modules": {},
    }

    def watch(module, name):
        if profiler is not None:
            profiler.watch(module, name)
        return module

    local = {"local_files_only": True}
    tokenizers = [
        CLIPTokenizer.from_pretrained(checkpoint / name, **local)
        for name in ("tokenizer", "tokenizer_2")
    ]
    load_model = {"torch_dtype": dtype, "low_cpu_mem_usage": True, **local}
    memory_mode = config.get("memory_mode", "baseline")
    if memory_mode not in ("baseline", "sequential_cfg", "phase_release", "combined"):
        raise ValueError(f"Unknown memory mode: {memory_mode}")
    phase_release = memory_mode in ("phase_release", "combined")
    defer_quantized_transfer = config.get("defer_quantized_transfer", False)
    if defer_quantized_transfer and not (
        device.type == "cuda"
        and quantization == "int4"
        and config.get("quantized_state_dir")
    ):
        raise ValueError("Deferred transfer requires CUDA and saved INT4 states")
    memory_events: list[dict] = []
    report["memory_optimization"] = {
        "mode": memory_mode,
        "events": memory_events,
        "phase_release": phase_release,
        "sequential_cfg": memory_mode in ("sequential_cfg", "combined"),
        "defer_quantized_transfer": defer_quantized_transfer,
    }
    vae_events: list[dict] = []
    report["vae_optimization"] = {
        "encoder_only": config.get("vae_encoder_only", False),
        "tile_size": config.get("vae_tile_size", 0),
        "tile_overlap_factor": 0.25,
        "events": vae_events,
    }
    decoder_dtype_name = config.get("decoder_dtype", "fp32")
    if decoder_dtype_name not in ("fp32", "fp16"):
        raise ValueError(f"Unknown decoder dtype: {decoder_dtype_name}")
    decoder_dtype = {"fp32": torch.float32, "fp16": torch.float16}[decoder_dtype_name]
    decoder_events: list[dict] = []
    report["decoder_optimization"] = {
        "dtype": decoder_dtype_name,
        "events": decoder_events,
        "latent_artifact": str((output / "decoder_latents.pt").relative_to(ROOT)),
    }
    lifetime_events: list[dict] = []
    report["working_tensor_release"] = {
        "enabled": config.get("release_working_tensors", False),
        "events": lifetime_events,
    }
    if config.get("release_working_tensors"):
        from mobile_vton_lifetimes import release_working_tensors

        class WorkingTensorPipeline(Pipeline):
            __call__ = release_working_tensors(
                Pipeline.__call__, garment_cached=use_garment_cache
            )

        pipeline_type = WorkingTensorPipeline
    else:
        pipeline_type = Pipeline

    def load_image_encoder():
        return watch(
            AutoModel.from_pretrained(checkpoint / "image_encoder", **load_model).to(
                device=device, dtype=dtype
            ),
            "image_encoder",
        )

    def load_vae():
        from mobile_vton_vae import EncoderOnlyVAE, configure_vae_tiling

        if config.get("vae_encoder_only"):
            vae = EncoderOnlyVAE.from_checkpoint(checkpoint / "vae")
        else:
            vae = AutoencoderKL.from_pretrained(checkpoint / "vae", **load_model)
        vae.to(device=device, dtype=dtype)
        configure_vae_tiling(vae, config.get("vae_tile_size", 0), vae_events)
        return watch(vae, "vae")

    def load_decoder():
        from mobile_vton_decoder import PrecisionDecoder

        decoder = Decoder(
            **json.loads((checkpoint / "vae_decoder/decoder.json").read_text())
        )
        decoder.load_state_dict(
            torch.load(
                checkpoint / "vae_decoder/decoder.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        decoder.to(device=device, dtype=decoder_dtype)
        if config.get("spatial_conv_rows"):
            from mobile_vton_spatial import install_spatial_tiling

            report["compute_optimization"]["modules"]["vae_decoder_spatial"] = (
                install_spatial_tiling(
                    decoder, config["spatial_conv_rows"], separable=False
                )
            )
        return PrecisionDecoder(
            watch(decoder, "vae_decoder"),
            decoder_events,
            output / "decoder_latents.pt",
            audit_latents=config.get("audit_decoder_latents", True),
        )

    from mobile_vton_memory import PhaseModel, sequential_cfg_forward

    def phase_or_eager(name, factory, calls):
        if phase_release:
            return PhaseModel(
                name,
                factory,
                calls,
                memory_events,
                device,
                dtype=decoder_dtype if name == "vae_decoder" else dtype,
            )
        return factory()

    components = {
        "text_encoder": (
            None
            if cached
            else CLIPTextModelWithProjection.from_pretrained(
                checkpoint / "text_encoder", **load_model
            )
        ),
        "text_encoder_2": (
            None
            if cached
            else CLIPTextModelWithProjection.from_pretrained(
                checkpoint / "text_encoder_2", **load_model
            )
        ),
        "image_encoder": (
            None
            if use_garment_cache
            else phase_or_eager("image_encoder", load_image_encoder, 2)
        ),
        "vae": phase_or_eager("vae", load_vae, 2),
        "denoiser": Tryon.from_pretrained(checkpoint / "denoiser", **load_model),
        "denoiser_garment": Garment.from_pretrained(
            checkpoint / "denoiser_garment", **load_model
        ),
    }
    components["vae_decoder"] = phase_or_eager("vae_decoder", load_decoder, 1)
    report["components"] = {}
    for name, module in components.items():
        if module is None:
            continue
        if isinstance(module, PhaseModel):
            report["components"][name] = {"deferred_load": True, "parameters": 0}
            continue
        module.eval().requires_grad_(False)
        if not (defer_quantized_transfer and name in ("denoiser", "denoiser_garment")):
            module.to(
                device=device, dtype=decoder_dtype if name == "vae_decoder" else dtype
            )
        report["components"][name] = {
            "parameters": sum(p.numel() for p in module.parameters()),
            "parameter_bytes_in_memory": sum(
                p.numel() * p.element_size() for p in module.parameters()
            ),
        }
    report["total_parameters"] = sum(
        c["parameters"] for c in report["components"].values()
    )
    from mobile_vton_quantization import storage_bytes

    report["quantization"] = {}
    if quantization == "int4":
        from mobile_vton_quantization import (
            load_quantized_state,
            quantize_linear_weights,
            save_quantized_state,
        )

        for name in ("denoiser", "denoiser_garment"):
            if config.get("quantized_state_dir"):
                state_path = ROOT / config["quantized_state_dir"] / f"{name}_int4.pt"
                measurement = load_quantized_state(components[name], state_path, device)
                if defer_quantized_transfer:
                    # assign=True installs the packed CUDA state directly while
                    # the original BF16 parameters still reside on the CPU.
                    components[name].to(device=device)
                    measurement["original_bf16_weights_transferred_to_cuda"] = False
            else:
                print(f"Quantizing {name} linear weights to INT4", flush=True)
                measurement = quantize_linear_weights(
                    components[name], device, config.get("group_size", 128)
                )
                if config.get("save_quantized_states", True):
                    measurement["checkpoint"] = save_quantized_state(
                        components[name], output / f"{name}_int4.pt"
                    )
            report["quantization"][name] = measurement
            save_json(output / "metrics.json", report)
    report["resolution_adapter"] = {}
    for name, height_multiplier in (("denoiser", 2), ("denoiser_garment", 1)):
        module = components[name]
        target_height = config["height"] * height_multiplier
        if (target_height, config["width"]) != (
            module.config.height,
            module.config.width,
        ):
            # The pipeline only resets TryonNet to a square; GarmentNet is not
            # resized at all. Use upstream rectangular RoPE interpolation for both.
            module.set_sample_size_rsd(target_height // 8, config["width"] // 8)
            module.sample_size = config["height"] // 8
            report["resolution_adapter"][name] = {
                "method": "upstream set_sample_size_rsd; suppress pipeline square reset",
                "latent_height": target_height // 8,
                "latent_width": config["width"] // 8,
                "rope_tokens": module.reso1_image_rotary_emb_cos.shape[0],
            }
    report["model_tensor_storage_bytes"] = sum(
        storage_bytes(m) for m in components.values() if m is not None
    )
    for name in ("denoiser", "denoiser_garment"):
        module = components[name]
        changes = {}
        if config.get("ffn_chunk_size", 0):
            changes["ffn"] = chunk_feed_forward(module, config["ffn_chunk_size"])
        if config.get("native_gqa"):
            changes["attention"] = install_native_gqa(module)
        if config.get("spatial_conv_rows") and not config.get("spatial_decoder_only"):
            from mobile_vton_spatial import install_spatial_tiling

            changes["spatial"] = install_spatial_tiling(
                module, config["spatial_conv_rows"], separable=True
            )
        report["compute_optimization"]["modules"][name] = changes
        watch(module, name)
    pipe = pipeline_type(
        **components,
        scheduler=FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=config["scheduler_shift"]
        ),
        tokenizer=tokenizers[0],
        tokenizer_2=tokenizers[1],
    )

    def record_working_tensors(stage, step):
        event = {"stage": stage, "step": step}
        if device.type == "cuda":
            event.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                cuda_peak_allocated_so_far_bytes=torch.cuda.max_memory_allocated(
                    device
                ),
            )
        lifetime_events.append(event)

    pipe._record_working_tensors = record_working_tensors
    if report["memory_optimization"]["sequential_cfg"]:
        pipe.denoiser.forward = sequential_cfg_forward(
            pipe.denoiser.forward, lambda: pipe.do_classifier_free_guidance
        )
    sync()
    report["model_load_seconds"] = time.perf_counter() - load_start
    save_json(output / "metrics.json", report)
    print(
        f"Loaded {report['total_parameters']:,} parameters in {report['model_load_seconds']:.2f}s",
        flush=True,
    )

    start = time.perf_counter()
    size = (config["width"], config["height"])
    person = source_images["person"].resize(size, Image.Resampling.LANCZOS)
    garment = source_images["garment"].resize(size, Image.Resampling.LANCZOS)
    cloth = (
        torch.from_numpy(np.array(garment).copy()).permute(2, 0, 1).float().unsqueeze(0)
        / 127.5
        - 1
    )
    cloth = cloth.to(device=device, dtype=dtype)
    ip_image = None
    if not use_garment_cache:
        processor = AutoImageProcessor.from_pretrained(
            checkpoint / "image_encoder", **local
        )
        ip_image = processor(
            images=source_images["garment"], return_tensors="pt"
        ).pixel_values.to(device=device, dtype=dtype)
    description = config["garment_description"]
    negative = "deformed, distorted, disfigured, bad anatomy, extra limb, missing limb, mutated hands, blurry"
    if cached:
        prompt = tuple(
            t.to(device=device, dtype=dtype) if t is not None else None
            for t in cached["prompt"]
        )
        garment_prompt = tuple(
            t.to(device=device, dtype=dtype) if t is not None else None
            for t in cached["garment_prompt"]
        )
    else:
        with torch.inference_mode():
            prompt = pipe.encode_prompt(
                prompt=["Replace the upper body with " + description],
                prompt_2=["Replace the upper body with " + description],
                negative_prompt=[negative],
                do_classifier_free_guidance=True,
                device=device,
            )
            garment_prompt = pipe.encode_prompt(
                prompt=[description],
                prompt_2=[description],
                negative_prompt=[negative],
                do_classifier_free_guidance=False,
                device=device,
            )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "description": description,
                    "model_revision": config["model_revision"],
                    "prompt": tuple(t.cpu() if t is not None else None for t in prompt),
                    "garment_prompt": tuple(
                        t.cpu() if t is not None else None for t in garment_prompt
                    ),
                    "source_device": str(device),
                    "source_dtype": str(dtype),
                },
                cache_path,
            )
    if cache_path:
        report["embedding_cache"] = {
            "path": str(cache_path),
            "bytes": cache_path.stat().st_size,
            "sha256": sha256(cache_path),
            "loaded": cached is not None,
        }
        # Both precisions use the same cached tensors and exclude text model residency.
        pipe.register_modules(text_encoder=None, text_encoder_2=None)
        for name in ("text_encoder", "text_encoder_2"):
            components[name] = None
        del module
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        report["runtime_model_tensor_storage_bytes"] = sum(
            storage_bytes(m) for m in components.values() if m is not None
        )
    sync()
    report["preprocess_and_text_encode_seconds"] = time.perf_counter() - start
    if garment_cache_path is not None:
        from mobile_vton_garment_cache import (
            build_garment_cache,
            cache_identity,
            install_garment_cache,
            load_garment_cache,
        )

        identity_start = time.perf_counter()
        identity = cache_identity(
            config,
            report["inputs"]["garment"]["sha256"],
            report["embedding_cache"]["sha256"],
            ROOT,
        )
        report["garment_cache_identity_check_seconds"] = (
            time.perf_counter() - identity_start
        )
        begin = time.perf_counter()
        if config.get("build_garment_cache"):
            sync()
            build_begin = time.perf_counter()
            report["garment_cache"] = build_garment_cache(
                pipe,
                cloth,
                ip_image,
                identity,
                garment_cache_path,
                config["guidance_scale"],
            )
            sync()
            report["garment_cache"]["build_encode_and_save_seconds"] = (
                time.perf_counter() - build_begin
            )
            report["garment_cache"]["precompute_cuda_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device)
            )
            report["status"] = "prepared"
            report["host_peak_rss_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            print(json.dumps(report["garment_cache"]), flush=True)
            return
        cache = load_garment_cache(garment_cache_path, identity, device)
        report["garment_cache"] = {
            "path": str(garment_cache_path),
            "bytes": garment_cache_path.stat().st_size,
            "sha256": sha256(garment_cache_path),
            "loaded": True,
            "gpu_tensors": install_garment_cache(pipe, cache),
        }
        sync()
        report["garment_cache"]["load_and_install_seconds"] = (
            time.perf_counter() - begin
        )
        # Full garment raster is no longer needed by the adapted pipeline.
        del cloth, cache
        cloth = torch.empty(0, device=device, dtype=dtype)

    def resident_snapshot():
        result = {}
        for name, component in components.items():
            if component is not None:
                devices = sorted(
                    {
                        str(t.device)
                        for t in list(component.parameters())
                        + list(component.buffers())
                    }
                )
                result[name] = {
                    "devices": devices,
                    "tensor_storage_bytes": storage_bytes(component),
                    "phase_loaded": isinstance(component, PhaseModel),
                }
                if config.get("gpu_resident") and (
                    isinstance(component, PhaseModel)
                    or any(not d.startswith("cuda") for d in devices)
                ):
                    raise RuntimeError(
                        f"Requested GPU residency violated: {name}/{devices}"
                    )
        return result

    report["gpu_residency_before"] = resident_snapshot()
    kwargs = {
        "prompt_embeds": prompt[0],
        "negative_prompt_embeds": prompt[1],
        "pooled_prompt_embeds": prompt[2],
        "negative_pooled_prompt_embeds": prompt[3],
        "height": config["height"],
        "width": config["width"],
        "guidance_scale": config["guidance_scale"],
        "text_embeds_cloth": garment_prompt[0],
        "negative_text_embeds_cloth": garment_prompt[1],
        "cloth": cloth,
        "image": person,
        "ip_adapter_image": ip_image,
        "device": device,
    }
    if config.get("prepare_only"):
        report["status"] = "prepared"
        report["host_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        print("Model prepared; inference explicitly skipped", flush=True)
        return
    report["runs"] = []

    def generate(steps: int):
        torch.manual_seed(config["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        generator = torch.Generator(device=device).manual_seed(config["seed"])
        event_begin = len(memory_events)
        vae_event_begin = len(vae_events)
        decoder_event_begin = len(decoder_events)
        with torch.inference_mode():
            pixels = pipe(
                **kwargs,
                num_inference_steps=steps,
                generator=generator,
                output_type="np",
            ).images[0]
        if phase_release:
            for name in ("vae", "image_encoder", "vae_decoder"):
                if (
                    components[name] is not None
                    and components[name]._active is not None
                ):
                    raise RuntimeError(f"Phase model was not released: {name}")
        report["memory_optimization"]["last_generation_event_range"] = [
            event_begin,
            len(memory_events),
        ]
        report["vae_optimization"]["last_generation_event_range"] = [
            vae_event_begin,
            len(vae_events),
        ]
        report["decoder_optimization"]["last_generation_event_range"] = [
            decoder_event_begin,
            len(decoder_events),
        ]
        return pixels

    if config["warmup_steps"]:
        warmup_begin = time.perf_counter()
        generate(config["warmup_steps"])
        sync()
        report["warmup_seconds"] = time.perf_counter() - warmup_begin
        report["warmup_steps"] = config["warmup_steps"]
    if device.type == "cuda":
        report["cuda_peak_allocated_bytes_before_benchmark"] = (
            torch.cuda.max_memory_allocated(device)
        )
        report["cuda_peak_reserved_bytes_before_benchmark"] = (
            torch.cuda.max_memory_reserved(device)
        )
        torch.cuda.reset_peak_memory_stats(device)
    report["host_peak_rss_bytes_before_benchmark"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    previous = None
    for index in range(config["runs"]):
        sync()
        begin = time.perf_counter()
        event_begin = len(memory_events)
        vae_event_begin = len(vae_events)
        decoder_event_begin = len(decoder_events)
        pixels = generate(config["steps"])
        sync()
        elapsed = time.perf_counter() - begin
        if not np.isfinite(pixels).all():
            raise ValueError("Generated output contains NaN or Inf")
        if pixels.shape != (config["height"], config["width"], 3):
            raise ValueError(f"Unexpected output shape: {pixels.shape}")
        image = Image.fromarray((pixels.clip(0, 1) * 255).round().astype("uint8"))
        path = output / f"result_{index + 1:02d}.png"
        image.save(path)
        measurement = {
            "pipeline_seconds": elapsed,
            "output": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "pixel_min": float(pixels.min()),
            "pixel_max": float(pixels.max()),
            "pixel_std": float(pixels.std()),
            "memory_event_range": [event_begin, len(memory_events)],
            "vae_event_range": [vae_event_begin, len(vae_events)],
            "decoder_event_range": [decoder_event_begin, len(decoder_events)],
        }
        measurement["host_current_rss_bytes"] = int(
            Path("/proc/self/statm").read_text().split()[1]
        ) * os.sysconf("SC_PAGE_SIZE")
        if previous is not None:
            measurement["repeat_max_absolute_pixel_difference"] = float(
                np.abs(pixels - previous).max()
            )
        previous = pixels.copy()
        if device.type == "cuda":
            measurement["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            measurement["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
        report["runs"].append(measurement)
        save_json(output / "metrics.json", report)
        print(json.dumps(measurement), flush=True)
        sheet = Image.new("RGB", (size[0] * 3, size[1] + 40), "white")
        for column, (label, tile) in enumerate(
            (("Person", person), ("Garment", garment), ("Mobile-VTON result", image))
        ):
            sheet.paste(tile, (column * size[0], 40))
            ImageDraw.Draw(sheet).text((column * size[0] + 12, 12), label, fill="black")
        sheet.save(output / f"comparison_{index + 1:02d}.jpg", quality=95)
    report["status"] = "passed"
    report["gpu_residency_after"] = resident_snapshot()
    if device.type == "cuda":
        report["cuda_peak_reserved_bytes_all_phases"] = max(
            [report["cuda_peak_reserved_bytes_before_benchmark"]]
            + [run["cuda_peak_reserved_bytes"] for run in report["runs"]]
        )
    report["cuda_peak_allocated_bytes_all_phases"] = (
        max(
            [report.get("cuda_peak_allocated_bytes_before_benchmark", 0)]
            + [run.get("cuda_peak_allocated_bytes", 0) for run in report["runs"]]
        )
        if device.type == "cuda"
        else None
    )
    for event in memory_events:
        if event["action"] == "load":
            report["components"][event["component"]].update(
                parameters=event["parameters"],
                model_tensor_storage_bytes=event["model_tensor_storage_bytes"],
            )
    report["total_parameters"] = sum(
        c["parameters"] for c in report["components"].values()
    )
    report["host_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    report["limitations"] = [
        "Desktop functional test; no Android/iOS inference or mobile FPS measurement.",
        "One pair is insufficient to establish model quality or category coverage.",
        "pipeline_seconds excludes model load, image preprocessing, and text encoding; these are reported separately.",
        "No reference target image: SSIM/LPIPS/FID are not calculated.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    for name, kind in (
        ("device", str),
        ("dtype", str),
        ("width", int),
        ("height", int),
        ("steps", int),
        ("runs", int),
        ("warmup_steps", int),
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    parser.add_argument("--output-dir")
    parser.add_argument("--quantization", choices=("none", "int4"))
    parser.add_argument("--embedding-cache")
    parser.add_argument("--quantized-state-dir")
    parser.add_argument("--vae-encoder-only", action="store_true", default=None)
    parser.add_argument("--vae-tile-size", type=int)
    parser.add_argument("--decoder-dtype", choices=("fp32", "fp16"))
    parser.add_argument("--defer-quantized-transfer", action="store_true", default=None)
    parser.add_argument("--release-working-tensors", action="store_true", default=None)
    parser.add_argument("--ffn-chunk-size", type=int)
    parser.add_argument("--native-gqa", action="store_true", default=None)
    parser.add_argument("--profile-modules", action="store_true", default=None)
    parser.add_argument("--spatial-conv-rows", type=int)
    parser.add_argument("--spatial-decoder-only", action="store_true", default=None)
    parser.add_argument("--garment-cache")
    parser.add_argument("--build-garment-cache", action="store_true", default=None)
    parser.add_argument("--gpu-resident", action="store_true", default=None)
    parser.add_argument(
        "--memory-mode",
        choices=("baseline", "sequential_cfg", "phase_release", "combined"),
    )
    parser.add_argument("--prepare-only", action="store_true", default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value
    if (
        config["width"] % 32
        or config["height"] % 32
        or min(config["width"], config["height"], config["steps"], config["runs"]) <= 0
    ):
        parser.error("Use positive dimensions divisible by 32, and positive steps/runs")
    output = (ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(output / "metrics.json", report)
    try:
        run(config, output, report)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        save_json(output / "metrics.json", report)


if __name__ == "__main__":
    main()
