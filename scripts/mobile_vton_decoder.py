"""Evaluate the final decoder in a chosen precision with identical latents."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch


class PrecisionDecoder(torch.nn.Module):
    """Cast only at the decoder boundary; return FP32 to image postprocessing."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        events: list[dict],
        latent_path: Path | None = None,
        audit_latents: bool = True,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.events = events
        self.latent_path = latent_path
        self.audit_latents = audit_latents

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # Upstream explicitly passes FP32 latents even when the decoder is FP16.
        latent_hash = None
        if self.audit_latents:
            cpu_latents = sample.detach().float().cpu().contiguous()
            latent_hash = hashlib.sha256(cpu_latents.numpy().tobytes()).hexdigest()
            if self.latent_path is not None:
                torch.save(cpu_latents, self.latent_path)
        dtype = next(self.decoder.parameters()).dtype
        if sample.device.type == "cuda":
            torch.cuda.synchronize(sample.device)
        begin = time.perf_counter()
        output = self.decoder(sample.to(dtype=dtype)).float()
        if sample.device.type == "cuda":
            torch.cuda.synchronize(sample.device)
        elapsed = time.perf_counter() - begin
        finite = bool(torch.isfinite(output).all())
        event = {
            "input_shape": list(sample.shape),
            "input_dtype": str(sample.dtype),
            "latent_sha256_fp32": latent_hash,
            "model_dtype": str(dtype),
            "output_dtype": str(output.dtype),
            "output_shape": list(output.shape),
            "finite": finite,
            "decoder_forward_seconds": elapsed,
        }
        if sample.device.type == "cuda":
            event["cuda_peak_allocated_so_far_bytes"] = torch.cuda.max_memory_allocated(
                sample.device
            )
        self.events.append(event)
        if not finite:
            raise ValueError("Final decoder produced NaN or Inf")
        return output
