"""Opt-in memory experiments without changing the pinned upstream checkout."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

import torch
from mobile_vton_quantization import storage_bytes


class PhaseModel(torch.nn.Module):
    """Load a preprocessing/decoding model only during its phase, then release it."""

    def __init__(
        self,
        name: str,
        factory: Callable[[], torch.nn.Module],
        calls_per_phase: int,
        events: list[dict],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.name = name
        self.factory = factory
        self.calls_per_phase = calls_per_phase
        self.events = events
        self.device = device
        # Upstream reads the first parameter's dtype before image encoding.
        # A zero-element marker avoids loading weights during bookkeeping.
        self._dtype_marker = torch.nn.Parameter(
            torch.empty(0, device=device, dtype=dtype), requires_grad=False
        )
        self._active = None
        self._calls = 0

    def _record(self, action: str, **values) -> None:
        event = {"component": self.name, "action": action, **values}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            event.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                cuda_peak_allocated_so_far_bytes=torch.cuda.max_memory_allocated(
                    self.device
                ),
            )
        self.events.append(event)

    def _load(self) -> torch.nn.Module:
        if self._active is None:
            begin = time.perf_counter()
            cuda_devices = (
                [
                    (
                        self.device.index
                        if self.device.index is not None
                        else torch.cuda.current_device()
                    )
                ]
                if self.device.type == "cuda"
                else []
            )
            # In particular, constructing the final decoder initializes random
            # weights before loading its state. Preserve generation RNG streams.
            with torch.random.fork_rng(devices=cuda_devices):
                self._active = self.factory().eval().requires_grad_(False)
            self._calls = 0
            self._record(
                "load",
                seconds=time.perf_counter() - begin,
                parameters=sum(p.numel() for p in self._active.parameters()),
                model_tensor_storage_bytes=storage_bytes(self._active),
            )
        return self._active

    def _finish_call(self) -> None:
        self._calls += 1
        self._record("call", call=self._calls)
        if self._calls == self.calls_per_phase:
            self._active = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._record("release")

    def encode(self, *args, **kwargs):
        result = self._load().encode(*args, **kwargs)
        self._finish_call()
        return result

    def forward(self, *args, **kwargs):
        result = self._load()(*args, **kwargs)
        self._finish_call()
        return result


def sequential_cfg_forward(forward: Callable, enabled: Callable[[], bool]) -> Callable:
    """Evaluate the two CFG branches separately, keeping upstream guidance intact.

    Upstream still constructs the batched input and garment features. This lowers
    denoiser intermediates, not all duplicated conditioning tensors or weights.
    """

    def wrapped(*args, **kwargs):
        if not enabled():
            return forward(*args, **kwargs)
        if args or kwargs.get("return_dict", True):
            raise ValueError(
                "Sequential CFG requires keyword inputs and return_dict=False"
            )
        batch = kwargs["sample"].shape[0]
        if batch % 2:
            raise ValueError(
                "CFG input must contain equal unconditional/conditional halves"
            )
        half = batch // 2

        def branch(value, begin):
            if isinstance(value, torch.Tensor):
                if value.ndim and value.shape[0] == batch:
                    return value[begin : begin + half]
                return value
            if isinstance(value, dict):
                return {key: branch(item, begin) for key, item in value.items()}
            if isinstance(value, list):
                return [branch(item, begin) for item in value]
            if isinstance(value, tuple):
                return tuple(branch(item, begin) for item in value)
            return value

        unconditional = forward(**branch(kwargs, 0))[0]
        conditional = forward(**branch(kwargs, half))[0]
        return (torch.cat([unconditional, conditional]),)

    return wrapped
