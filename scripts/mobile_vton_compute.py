"""Optional compute-memory experiments for the pinned Mobile-VTON modules."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable

import torch
import torch.nn.functional as F


def chunk_feed_forward(module: torch.nn.Module, size: int) -> list[str]:
    """Use upstream token-wise FFN chunking, leaving attention unchanged."""
    if size <= 0:
        raise ValueError("FFN chunk size must be positive")
    changed = []
    for name, child in module.named_modules():
        setter = getattr(child, "set_chunk_feed_forward", None)
        if setter is not None:
            setter(chunk_size=size, dim=1)
            changed.append(name)
    if not changed:
        raise ValueError("No upstream FFN chunking modules found")
    return changed


def native_gqa_call(call: Callable) -> Callable:
    """Remove explicit KV replication and dispatch through a checked backend."""
    original = inspect.unwrap(call)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    counts = {"replication": 0, "attention": 0}

    class Rewrite(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == "kv_heads != attn.heads":
                counts["replication"] += 1
                if (
                    sum(
                        isinstance(child, ast.Call)
                        and ast.unparse(child.func) == "torch.repeat_interleave"
                        for child in ast.walk(node)
                    )
                    != 2
                ):
                    raise ValueError("Unexpected KV replication block")
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == "F.scaled_dot_product_attention":
                counts["attention"] += 1
                node.func = ast.parse("self._attention", mode="eval").body
                node.keywords.append(
                    ast.keyword(
                        arg="enable_gqa",
                        value=ast.parse("kv_heads != attn.heads", mode="eval").body,
                    )
                )
            return self.generic_visit(node)

    adapted = Rewrite().visit(tree)
    if counts != {"replication": 1, "attention": 1}:
        raise ValueError(f"Unexpected attention processor anchors: {counts}")
    ast.fix_missing_locations(adapted)
    namespace = {}
    exec(  # noqa: S102 - transforms trusted, pinned upstream source only
        compile(adapted, original.__code__.co_filename + " [native GQA]", "exec"),
        original.__globals__,
        namespace,
    )
    return namespace[original.__name__]


def install_native_gqa(module: torch.nn.Module) -> list[str]:
    """Require Flash SDPA on CUDA so math fallback cannot hide KV replication."""
    from Mobile_VTON.models.attention_processor import AttnProcessor2_0

    class NativeGQA(AttnProcessor2_0):
        __call__ = native_gqa_call(AttnProcessor2_0.__call__)

        @staticmethod
        def _attention(query, key, value, **kwargs):
            if query.device.type == "cuda":
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(query, key, value, **kwargs)
            return F.scaled_dot_product_attention(query, key, value, **kwargs)

    changed = []
    for name, child in module.named_modules():
        if isinstance(getattr(child, "processor", None), AttnProcessor2_0):
            child.set_processor(NativeGQA())
            changed.append(name)
    if not changed:
        raise ValueError("No compatible upstream attention processors found")
    return changed


class ModuleMemoryProfile:
    """Record modules crossing a cumulative CUDA high without resetting counters.

    Nested records identify both the containing block and the narrowest module
    spanning the high. Records store scalar metadata, never tensor references.
    Profiling runs must be kept separate from timing comparisons.
    """

    def __init__(self, device: torch.device, events: list[dict]) -> None:
        if device.type != "cuda":
            raise ValueError("Module memory profiling requires CUDA")
        self.device = device
        self.events = events

    def watch(self, module: torch.nn.Module, prefix: str) -> None:
        for name, child in module.named_modules():
            frames = []
            qualified = f"{prefix}.{name}" if name else prefix

            def before(mod, args, kwargs, frames=frames):
                frames.append(
                    (
                        torch.cuda.max_memory_allocated(self.device),
                        torch.cuda.memory_allocated(self.device),
                    )
                )

            def after(mod, args, kwargs, result, frames=frames, name=qualified):
                peak_before, allocated_before = frames.pop()
                peak = torch.cuda.max_memory_allocated(self.device)
                if peak > peak_before:
                    self.events.append(
                        {
                            "module": name,
                            "type": type(mod).__name__,
                            "allocated_before_bytes": allocated_before,
                            "allocated_after_bytes": torch.cuda.memory_allocated(
                                self.device
                            ),
                            "peak_before_bytes": peak_before,
                            "peak_after_bytes": peak,
                        }
                    )

            child.register_forward_pre_hook(before, with_kwargs=True)
            child.register_forward_hook(after, with_kwargs=True)
