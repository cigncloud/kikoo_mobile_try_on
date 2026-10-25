"""Weight-only INT4 helpers; activations and convolution weights keep their dtype."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def storage_bytes(module: torch.nn.Module) -> int:
    """Count packed backing storage instead of a quantized tensor's logical size."""
    seen: set[tuple] = set()

    def visit(tensor: torch.Tensor) -> int:
        if hasattr(tensor, "__tensor_flatten__"):
            names, _ = tensor.__tensor_flatten__()
            return sum(visit(getattr(tensor, name)) for name in names)
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()

    return sum(visit(tensor) for tensor in (*module.parameters(), *module.buffers()))


def quantize_linear_weights(
    module: torch.nn.Module, device: torch.device, group_size: int = 128
) -> dict:
    """Apply native torchao INT4 kernels, then verify which weights were packed."""
    from torchao.dtypes import (
        AffineQuantizedTensor,
        Int4CPULayout,
        TensorCoreTiledLayout,
    )
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    candidates = {
        name: child.weight.numel()
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.Linear)
    }
    before = storage_bytes(module)
    layout = Int4CPULayout() if device.type == "cpu" else TensorCoreTiledLayout()

    def compatible(layer: torch.nn.Module, name: str) -> bool:
        return (
            isinstance(layer, torch.nn.Linear)
            and layer.in_features % group_size == 0
            and (device.type != "cpu" or layer.out_features % 16 == 0)
        )

    quantize_(
        module,
        Int4WeightOnlyConfig(group_size=group_size, layout=layout),
        filter_fn=compatible,
    )
    packed = {
        name: candidates[name]
        for name, child in module.named_modules()
        if name in candidates and isinstance(child.weight, AffineQuantizedTensor)
    }
    if not packed:
        raise RuntimeError("INT4 conversion packed no weights")
    return {
        "method": "torchao Int4WeightOnlyConfig",
        "layout": type(layout).__name__,
        "group_size": group_size,
        "activation_dtype": str(next(module.parameters()).dtype),
        "linear_layers_total": len(candidates),
        "linear_layers_quantized": len(packed),
        "linear_weight_parameters_quantized": sum(packed.values()),
        "skipped_linear_layers": {
            name: count for name, count in candidates.items() if name not in packed
        },
        "storage_bytes_before": before,
        "storage_bytes_after": storage_bytes(module),
        "quantized_layers": packed,
    }


def save_quantized_state(module: torch.nn.Module, path: Path) -> dict:
    """Save the actual packed tensor state, including scale/zero metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def load_quantized_state(
    module: torch.nn.Module, path: Path, device: torch.device
) -> dict:
    """Restore our local tensor state with PyTorch's restricted weights loader."""
    # Import registers torchao's allowed tensor types with weights_only=True.
    from torchao.dtypes import AffineQuantizedTensor

    before = storage_bytes(module)
    load_device = (
        torch.device("cuda", torch.cuda.current_device())
        if device.type == "cuda" and device.index is None
        else device
    )
    state = torch.load(path, map_location=load_device, weights_only=True)
    module.load_state_dict(state, strict=True, assign=True)
    packed = {
        name: child.weight.numel()
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.Linear)
        and isinstance(child.weight, AffineQuantizedTensor)
    }
    if not packed:
        raise ValueError(f"Checkpoint contains no INT4 linear weights: {path}")
    return {
        "loaded_checkpoint": str(path),
        "linear_layers_quantized": len(packed),
        "linear_weight_parameters_quantized": sum(packed.values()),
        "storage_bytes_before": before,
        "storage_bytes_after": storage_bytes(module),
    }
