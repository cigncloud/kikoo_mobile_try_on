"""Export DM-VTON's two student stages through an ONNX/MNN mobile path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "models/dm_vton/upstream"
sys.path.insert(0, str(UPSTREAM))


def torch_correlation(
    first: torch.Tensor | None = None,
    second: torch.Tensor | None = None,
    stride: int = 1,
    *,
    tenFirst: torch.Tensor | None = None,
    tenSecond: torch.Tensor | None = None,
    intStride: int | None = None,
) -> torch.Tensor:
    """Pure ATen equivalent of DM-VTON's CuPy 7x7 correlation kernel."""
    first = first if first is not None else tenFirst
    second = second if second is not None else tenSecond
    stride = stride if intStride is None else intStride
    assert first is not None and second is not None
    height, width = first.shape[-2:]
    padded = F.pad(
        second,
        (3 * stride, 3 * stride, 3 * stride, 3 * stride),
        mode="constant",
        value=0,
    )
    outputs = []
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            y0 = 3 * stride + dy * stride
            x0 = 3 * stride + dx * stride
            shifted = padded[:, :, y0 : y0 + height : stride, x0 : x0 + width : stride]
            outputs.append(
                (first[:, :, ::stride, ::stride] * shifted).mean(dim=1, keepdim=True)
            )
    return torch.cat(outputs, dim=1)


class WarpExport(torch.nn.Module):
    def __init__(self, warp):
        super().__init__()
        self.warp = warp

    def forward(self, person, clothes, edge):
        edge = (edge > 0.5).float()
        clothes = clothes * edge
        warped, flow = self.warp(person, clothes, phase="test")
        warped_edge = F.grid_sample(
            edge,
            flow.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped, warped_edge


class GeneratorExport(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, person, warped, warped_edge):
        values = self.generator(torch.cat([person, warped, warped_edge], dim=1))
        rendered, composite = torch.split(values, [3, 1], dim=1)
        rendered = torch.tanh(rendered)
        composite = torch.sigmoid(composite) * warped_edge
        return warped * composite + rendered * (1 - composite)


def load(model, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    expected = model.state_dict()
    model.load_state_dict({name: state[name] for name in expected}, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warp", type=Path, default=Path("/tmp/dmvton_pf_warp.pt"))
    parser.add_argument(
        "--generator", type=Path, default=Path("/tmp/dmvton_pf_gen.1.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/2026-10-25-dm_vton"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        import cupy
    except ModuleNotFoundError:
        # The export replaces the only CuPy-dependent operation with ATen.
        # A stub keeps the upstream module importable in the pinned .venv.
        import types

        cupy = types.SimpleNamespace(
            cuda=types.SimpleNamespace(),
            memoize=lambda **_: lambda function: function,
        )
        sys.modules["cupy"] = cupy
    from models.common import correlation
    from models.warp_modules import mobile_afwm
    from pipelines import DMVTONPipeline

    mobile_afwm.FunctionCorrelation = torch_correlation
    correlation.FunctionCorrelation = torch_correlation
    pipeline = DMVTONPipeline(align_corners=True)
    load(pipeline.warp_model, args.warp)
    load(pipeline.gen_model, args.generator)
    pipeline.eval().requires_grad_(False).float()
    person = (
        torch.from_numpy(np.load(args.output / "person.npy")).float()
        if (args.output / "person.npy").exists()
        else None
    )
    if person is None:
        from PIL import Image

        def image(path):
            value = (
                np.asarray(
                    Image.open(path).convert("RGB").resize((192, 256)), dtype=np.float32
                ).transpose(2, 0, 1)
                / 127.5
                - 1
            )
            return torch.from_numpy(value).unsqueeze(0)

        person = image(ROOT / "experiments/2026-10-23-mobile_vton/inputs/person.jpg")
        clothes = image(ROOT / "experiments/2026-10-23-mobile_vton/inputs/garment.jpg")
        rgb = clothes.numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
        edge = torch.from_numpy(
            (rgb.min(axis=3, keepdims=True) < 0.97)
            .astype(np.float32)
            .transpose(0, 3, 1, 2)
        )
        np.save(args.output / "person.npy", person.numpy())
        np.save(args.output / "clothes.npy", clothes.numpy())
        np.save(args.output / "edge.npy", edge.numpy())
    else:
        clothes = torch.from_numpy(np.load(args.output / "clothes.npy")).float()
        edge = torch.from_numpy(np.load(args.output / "edge.npy")).float()
    warp = WarpExport(pipeline.warp_model).eval()
    generator = GeneratorExport(pipeline.gen_model).eval()
    with torch.inference_mode():
        warped, warped_edge = warp(person, clothes, edge)
        result = generator(person, warped, warped_edge)
    inputs = (person, clothes, edge)
    input_names = ["person", "clothes", "edge"]
    torch.onnx.export(
        warp,
        inputs,
        args.output / "dmvton_warp.onnx",
        input_names=input_names,
        output_names=["warped_cloth", "warped_edge"],
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        generator,
        (person, warped, warped_edge),
        args.output / "dmvton_generator.onnx",
        input_names=["person", "warped_cloth", "warped_edge"],
        output_names=["tryon"],
        opset_version=17,
        dynamo=False,
    )
    for name, value in zip(input_names, inputs):
        value.numpy().tofile(args.output / f"dmvton_input_{name}.f32")
    warped.numpy().tofile(args.output / "dmvton_reference_warped_cloth.f32")
    warped_edge.numpy().tofile(args.output / "dmvton_reference_warped_edge.f32")
    result.numpy().tofile(args.output / "dmvton_reference_tryon.f32")
    report = {
        "status": "exported",
        "correlation": "pure ATen 7x7 mean cost volume replacing upstream CuPy kernel for ONNX export",
        "resolution": [256, 192],
        "inputs": {
            "person": [1, 3, 256, 192],
            "clothes": [1, 3, 256, 192],
            "edge": [1, 1, 256, 192],
        },
        "outputs": {
            "warped_cloth": [1, 3, 256, 192],
            "warped_edge": [1, 1, 256, 192],
            "tryon": [1, 3, 256, 192],
        },
        "model": "DM-VTON PF student",
        "upstream_revision": "5dcd04ee3f5b1fb392df1184c887c1a7e910c72e",
    }
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
