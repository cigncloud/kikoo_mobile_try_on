"""Build a structural INT4/FP16 trial policy from an MNN guide and inventory.

The inventory must describe the original float model, including actual stored
weight counts. This is a structural heuristic, not calibrated layer sensitivity.
Convert with this policy first, then convert the resulting MNN using --fp16:
MNN 3.6.1 leaves existing quantized weights intact and packs excluded weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def build_policy(guide: dict, inventory: list[dict]) -> tuple[dict, dict]:
    weights = {row["name"]: int(row["float_elements"]) for row in inventory}
    decisions = []
    for algo in guide["algo"]:
        if algo["type"] != "QUANTIZE":
            continue
        for layer in algo["quantParams"]["layer"]:
            name = layer["opName"]
            count = weights[name]
            kernel = layer["conv"]["kernelSize"]
            reason = None
            if kernel != [1, 1]:
                reason = "spatial_convolution"
            elif count <= 65536:
                reason = "small_weight_tensor"
            elif "/time_embedding/" in name:
                reason = "time_embedding"
            elif "/attn1/to_k/" in name:
                reason = "self_attention_key"
            for weight in layer["weight"]:
                weight.update(bits=0 if reason else 4, asymmetric=True, blockSize=32)
            decisions.append(
                {
                    "name": name,
                    "weight_elements": count,
                    "storage_target": "fp16" if reason else "int4",
                    "reason": reason or "block32_weight_quantization",
                }
            )
    summary = {
        "method": "structural_heuristic_not_calibrated_sensitivity",
        "int4_block_size": 32,
        "asymmetric": True,
        "storage_layer_counts": dict(Counter(x["storage_target"] for x in decisions)),
        "storage_weight_elements": {
            target: sum(
                x["weight_elements"] for x in decisions if x["storage_target"] == target
            )
            for target in ("fp16", "int4")
        },
        "instructions": "Apply policy to float MNN, then convert result with --fp16. bits=16 is clamped to 8 in MNN 3.6.1; bits=0 excludes a layer from weight quantization.",
        "layers": decisions,
    }
    return guide, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    with args.inventory.open() as stream:
        inventory = list(csv.DictReader(stream, delimiter="\t"))
    policy, summary = build_policy(json.loads(args.guide.read_text()), inventory)
    for path, document in ((args.output, policy), (args.summary, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
