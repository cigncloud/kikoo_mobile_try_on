"""Check structural policy choices against actual stored weight counts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_mnn_hybrid_policy import build_policy


def test_policy_uses_stored_weights_and_distinguishes_attention_keys():
    specifications = [
        ("depthwise", [3, 3], 1152, "spatial_convolution"),
        ("small", [1, 1], 65536, "small_weight_tensor"),
        ("/time_embedding/linear", [1, 1], 262144, "time_embedding"),
        ("/block/attn1/to_k/conv", [1, 1], 262144, "self_attention_key"),
        ("/block/attn2/to_k/conv", [1, 1], 262144, None),
        ("large", [1, 1], 1048576, None),
    ]
    guide = {
        "version": "test",
        "algo": [
            {"type": "OTHER", "unchanged": True},
            {
                "type": "QUANTIZE",
                "quantParams": {
                    "layer": [
                        {
                            "opName": name,
                            "conv": {"kernelSize": kernel},
                            "weight": [{"bits": 8, "blockSize": -1}],
                        }
                        for name, kernel, _, _ in specifications
                    ]
                },
            },
        ],
    }
    inventory = [
        {"name": name, "float_elements": str(count)}
        for name, _, count, _ in specifications
    ]
    policy, summary = build_policy(guide, inventory)
    assert policy["version"] == "test"
    assert policy["algo"][0] == {"type": "OTHER", "unchanged": True}
    assert summary["storage_layer_counts"] == {"fp16": 4, "int4": 2}
    assert summary["storage_weight_elements"] == {
        "fp16": 1152 + 65536 + 262144 * 2,
        "int4": 262144 + 1048576,
    }
    layers = policy["algo"][1]["quantParams"]["layer"]
    for layer, decision, (_, _, _, reason) in zip(
        layers, summary["layers"], specifications
    ):
        assert layer["weight"][0] == {
            "bits": 0 if reason else 4,
            "blockSize": 32,
            "asymmetric": True,
        }
        assert decision["storage_target"] == ("fp16" if reason else "int4")
        assert decision["reason"] == (reason or "block32_weight_quantization")
    assert summary["method"] == "structural_heuristic_not_calibrated_sensitivity"


def test_policy_rejects_unmatched_inventory_instead_of_guessing():
    guide = {
        "algo": [
            {
                "type": "QUANTIZE",
                "quantParams": {
                    "layer": [
                        {
                            "opName": "missing",
                            "conv": {"kernelSize": [1, 1]},
                            "weight": [{"bits": 8}],
                        }
                    ]
                },
            }
        ]
    }
    with pytest.raises(KeyError, match="missing"):
        build_policy(guide, [])
