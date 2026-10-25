"""Verify official file sizes, SHA256 hashes, tensor counts, and stored dtypes."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    files = []
    for entry in manifest["siblings"]:
        name = entry["rfilename"]
        if not name.startswith("checkpoint/") or name.endswith(".DS_Store"):
            continue
        path = args.weights_root / name
        item = {"path": name, "expected_bytes": entry["size"], "exists": path.is_file()}
        if path.is_file():
            item["actual_bytes"] = path.stat().st_size
            item["size_matches"] = item["actual_bytes"] == item["expected_bytes"]
            if "lfs" in entry:
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                item.update(sha256=digest, sha256_matches=digest == entry["lfs"]["sha256"])
            if path.suffix == ".safetensors" and item["size_matches"]:
                with path.open("rb") as handle:
                    length = struct.unpack("<Q", handle.read(8))[0]
                    if length > 16_000_000:
                        raise ValueError(f"Invalid safetensors header length: {length}")
                    header = json.loads(handle.read(length))
                tensors = [value for key, value in header.items() if key != "__metadata__"]
                item["tensor_parameters"] = sum(math.prod(tensor["shape"]) for tensor in tensors)
                item["dtypes"] = sorted({tensor["dtype"] for tensor in tensors})
        files.append(item)
    complete = all(item.get("size_matches", False) and item.get("sha256_matches", True) for item in files)
    report = {
        "model_id": manifest["id"], "revision": manifest["sha"],
        "status": "verified" if complete else "incomplete_or_mismatch",
        "checkpoint_total_bytes": sum(item["expected_bytes"] for item in files),
        "checkpoint_weight_bytes": sum(item["expected_bytes"] for item in files if item["path"].endswith((".safetensors", ".pt"))),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
