#!/usr/bin/env python3
"""Rebuild a global PEFT adapter by weighted FedAvg of client state dicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True, type=Path,
                        help="Client model_weights.pt; specify exactly twice.")
    parser.add_argument("--weight", action="append", required=True, type=float,
                        help="Example count corresponding to each --state.")
    parser.add_argument("--adapter-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if len(args.state) != 2 or len(args.weight) != 2:
        parser.error("exactly two --state and two --weight values are required")
    if any(weight <= 0 for weight in args.weight):
        parser.error("weights must be positive")

    states = [torch.load(path, map_location="cpu", weights_only=True) for path in args.state]
    keys = set(states[0])
    if set(states[1]) != keys:
        raise ValueError("client state dict keys differ; refusing to aggregate")

    total = sum(args.weight)
    merged = {}
    for key in sorted(keys):
        left, right = states[0][key], states[1][key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"incompatible tensor for {key}: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}")
        value = (left.float() * args.weight[0] + right.float() * args.weight[1]) / total
        merged[key] = value.to(dtype=left.dtype)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    output_model = output_dir / "adapter_model.bin"
    output_config = output_dir / "adapter_config.json"
    torch.save(merged, output_model)
    output_config.write_bytes(args.adapter_config.read_bytes())
    rebuild_record = {
        "method": "FedAvg",
        "client_states": [str(path) for path in args.state],
        "weights": args.weight,
        "num_tensors": len(merged),
        "adapter_model_sha256": sha256(output_model),
        "adapter_config_sha256": sha256(output_config),
    }
    (output_dir / "rebuild_record.json").write_text(
        json.dumps(rebuild_record, indent=2) + "\n"
    )
    print(json.dumps(rebuild_record, indent=2))


if __name__ == "__main__":
    main()
