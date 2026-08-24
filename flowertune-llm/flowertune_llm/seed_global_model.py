"""Publish the deterministic round-0 full-model state used by object-store runs.

Run this in a controlled Job with the same warmed HuggingFace snapshot and S3
credentials as the clients. It intentionally refuses LoRA/quantized models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowertune_llm.model_state import get_federated_state_dict
from flowertune_llm.model_storage import ModelStorage
from flowertune_llm.models import get_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--output-dir", default="/scratch")
    args = parser.parse_args()
    if args.round < 0:
        parser.error("--round cannot be negative")
    model_cfg = OmegaConf.create(
        {"name": args.model, "finetuning_type": "full", "quantization": 0,
         "gradient_checkpointing": False}
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "global.pt"
    model = get_model(model_cfg)
    try:
        torch.save(get_federated_state_dict(model, model_cfg), path)
    finally:
        del model
    artifact = ModelStorage.from_env().upload_model(
        path, experiment_id=args.experiment_id, round_number=args.round, role="global"
    )
    print(artifact.uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
