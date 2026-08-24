"""One-shot CPU FedAvg Worker for object-store full-model federation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch

from flowertune_llm.model_storage import ModelArtifact, ModelStorage, ObjectStoreError


def _validate_states(states: list[dict[str, torch.Tensor]]) -> tuple[str, ...]:
    if len(states) < 2:
        raise ObjectStoreError("FedAvg requires at least two client model states")
    keys = tuple(states[0].keys())
    if not keys:
        raise ObjectStoreError("client model state is empty")
    expected = set(keys)
    for position, state in enumerate(states[1:], start=1):
        if set(state) != expected:
            raise ObjectStoreError(f"client state {position} has different parameter keys")
    for key in keys:
        reference = states[0][key]
        if not isinstance(reference, torch.Tensor):
            raise ObjectStoreError(f"state {key!r} is not a tensor")
        for position, state in enumerate(states[1:], start=1):
            candidate = state[key]
            if not isinstance(candidate, torch.Tensor):
                raise ObjectStoreError(f"state {position} key {key!r} is not a tensor")
            if tuple(candidate.shape) != tuple(reference.shape):
                raise ObjectStoreError(f"state shape mismatch for {key!r}")
            if candidate.dtype != reference.dtype:
                raise ObjectStoreError(f"state dtype mismatch for {key!r}")
    return keys


def fedavg_state_dicts(
    states: list[dict[str, torch.Tensor]], weights: list[int]
) -> dict[str, torch.Tensor]:
    """Aggregate full state dicts, keeping only one temporary tensor at a time."""
    keys = _validate_states(states)
    if len(states) != len(weights) or any(int(weight) <= 0 for weight in weights):
        raise ObjectStoreError("FedAvg requires one positive sample weight per client")
    weight_total = float(sum(int(weight) for weight in weights))
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key in keys:
            source = states[0][key]
            if source.is_floating_point():
                accumulator = torch.zeros_like(source, dtype=torch.float32, device="cpu")
                for state, weight in zip(states, weights, strict=True):
                    accumulator.add_(state[key].to(device="cpu", dtype=torch.float32), alpha=float(weight) / weight_total)
                result[key] = accumulator.to(dtype=source.dtype)
            else:
                # Buffers such as token IDs cannot be averaged. They must be
                # deterministic and identical across every participating client.
                if any(not torch.equal(source, state[key]) for state in states[1:]):
                    raise ObjectStoreError(f"non-floating tensor differs across clients: {key!r}")
                result[key] = source.detach().clone()
    return result


def aggregate_artifacts(
    storage: ModelStorage,
    updates: Iterable[ModelArtifact],
    *,
    experiment_id: str,
    round_number: int,
    workdir: str | Path,
) -> ModelArtifact:
    """Download verified updates, FedAvg them, publish a new global artifact."""
    updates = list(updates)
    if len(updates) < 2:
        raise ObjectStoreError("aggregation requires updates from both clients")
    roles = {update.role for update in updates}
    if len(roles) != len(updates):
        raise ObjectStoreError("aggregation received duplicate client roles")
    for update in updates:
        if update.experiment_id != experiment_id or update.round != int(round_number):
            raise ObjectStoreError("client artifact does not belong to this aggregation round")
        if update.role == "global":
            raise ObjectStoreError("a global artifact cannot be submitted as a client update")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    local_paths = [
        storage.download_model(update, workdir / f"{update.role}.pt")
        for update in updates
    ]
    # The central cluster has a 30Gi node, while two FP16 OpenLLaMA states are
    # already about 13.7Gi. Memory-map source checkpoints so aggregation only
    # materializes the one output state and the tensor currently being reduced.
    states = [
        torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for path in local_paths
    ]
    if not all(isinstance(state, dict) for state in states):
        raise ObjectStoreError("a client checkpoint is not a tensor state dictionary")
    global_state = fedavg_state_dicts(states, [update.num_examples for update in updates])
    output_path = workdir / "global.pt"
    torch.save(global_state, output_path)
    global_artifact = storage.upload_model(
        output_path,
        experiment_id=experiment_id,
        round_number=round_number,
        role="global",
    )
    storage.put_round_state(
        experiment_id,
        round_number,
        {
            "status": "GLOBAL_READY",
            "experiment_id": experiment_id,
            "round": int(round_number),
            "clients": {update.role: asdict(update) for update in updates},
            "global": asdict(global_artifact),
        },
    )
    return global_artifact


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--updates-json", required=True, help="JSON list of ModelArtifact fields")
    parser.add_argument("--workdir", default="/scratch")
    args = parser.parse_args()
    updates = [ModelArtifact(**item) for item in json.loads(args.updates_json)]
    result = aggregate_artifacts(
        ModelStorage.from_env(),
        updates,
        experiment_id=args.experiment_id,
        round_number=args.round,
        workdir=args.workdir,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
