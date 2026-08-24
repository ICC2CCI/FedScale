"""Unit tests for the no-large-ArrayRecord object-store transport path."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from flowertune_llm.aggregation import aggregate_artifacts, fedavg_state_dicts
from flowertune_llm.model_storage import ModelArtifact, ModelStorage, ObjectStoreError, sha256_file
from flowertune_llm.object_store_strategy import (
    ARTIFACT_RECORD_KEY,
    CONTROL_ARRAY_KEY,
    ObjectStoreFedAvg,
)


def artifact(path: Path, role: str, samples: int, round_number: int = 1) -> ModelArtifact:
    return ModelArtifact(
        uri=f"s3://fl-models/demo/round-{round_number}/{role}.pt",
        sha256=sha256_file(path),
        size=path.stat().st_size,
        experiment_id="demo",
        round=round_number,
        role=role,
        num_examples=samples,
    )


class LocalStorage:
    """Filesystem fake with the same narrow API used by aggregation.py."""

    def __init__(self, source: dict[str, Path], root: Path):
        self.source = source
        self.root = root
        self.state = {}

    def download_model(self, model: ModelArtifact, destination: Path) -> Path:
        shutil.copyfile(self.source[model.uri], destination)
        return destination

    def upload_model(self, path: Path, *, experiment_id: str, round_number: int, role: str, num_examples: int = 0) -> ModelArtifact:
        target = self.root / f"{role}.pt"
        shutil.copyfile(path, target)
        uri = f"s3://fl-models/{experiment_id}/round-{round_number}/{role}.pt"
        self.source[uri] = target
        return ModelArtifact(uri, sha256_file(target), target.stat().st_size, experiment_id, round_number, role, num_examples)

    def put_round_state(self, experiment_id, round_number, payload):
        self.state[(experiment_id, round_number)] = payload
        return f"s3://fl-models/{experiment_id}/round-{round_number}/round_state.json"


def test_weighted_fedavg_preserves_dtype_and_buffers():
    a = {"weight": torch.tensor([1.0, 3.0], dtype=torch.float16), "token": torch.tensor([7])}
    b = {"weight": torch.tensor([5.0, 9.0], dtype=torch.float16), "token": torch.tensor([7])}
    result = fedavg_state_dicts([a, b], [1, 3])
    assert result["weight"].dtype == torch.float16
    assert torch.allclose(result["weight"], torch.tensor([4.0, 7.5], dtype=torch.float16))
    assert torch.equal(result["token"], a["token"])


def test_aggregate_artifacts_publishes_global_and_round_state(tmp_path: Path):
    a_path, b_path = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"weight": torch.tensor([1.0], dtype=torch.float16)}, a_path)
    torch.save({"weight": torch.tensor([5.0], dtype=torch.float16)}, b_path)
    a, b = artifact(a_path, "client-a", 1), artifact(b_path, "client-b", 3)
    storage = LocalStorage({a.uri: a_path, b.uri: b_path}, tmp_path)
    global_artifact = aggregate_artifacts(storage, [a, b], experiment_id="demo", round_number=1, workdir=tmp_path / "scratch")
    assert global_artifact.role == "global"
    output = torch.load(storage.source[global_artifact.uri], weights_only=True)
    assert torch.allclose(output["weight"], torch.tensor([4.0], dtype=torch.float16))
    assert storage.state[("demo", 1)]["status"] == "GLOBAL_READY"


def test_aggregate_rejects_mismatched_non_floating_buffers():
    with torch.no_grad():
        a = {"weight": torch.ones(1), "buffer": torch.tensor([1])}
        b = {"weight": torch.ones(1), "buffer": torch.tensor([2])}
    try:
        fedavg_state_dicts([a, b], [1, 1])
    except ObjectStoreError as exc:
        assert "non-floating" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched deterministic buffer was accepted")


class FakeLauncher:
    def __init__(self, result: ModelArtifact):
        self.result = result
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _reply(node_id: int, update: ModelArtifact) -> Message:
    request = Message(
        content=RecordDict({"arrays": ArrayRecord({CONTROL_ARRAY_KEY: torch.zeros(1)})}),
        dst_node_id=node_id,
        message_type="train",
    )
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord({CONTROL_ARRAY_KEY: torch.zeros(1, dtype=torch.uint8)}),
                "metrics": MetricRecord({"num-examples": update.num_examples}),
                ARTIFACT_RECORD_KEY: ConfigRecord({"status": "READY", **update.to_config()}),
            }
        ),
        reply_to=request,
    )


def test_strategy_only_uses_tiny_control_array(tmp_path: Path):
    source = tmp_path / "seed.pt"
    torch.save({"weight": torch.ones(1)}, source)
    initial = artifact(source, "global", 0, round_number=0)
    a, b = artifact(source, "client-a", 1), artifact(source, "client-b", 1)
    result = artifact(source, "global", 0)
    launcher = FakeLauncher(result)
    strategy = ObjectStoreFedAvg(
        storage=object(), launcher=launcher, experiment_id="demo", initial_global=initial,
        fraction_train=1.0, fraction_evaluate=0.0, min_train_nodes=2, min_available_nodes=2,
        train_metrics_aggr_fn=lambda contents, _: MetricRecord({"num-examples": sum(int(item["metrics"]["num-examples"]) for item in contents)}),
    )
    arrays, metrics = strategy.aggregate_train(1, [_reply(1, a), _reply(2, b)])
    assert launcher.calls and launcher.calls[0]["round_number"] == 1
    assert arrays.to_torch_state_dict()[CONTROL_ARRAY_KEY].numel() == 1
    assert metrics["object_store_global_bytes"] == result.size


def test_storage_rejects_non_positive_transport_timeout(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT", "https://minio.example.test")
    monkeypatch.setenv("S3_BUCKET", "fl-models")
    monkeypatch.setenv("S3_ACCESS_KEY", "access")
    monkeypatch.setenv("S3_SECRET_KEY", "secret")
    monkeypatch.setenv("S3_CONNECT_TIMEOUT_SECONDS", "0")
    try:
        ModelStorage.from_env()
    except ObjectStoreError as exc:
        assert "S3_CONNECT_TIMEOUT_SECONDS" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-positive S3 timeout was accepted")
