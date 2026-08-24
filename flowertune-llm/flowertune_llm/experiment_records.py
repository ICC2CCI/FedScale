"""Durable JSON metadata helpers for federated experiment records."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

from flwr.app import MetricRecord


_ARTIFACT_LOCK = threading.Lock()


def aggregate_train_metrics(records, weighting_metric_name):
    """Aggregate client metrics and retain max-time critical paths."""
    client_metrics = [next(iter(record.metric_records.values())) for record in records]
    weights = [float(metrics[weighting_metric_name]) for metrics in client_metrics]
    total_weight = sum(weights)
    aggregated = MetricRecord()

    if total_weight > 0:
        for key in client_metrics[0]:
            if key == weighting_metric_name:
                continue
            values = [metrics[key] for metrics in client_metrics]
            if all(isinstance(value, (int, float)) for value in values):
                aggregated[key] = sum(
                    value * weight for value, weight in zip(values, weights, strict=True)
                ) / total_weight

    aggregated["total_client_dataset_train_samples"] = total_weight
    for key in (
        "client_training_seconds",
        "client_evaluation_seconds",
        "client_round_seconds",
        "client_non_training_seconds",
    ):
        values = [metrics.get(key) for metrics in client_metrics]
        numeric_values = [
            float(value) for value in values if isinstance(value, (int, float))
        ]
        if numeric_values:
            aggregated[f"critical_path_{key}"] = max(numeric_values)
    return aggregated


def atomic_write_json(path: str, payload) -> None:
    """Atomically publish a JSON artifact on the results PVC."""
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def write_experiment_state(path: str, replace: bool = False, **fields) -> None:
    """Merge and atomically publish supervisor-readable experiment state."""
    with _ARTIFACT_LOCK:
        state = {}
        if not replace:
            try:
                with open(path, encoding="utf-8") as handle:
                    state = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                state = {}
        state.update(fields)
        atomic_write_json(path, state)


def append_experiment_event(path: str, event_type: str, title: str, message: str, **details) -> dict:
    """Append one durable, human-readable progress event to the JSONL stream."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "title": title,
        "message": message,
        "details": json_safe(details),
    }
    with _ARTIFACT_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return event


class ExperimentProgress:
    """Publish phase changes and heartbeats while a ServerApp is waiting."""

    def __init__(self, save_path: str, state_path: str, experiment_id: str, run_id, heartbeat_seconds: float = 30.0):
        self.save_path = save_path
        self.state_path = state_path
        self.events_path = os.path.join(save_path, "experiment_events.jsonl")
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.heartbeat_seconds = max(5.0, float(heartbeat_seconds))
        self._phase = "starting"
        self._message = "ServerApp 已启动"
        self._round = None
        self._details = {}
        self._started_monotonic = time.monotonic()
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _publish_state(self, *, event_timestamp: str | None = None) -> None:
        now = event_timestamp or self._now()
        fields = {
            "status": "running",
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "phase": self._phase,
            "phase_message": self._message,
            "current_round": self._round,
            "heartbeat_at": now,
            "updated_at": now,
        }
        fields.update(self._details)
        write_experiment_state(self.state_path, **fields)

    def phase(self, phase: str, message: str, *, round_number=None, emit: bool = True, **details) -> None:
        self._phase = phase
        self._message = message
        self._round = round_number
        self._details = details
        timestamp = self._now()
        if emit:
            append_experiment_event(
                self.events_path,
                "phase",
                message,
                message,
                phase=phase,
                round=round_number,
                run_id=self.run_id,
                **details,
            )
        self._publish_state(event_timestamp=timestamp)

    def event(self, event_type: str, title: str, message: str, **details) -> None:
        append_experiment_event(
            self.events_path,
            event_type,
            title,
            message,
            phase=self._phase,
            round=self._round,
            run_id=self.run_id,
            **details,
        )

    def heartbeat(self) -> None:
        timestamp = self._now()
        append_experiment_event(
            self.events_path,
            "heartbeat",
            "运行心跳",
            self._message,
            phase=self._phase,
            round=self._round,
            run_id=self.run_id,
            elapsed_seconds=round(time.monotonic() - self._started_monotonic, 3),
        )
        self._publish_state(event_timestamp=timestamp)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="experiment-progress", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                self.heartbeat()
            except OSError:
                # The ServerApp is already terminating or the PVC is briefly unavailable.
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)


def json_safe(value):
    """Convert Flower/OmegaConf metric and config values to JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def write_attempt_artifact(save_path, filename, run_id, payload) -> None:
    """Write both the latest view and an immutable per-Run copy."""
    atomic_write_json(os.path.join(save_path, filename), payload)
    if run_id is not None:
        atomic_write_json(
            os.path.join(save_path, f"run_{run_id}_{filename}"), payload
        )


def upsert_attempt(path, run_id, **fields) -> None:
    """Maintain the experiment's retry/resume attempt history."""
    try:
        with open(path, encoding="utf-8") as handle:
            attempts = json.load(handle)
        if not isinstance(attempts, list):
            attempts = []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        attempts = []

    run_key = str(run_id) if run_id is not None else "unknown"
    for attempt in attempts:
        if str(attempt.get("run_id", "unknown")) == run_key:
            attempt.update(fields)
            break
    else:
        attempts.append({"run_id": run_id, **fields})
    atomic_write_json(path, attempts)


def round_metrics(records, resume_round):
    """Return Result MetricRecords indexed by absolute federated round."""
    return {
        str(int(server_round) + int(resume_round)): json_safe(dict(metrics))
        for server_round, metrics in records.items()
    }
