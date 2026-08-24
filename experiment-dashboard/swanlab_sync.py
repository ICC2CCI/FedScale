"""Asynchronously publish persisted Flower metrics to a private SwanLab.

The result PVC remains the source of truth.  This adapter only reads the PVC
and sends scalar run metadata/metrics to SwanLab, so a SwanLab outage cannot
stop or modify a federated experiment.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


def _scalar_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten JSON values into SwanLab-compatible scalar metrics."""
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.update(_scalar_metrics(child, child_prefix))
    elif isinstance(value, (bool, int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            result[prefix[:128]] = number
    elif isinstance(value, bool):
        result[prefix[:128]] = float(value)
    return result


def _safe_run_id(experiment_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", experiment_id).strip("-")
    return (value or "flower-run")[:64]


def _dashboard_run_id(experiment_id: str) -> str:
    """Namespace dashboard-owned runs to avoid resuming unrelated old runs."""
    return f"flower-{_safe_run_id(experiment_id)}"[:64]


def _read_json(path: Path, default: Any) -> Any:
    import json

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _upload_worker(connection: Any, payload: dict[str, Any]) -> None:
    """Upload one experiment in a fresh process to isolate SwanLab SDK state."""
    try:
        import swanlab  # type: ignore

        api_key = os.environ["SWANLAB_API_KEY"]
        host = os.environ.get("SWANLAB_WEB_HOST") or os.environ.get(
            "SWANLAB_API_HOST"
        )
        swanlab.login(api_key=api_key, host=host, save=False)
        run = swanlab.init(**payload["run_kwargs"])
        for metrics, step in payload["events"]:
            run.log(metrics, step=step)
        run.finish()
        connection.send({"ok": True})
    except Exception as exc:
        connection.send({"ok": False, "error": str(exc)})
    finally:
        connection.close()


class SwanLabSync:
    """Watch the center result PVC and publish each experiment once per change."""

    def __init__(self, store: Any):
        self.store = store
        self.interval_seconds = max(
            5.0, float(os.environ.get("SWANLAB_SYNC_INTERVAL_SECONDS", "10"))
        )
        self.enabled = os.environ.get("SWANLAB_ENABLED", "false").lower() in {
            "1", "true", "yes", "on"
        }
        # SWANLAB_PROJECT is reserved by the SDK for its nested ``project``
        # settings object. Passing a plain project-name string through that
        # environment variable makes recent SDK settings parsers reject it.
        self.project = os.environ.get(
            "DASHBOARD_SWANLAB_PROJECT", "flower-federated-experiments"
        )
        self.workspace = os.environ.get("SWANLAB_WORKSPACE") or None
        self.per_sync = max(1, int(os.environ.get("SWANLAB_BACKFILL_PER_SYNC", "1")))
        self.worker_timeout_seconds = max(
            10.0, float(os.environ.get("SWANLAB_UPLOAD_TIMEOUT_SECONDS", "120"))
        )
        self._synced_experiments: set[str] = set()
        self._completed_experiments: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_error: str | None = None
        self._last_sync_at: float | None = None
        self._swanlab: Any = None
        self._auth_configured = bool(os.environ.get("SWANLAB_API_KEY"))

        if self.enabled:
            try:
                import swanlab  # type: ignore

                self._swanlab = swanlab
            except Exception as exc:  # optional integration must be fail-open
                self._last_error = f"SwanLab SDK unavailable: {exc}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "sdk_available": self._swanlab is not None,
                "auth_configured": self._auth_configured,
                "project": self.project,
                "synced_experiments": len(self._synced_experiments),
                "last_sync_at": self._last_sync_at,
                "last_error": self._last_error,
            }

    def start(self) -> None:
        if (
            not self.enabled
            or self._swanlab is None
            or not self._auth_configured
            or self._thread is not None
        ):
            return
        self._thread = threading.Thread(
            target=self._loop, name="swanlab-sync", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
    def sync_once(self) -> None:
        """Run one scan; exposed for deterministic unit tests."""
        if not self.enabled or self._swanlab is None or not self._auth_configured:
            return
        current_error: str | None = None
        pending: list[tuple[str, dict[str, Any]]] = []
        active_new: list[tuple[str, dict[str, Any]]] = []
        active_existing: list[tuple[str, dict[str, Any]]] = []
        for item in self.store.list():
            experiment_id = str(item.get("id") or "")
            if not experiment_id:
                continue
            try:
                payload = self._build_payload(experiment_id)
            except Exception as exc:
                current_error = f"{experiment_id}: {exc}"
                continue
            if payload["terminal"]:
                if experiment_id not in self._completed_experiments:
                    pending.append((experiment_id, payload))
            else:
                target = (
                    active_existing
                    if experiment_id in self._synced_experiments
                    else active_new
                )
                target.append((experiment_id, payload))

        # Complete terminal history first, then make sure every running-state
        # directory is represented once before refreshing already-live runs.
        for experiment_id, payload in (
            pending + active_new + active_existing
        )[: self.per_sync]:
            try:
                self._run_upload(payload)
                self._synced_experiments.add(experiment_id)
                if payload["terminal"]:
                    self._completed_experiments.add(experiment_id)
            except Exception as exc:  # never take down the dashboard
                current_error = f"{experiment_id}: {exc}"
        with self._lock:
            # Errors describe the most recent scan. A transient failure must
            # not keep the health endpoint red after a later successful sync.
            self._last_error = current_error
            self._last_sync_at = time.time()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sync_once()
            self._stop.wait(self.interval_seconds)

    def _load(self, experiment_id: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        directory = self.store._experiment_dir(experiment_id)
        config = _read_json(directory / "experiment_config.json", {})
        state = _read_json(directory / "experiment_state.json", {})
        summary = _read_json(directory / "experiment_summary.json", {})
        evaluation = _read_json(directory / "evaluation_summary.json", {})
        return directory, config, state, summary, evaluation

    def _build_payload(self, experiment_id: str) -> dict[str, Any]:
        directory, config, state_file, summary, evaluation = self._load(experiment_id)
        if not isinstance(config, dict):
            raise ValueError("experiment config is not a JSON object")
        run_config = _scalar_metrics(config.get("run_config", {}))
        # Config values are represented as metrics only when numeric. Keep the
        # human-readable identity fields in config as well.
        config_payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "run_id": str(config.get("run_id") or ""),
            "model": str(config.get("model") or ""),
            "dataset": str(config.get("dataset") or ""),
            "strategy": str(config.get("distributed_strategy") or ""),
            "finetuning_type": str(config.get("finetuning_type") or ""),
            **run_config,
        }
        run_kwargs: dict[str, Any] = {
            "mode": "online",
            "project": self.project,
            "name": experiment_id,
            "description": "Flower federated experiment; scalar metrics only",
            "config": config_payload,
            "tags": ["flower", str(config.get("distributed_strategy") or "unknown")],
            "group": "cross-centre-federated-learning",
            "job_type": "federated-training",
            "id": _dashboard_run_id(experiment_id),
            "resume": "allow",
            "log_dir": f"/tmp/swanlab/{_safe_run_id(experiment_id)}",
        }
        if self.workspace:
            run_kwargs["workspace"] = self.workspace

        events: list[tuple[dict[str, float], int]] = []

        round_paths = sorted(directory.glob("federated_metrics_round_*.json"))
        train_by_round = summary.get("aggregated_client_train_metrics", {})
        for path in round_paths:
            record = _read_json(path, {})
            if not isinstance(record, dict):
                continue
            round_number = int(record.get("server_round") or 0)
            metrics = _scalar_metrics(record, "federated")
            train_record = train_by_round.get(str(round_number), {})
            if isinstance(train_record, dict):
                metrics.update(_scalar_metrics(train_record, "client/train"))
            metrics["federated/round"] = float(round_number)
            events.append((metrics, round_number))

        completed_round = max((step for _, step in events), default=0)
        current = _scalar_metrics(state_file if isinstance(state_file, dict) else {}, "run")
        current["run/completed_round"] = float(completed_round)
        current["run/has_summary"] = float(bool(summary))
        if current:
            events.append((current, max(completed_round, 0)))

        if isinstance(evaluation, dict) and evaluation:
            eval_metrics = _scalar_metrics(evaluation, "final_evaluation")
            if eval_metrics:
                events.append((eval_metrics, completed_round + 1))

        status = str(
            (summary or {}).get("status")
            or (state_file or {}).get("status")
            or "running"
        ).lower()
        terminal = status in {"completed", "failed", "stopped", "completed_with_evaluation_failure"}
        if terminal:
            events.append((
                {
                    "run/terminal": 1.0,
                    "run/success": float(status.startswith("completed")),
                },
                completed_round + 2,
            ))
        return {"run_kwargs": run_kwargs, "events": events, "terminal": terminal}

    def _run_upload(self, payload: dict[str, Any]) -> None:
        parent, child = multiprocessing.get_context("spawn").Pipe(duplex=False)
        process = multiprocessing.get_context("spawn").Process(
            target=_upload_worker, args=(child, payload), daemon=True
        )
        process.start()
        child.close()
        if not parent.poll(self.worker_timeout_seconds):
            process.terminate()
            process.join(timeout=5)
            raise TimeoutError("SwanLab upload worker timed out")
        result = parent.recv()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "SwanLab upload failed"))
