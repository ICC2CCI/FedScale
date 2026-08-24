#!/usr/bin/env python3
"""Serve the federated experiment comparison dashboard.

The service deliberately uses only the Python standard library so the same
container can mount the Flower ServerApp results PVC read-only and start
without a package-install step.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import copy
import hashlib
import hmac
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from swanlab_sync import SwanLabSync


APP_DIR = Path(__file__).resolve().parent
STATIC_ROOT = APP_DIR / "static"
DEFAULT_RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "/app/results"))
GPU_MODEL = os.environ.get("GPU_MODEL", "NVIDIA V100")
KUBECTL = os.environ.get("KUBECTL", "kubectl")
PROJECT_ROOT = APP_DIR.parent
FLWR = os.environ.get("FLWR", "flwr")
FLOWER_HOME = Path(os.environ.get("FLOWER_HOME", "/tmp/flower-flwr-home"))
FLOWER_PYTHONPATH = os.environ.get(
    "FLOWER_PYTHONPATH", "/home/fusion/.local/lib/python3.13/site-packages"
)
FLOWER_SUPERLINK = os.environ.get("FLOWER_SUPERLINK", "cross-cloud")
STALE_RUNNING_AFTER_SECONDS = float(os.environ.get("STALE_RUNNING_AFTER_SECONDS", "46800"))


def verify_password_hash(password: str, encoded: str) -> bool:
    """Verify a dashboard PBKDF2 password hash without extra dependencies."""
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iteration_count = int(iterations)
        decode = lambda value: base64.urlsafe_b64decode(
            value.encode("ascii") + b"=" * (-len(value) % 4)
        )
        salt_bytes = decode(salt)
        expected_bytes = decode(expected)
    except (ValueError, TypeError, UnicodeError, binascii.Error):
        return False
    if not 100_000 <= iteration_count <= 2_000_000 or not expected_bytes:
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt_bytes, iteration_count,
        dklen=len(expected_bytes),
    )
    return hmac.compare_digest(actual, expected_bytes)


class DashboardAuth:
    """Optional Basic Auth for the MVP TKE deployment.

    Authentication is enabled only when both environment variables are set.
    Local development can therefore continue without a credential secret.
    """

    def __init__(self) -> None:
        self.username = os.environ.get("DASHBOARD_AUTH_USERNAME", "")
        self.password_hash = os.environ.get("DASHBOARD_AUTH_PASSWORD_HASH", "")
        self.enabled = bool(self.username and self.password_hash)

    def accepts(self, header: str | None) -> bool:
        if not self.enabled:
            return True
        if not header or not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError, binascii.Error):
            return False
        return hmac.compare_digest(username, self.username) and verify_password_hash(
            password, self.password_hash
        )


CENTER_SCAN_SCRIPT = r'''import json, os
root = "/app/results"
result = {}
for name in os.listdir(root) if os.path.isdir(root) else []:
    path = os.path.join(root, name)
    if not os.path.isdir(path) or name == "lost+found":
        continue
    record = {}
    for key, filename in (("config", "experiment_config.json"), ("summary", "experiment_summary.json"), ("state", "experiment_state.json"), ("evaluation", "evaluation_summary.json")):
        try:
            with open(os.path.join(path, filename), encoding="utf-8") as handle:
                record[key] = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            record[key] = {}
    record["federated_rounds"] = []
    try:
        for filename in os.listdir(path):
            if not filename.startswith("federated_metrics_round_") or not filename.endswith(".json"):
                continue
            with open(os.path.join(path, filename), encoding="utf-8") as handle:
                record["federated_rounds"].append(json.load(handle))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    try:
        with open(os.path.join(path, "federated_timings.json"), encoding="utf-8") as handle:
            record["federated_timings"] = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        record["federated_timings"] = []
    if any(record.values()):
        result[name] = record
print(json.dumps(result, ensure_ascii=False))'''


CLIENT_SCAN_SCRIPT = r'''import json, os
root = "/app/outputs"
result = []
for name in os.listdir(root) if os.path.isdir(root) else []:
    path = os.path.join(root, name)
    metrics_path = os.path.join(path, "metrics.json")
    if not os.path.isfile(metrics_path):
        continue
    try:
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (json.JSONDecodeError, OSError):
        continue
    detailed = {}
    try:
        with open(os.path.join(path, "metrics_detailed.json"), encoding="utf-8") as handle:
            detailed = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    result.append({"job_id": name, "metrics": metrics, "detailed": detailed})
print(json.dumps(result, ensure_ascii=False))'''


CENTER_LOG_SCRIPT = r'''import json, os
root = "/app/results"
experiment_id = __EXPERIMENT_ID__
path = os.path.join(root, experiment_id)
record = {"config": {}, "state": {}, "summary": {}, "attempts": [], "rounds": [], "events": []}
for key, filename, default in (
    ("config", "experiment_config.json", {}),
    ("state", "experiment_state.json", {}),
    ("summary", "experiment_summary.json", {}),
    ("attempts", "experiment_attempts.json", []),
):
    try:
        with open(os.path.join(path, filename), encoding="utf-8") as handle:
            record[key] = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        record[key] = default
if os.path.isdir(path):
    for filename in sorted(os.listdir(path)):
        if not filename.startswith("federated_metrics_round_") or not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(path, filename), encoding="utf-8") as handle:
                record["rounds"].append({"artifact": filename, "metrics": json.load(handle)})
        except (json.JSONDecodeError, OSError):
            pass
    try:
        with open(os.path.join(path, "experiment_events.jsonl"), encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    record["events"].append(event)
    except (FileNotFoundError, OSError):
        pass
print(json.dumps(record, ensure_ascii=False))'''


SUPERVISOR_CONTROL_WRITE_SCRIPT = r'''import json, os, tempfile
root = "/app/results"
path = os.path.join(root, ".supervisor-control.json")
payload = __PAYLOAD__
fd, temporary = tempfile.mkstemp(prefix=".supervisor-control.", dir=root)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
print(json.dumps({"ok": True, "control": payload}, ensure_ascii=False))'''


SUPERVISOR_STATUS_SCRIPT = r'''import json, os
root = "/app/results"
def read_json(name, default):
    try:
        with open(os.path.join(root, name), encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default
events = []
try:
    with open(os.path.join(root, ".supervisor-events.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
except (FileNotFoundError, OSError):
    pass
print(json.dumps({"control": read_json(".supervisor-control.json", {}),
                  "status": read_json(".supervisor-status.json", {}),
                  "events": events[-200:]}, ensure_ascii=False))'''


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def load_federated_artifacts(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rounds = []
    for path in sorted(directory.glob("federated_metrics_round_*.json")):
        payload = load_json(path, {})
        if isinstance(payload, dict):
            rounds.append(payload)
    timings = load_json(directory / "federated_timings.json", [])
    return rounds, timings if isinstance(timings, list) else []


def nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def positive_number(value: Any) -> float | None:
    number = finite_number(value)
    return number if number is not None and number > 0 else None


def normalize_quality_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Flatten one formal checkpoint evaluation for dashboard comparison."""
    results = evaluation.get("results", []) if isinstance(evaluation, dict) else []
    result = results[-1] if results and isinstance(results[-1], dict) else {}
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    generation = metrics.get("generation_quality", {})
    return {
        "validation_loss": finite_number(nested(metrics, "assistant_only.loss")),
        "perplexity": finite_number(nested(metrics, "assistant_only.ppl")),
        "rouge_l_f1": finite_number(nested(metrics, "rouge_l.f1")),
        "bertscore_f1": finite_number(nested(metrics, "bertscore.f1")),
        "accuracy": finite_number(generation.get("accuracy")),
        "macro_f1": finite_number(generation.get("macro_f1")),
        "exact_match": finite_number(generation.get("exact_match")),
        "empty_prediction_rate": finite_number(
            generation.get("empty_prediction_rate")
        ),
        "average_generated_tokens": finite_number(
            generation.get("average_generated_tokens")
        ),
        "evaluated_samples": finite_number(
            nested(metrics, "assistant_only.evaluated_samples")
        ),
    }


def identifier(value: Any) -> str | None:
    """Return large Flower identifiers without losing precision in JavaScript."""
    return None if value in (None, "") else str(value)


def status_label(value: Any) -> str:
    return {
        "completed": "已完成",
        "completed_with_evaluation_failure": "训练完成但最终评估失败",
        "failed": "失败",
        "stopped": "已停止",
        "running": "运行中",
        "starting": "启动中",
        "pending": "等待中",
    }.get(str(value).lower(), str(value or "未知"))


def build_log_payload(
    experiment_id: str,
    config: Any,
    state: Any,
    summary: Any,
    attempts: Any,
    round_records: Any,
    progress_events: Any = None,
    supervisor: Any = None,
) -> dict[str, Any]:
    """Build the shared event representation for local or cloud artifacts."""
    if not any((config, state, summary)):
        raise ApiError(
            HTTPStatus.NOT_FOUND,
            f"实验没有可读取的运行记录：{experiment_id}",
        )

    events: list[dict[str, Any]] = []

    def append_event(
        timestamp: Any,
        level: str,
        title: str,
        message: str,
        **details: Any,
    ) -> None:
        events.append({
            "timestamp": timestamp or "",
            "level": level,
            "title": title,
            "message": message,
            "details": details,
        })

    if isinstance(config, dict):
        append_event(
            config.get("started_at"), "info", "实验已提交",
            f"Run {identifier(config.get('run_id')) or '—'} 已按当前配置启动。",
            run_id=identifier(config.get("run_id")),
            strategy=str(config.get("distributed_strategy", "")).upper(),
            total_rounds=config.get("rounds_this_attempt"),
        )

    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_status = str(attempt.get("status", "unknown"))
            timestamp = attempt.get("finished_at") or attempt.get("started_at")
            append_event(
                timestamp,
                "error" if attempt_status == "failed" else
                "success" if attempt_status == "completed" else "info",
                f"运行尝试：{status_label(attempt_status)}",
                f"Run {identifier(attempt.get('run_id')) or '—'}；从第 "
                f"{attempt.get('resume_round', 0)} 轮开始，计划运行 "
                f"{attempt.get('rounds_this_attempt', '—')} 轮。",
                run_id=identifier(attempt.get("run_id")),
                status=attempt_status,
            )

    if isinstance(round_records, list):
        for record in round_records:
            if not isinstance(record, dict):
                continue
            metrics = record.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            artifact = str(record.get("artifact", ""))
            round_number = metrics.get("server_round")
            if round_number is None:
                match = re.search(r"_(\d+)\.json$", artifact)
                round_number = int(match.group(1)) if match else "—"
            duration = finite_number(metrics.get("t_round_total_s"))
            duration_text = (
                f"，聚合端轮次耗时 {duration:.2f} s"
                if duration is not None else ""
            )
            title = (
                "初始模型就绪"
                if round_number == 0 else f"第 {round_number} 轮完成"
            )
            message = (
                "已写入联邦初始化指标。"
                if round_number == 0 else f"已写入联邦聚合指标{duration_text}。"
            )
            append_event(
                metrics.get("timestamp"), "round", title, message,
                round=round_number,
                duration_seconds=duration,
                artifact=artifact,
            )

    if isinstance(state, dict) and state:
        state_status = str(state.get("status", "unknown"))
        state_round = max(
            int(finite_number(state.get(key)) or 0)
            for key in ("latest_completed_round", "completed_global_round", "resume_round")
        )
        artifact_round = max(
            (
                int(finite_number(record.get("metrics", {}).get("server_round")) or 0)
                for record in round_records
                if isinstance(record, dict) and isinstance(record.get("metrics"), dict)
            ),
            default=0,
        ) if isinstance(round_records, list) else 0
        completed_round = max(state_round, artifact_round)
        append_event(
            state.get("updated_at"),
            "error" if state_status == "failed" else
            "success" if state_status == "completed" else "info",
            f"当前状态：{status_label(state_status)}",
            f"已完成至第 {completed_round} 轮，共 {state.get('total_rounds', '—')} 轮。",
            status=state_status,
            run_id=identifier(state.get("run_id")),
        )

    if isinstance(summary, dict) and summary:
        summary_status = str(summary.get("status", "completed"))
        duration = finite_number(summary.get("duration_seconds"))
        duration_text = (
            f"，端到端耗时 {duration:.2f} s" if duration is not None else ""
        )
        append_event(
            summary.get("finished_at") or summary.get("updated_at"),
            "error" if summary_status == "failed" else "success",
            f"实验汇总：{status_label(summary_status)}",
            f"已完成 {summary.get('completed_global_round', '—')} 个全局轮次"
            f"{duration_text}。",
            status=summary_status,
            run_id=identifier(summary.get("run_id")),
        )

    if isinstance(progress_events, list):
        for raw_event in progress_events:
            if not isinstance(raw_event, dict):
                continue
            event_type = str(raw_event.get("type", "info"))
            details = raw_event.get("details", {})
            if not isinstance(details, dict):
                details = {}
            append_event(
                raw_event.get("timestamp"),
                "heartbeat" if event_type == "heartbeat" else
                "round" if event_type == "round" else
                "error" if event_type == "error" else "info",
                raw_event.get("title", "运行进度"),
                raw_event.get("message", "—"),
                event_type=event_type,
                **details,
            )

    if isinstance(supervisor, dict):
        supervisor_events = supervisor.get("events", [])
        run_id = identifier(config.get("run_id")) if isinstance(config, dict) else None
        for raw_event in supervisor_events:
            if not isinstance(raw_event, dict):
                continue
            details = raw_event.get("details", {})
            if not isinstance(details, dict):
                details = {}
            event_experiment = details.get("experiment_id")
            event_run = identifier(details.get("run_id"))
            if event_experiment not in (None, "", experiment_id) and event_run != run_id:
                continue
            append_event(
                raw_event.get("timestamp"),
                "heartbeat" if raw_event.get("type") in {"heartbeat", "poll"} else
                "error" if raw_event.get("type") == "error" else "info",
                raw_event.get("title", "监督器事件"),
                raw_event.get("message", "—"),
                supervisor_event=True,
                **details,
            )

    events.sort(key=lambda event: str(event.get("timestamp") or ""), reverse=True)
    return {
        "experiment_id": experiment_id,
        "status": state.get("status") if isinstance(state, dict) else "unknown",
        "updated_at": state.get("updated_at") if isinstance(state, dict) else None,
        "progress": {
            key: state.get(key)
            for key in ("phase", "phase_message", "current_round", "heartbeat_at", "total_rounds")
            if isinstance(state, dict) and state.get(key) is not None
        },
        "events": events,
        "supervisor": supervisor or {},
    }


def round_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("aggregated_client_train_metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return {}
    numeric_rounds: list[tuple[int, dict[str, Any]]] = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            try:
                numeric_rounds.append((int(key), value))
            except (TypeError, ValueError):
                continue
    return max(numeric_rounds, default=(0, {}), key=lambda item: item[0])[1]


@dataclass(frozen=True)
class ExperimentStore:
    root: Path

    def _experiment_dir(self, experiment_id: str) -> Path:
        if not experiment_id or experiment_id in {".", ".."}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "实验 ID 不能为空")
        candidate = (self.root / experiment_id).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "实验 ID 非法") from exc
        if candidate.parent != self.root.resolve() or not candidate.is_dir():
            raise ApiError(HTTPStatus.NOT_FOUND, f"未找到实验：{experiment_id}")
        return candidate

    def read(self, experiment_id: str) -> dict[str, Any]:
        directory = self._experiment_dir(experiment_id)
        config = load_json(directory / "experiment_config.json", {})
        summary = load_json(directory / "experiment_summary.json", {})
        state = load_json(directory / "experiment_state.json", {})
        evaluation = load_json(directory / "evaluation_summary.json", {})
        federated_rounds, federated_timings = load_federated_artifacts(directory)
        if not any((config, summary, state)):
            raise ApiError(HTTPStatus.NOT_FOUND, f"实验没有可读取的结果：{experiment_id}")
        return normalize_experiment(
            experiment_id,
            config,
            summary,
            state,
            evaluation,
            federated_rounds,
            federated_timings,
        )

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        experiments: list[dict[str, Any]] = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.name == "lost+found":
                continue
            if not any(
                (directory / filename).is_file()
                for filename in (
                    "experiment_config.json",
                    "experiment_summary.json",
                    "experiment_state.json",
                )
            ):
                continue
            try:
                detail = self.read(directory.name)
            except ApiError:
                continue
            experiments.append(
                {
                    "id": detail["id"],
                    "strategy": detail["strategy"],
                    "status": detail["status"],
                    "model": detail["model"],
                    "started_at": detail["started_at"],
                    "run_id": detail["run_id"],
                    "has_summary": detail["has_summary"],
                }
            )
        experiments.sort(key=lambda item: item.get("started_at") or "", reverse=True)
        return experiments

    def logs(self, experiment_id: str) -> dict[str, Any]:
        """Build a chronological, read-only event stream for one experiment.

        ServerApp persists structured progress rather than its stdout.  Turning
        those files into events keeps the dashboard useful for both running and
        completed runs, without requiring write access to the results PVC.
        """
        directory = self._experiment_dir(experiment_id)
        config = load_json(directory / "experiment_config.json", {})
        state = load_json(directory / "experiment_state.json", {})
        summary = load_json(directory / "experiment_summary.json", {})
        attempts = load_json(directory / "experiment_attempts.json", [])
        round_records = [
            {"artifact": path.name, "metrics": load_json(path, {})}
            for path in sorted(
                directory.glob("federated_metrics_round_*.json"),
                key=lambda item: item.name,
            )
            ]
        progress_events = []
        events_path = directory / "experiment_events.jsonl"
        if events_path.is_file():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    progress_events.append(value)
        return build_log_payload(
            experiment_id,
            config,
            state,
            summary,
            attempts,
            round_records,
            progress_events,
        )


@dataclass(frozen=True)
class ClusterTarget:
    name: str
    kubeconfig: Path
    namespace: str
    deployment: str
    site: str


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def experiment_list_item(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": detail["id"],
        "strategy": detail["strategy"],
        "status": detail["status"],
        "model": detail["model"],
        "started_at": detail["started_at"],
        "run_id": detail["run_id"],
        "has_summary": detail["has_summary"],
        "cloud_client_count": len(detail.get("cloud", {}).get("clients", [])),
        "status_source": detail.get("status_source", "result-file"),
        "recorded_status": detail.get("recorded_status", detail.get("status")),
        "flower_status": detail.get("flower_status"),
    }


def canonical_flower_status(value: Any) -> str | None:
    status = str(value or "").lower()
    mapping = {
        "finished:completed": "completed",
        "finished:failed": "failed",
        "finished:stopped": "stopped",
        "pending": "pending",
        "starting": "starting",
        "running": "running",
    }
    return mapping.get(status)


def reconcile_experiment_status(
    detail: dict[str, Any],
    live_run: dict[str, Any] | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Prefer Flower's Run state; downgrade abandoned result-file running states."""
    recorded = str(detail.get("status") or "unknown").lower()
    detail["recorded_status"] = recorded
    if live_run:
        flower_status = live_run.get("status")
        canonical = canonical_flower_status(flower_status)
        detail["flower_status"] = flower_status
        if canonical:
            detail["status"] = canonical
            detail["status_source"] = "flower-run"
            if live_run.get("finished-at") not in (None, "N/A"):
                detail["finished_at"] = live_run.get("finished-at")
            return detail

    if recorded == "running":
        started = parse_timestamp(detail.get("started_at"))
        current = now or datetime.now(timezone.utc)
        if started is None or (current - started).total_seconds() > STALE_RUNNING_AFTER_SECONDS:
            detail["status"] = "stale"
            detail["status_source"] = "stale-timeout"
            return detail
    detail["status_source"] = "result-file"
    return detail


CONTROL_DEFAULTS = {
    "desired_state": "running",
    "poll_seconds": 120,
    "stall_seconds": 7200,
    "max_restarts": 3,
    "matrix_id": "",
    "strategy": "fedscale",
    "rounds": 10,
    "model": "Qwen/Qwen2.5-7B",
    "dataset": "HuggingFaceH4/ultrachat_200k",
    "finetuning_type": "lora",
}

SUPERVISOR_STRATEGIES = {"fsdp", "fedscale", "ddp"}
SUPERVISOR_FINETUNING_TYPES = {"lora", "full"}
HF_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)


def normalize_supervisor_control(current: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Validate the small, durable control contract shared with the supervisor."""
    if not isinstance(request, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "控制请求必须是 JSON 对象")
    action = str(request.get("action", "configure")).lower()
    if action not in {"configure", "start", "resume", "pause", "stop", "retry"}:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"不支持的监督器操作：{action}")
    control = dict(CONTROL_DEFAULTS)
    control.update(current if isinstance(current, dict) else {})
    for key, lower, upper in (
        ("poll_seconds", 5, 3600),
        ("stall_seconds", 60, 86400),
        ("max_restarts", 0, 10),
        ("rounds", 1, 1000),
    ):
        if key in request:
            try:
                value = int(request[key])
            except (TypeError, ValueError) as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} 必须是整数") from exc
            if not lower <= value <= upper:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} 必须在 {lower} 到 {upper} 之间")
            control[key] = value
    if "strategy" in request:
        strategy = str(request["strategy"]).strip().lower()
        if strategy not in SUPERVISOR_STRATEGIES:
            raise ApiError(HTTPStatus.BAD_REQUEST, "strategy 必须是 fsdp、fedscale 或 ddp")
        control["strategy"] = strategy
    if "finetuning_type" in request:
        finetuning_type = str(request["finetuning_type"]).strip().lower()
        if finetuning_type not in SUPERVISOR_FINETUNING_TYPES:
            raise ApiError(HTTPStatus.BAD_REQUEST, "finetuning_type 必须是 lora 或 full")
        control["finetuning_type"] = finetuning_type
    for key, label in (("model", "模型"), ("dataset", "数据集")):
        if key not in request:
            continue
        value = str(request[key]).strip()
        if not HF_REPOSITORY_PATTERN.fullmatch(value):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{label}必须是 Hugging Face 仓库名，例如 Qwen/Qwen2.5-7B",
            )
        control[key] = value
    if "matrix_id" in request:
        matrix_id = str(request["matrix_id"]).strip()
        if matrix_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", matrix_id):
            raise ApiError(HTTPStatus.BAD_REQUEST, "matrix_id 格式非法")
        control["matrix_id"] = matrix_id
    if action in {"start", "resume", "retry"}:
        control["desired_state"] = "running"
    elif action == "pause":
        control["desired_state"] = "paused"
    elif action == "stop":
        control["desired_state"] = "stopped"
    else:
        desired_state = str(request.get("desired_state", control.get("desired_state", "running"))).lower()
        if desired_state not in {"running", "paused", "stopped"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "desired_state 必须是 running、paused 或 stopped")
        control["desired_state"] = desired_state
    try:
        generation = int(control.get("generation", 0) or 0) + 1
    except (TypeError, ValueError):
        generation = 1
    control.update({
        "generation": generation,
        "last_action": action,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    })
    if action == "retry":
        control["retry_token"] = generation
    return control


class KubectlCloudStore:
    """Read center summaries and per-client metrics directly from three clusters."""

    def __init__(self, ttl_seconds: float = 30.0):
        self.ttl_seconds = ttl_seconds
        self.exec_retries = max(1, int(os.environ.get("CLOUD_EXEC_RETRIES", "3")))
        self.exec_timeout_seconds = max(
            5.0, float(os.environ.get("CLOUD_EXEC_TIMEOUT_SECONDS", "25"))
        )
        self.center = ClusterTarget(
            "center",
            Path(os.environ.get("CENTER_KUBECONFIG", PROJECT_ROOT / "config-center")),
            os.environ.get("CENTER_NAMESPACE", "flower-superlink"),
            os.environ.get("CENTER_DEPLOYMENT", "superexec-serverapp"),
            "CENTER",
        )
        self.clients = (
            ClusterTarget(
                "tke-a",
                Path(os.environ.get("TKE_A_KUBECONFIG", PROJECT_ROOT / "config-tke-a")),
                os.environ.get("TKE_A_NAMESPACE", "flower-supernode-a"),
                os.environ.get("TKE_A_DEPLOYMENT", "superexec-clientapp-a"),
                "TKE-A",
            ),
            ClusterTarget(
                "tke-b",
                Path(os.environ.get("TKE_B_KUBECONFIG", PROJECT_ROOT / "config-tke-b")),
                os.environ.get("TKE_B_NAMESPACE", "flower-supernode-b"),
                os.environ.get("TKE_B_DEPLOYMENT", "superexec-clientapp-b"),
                "TKE-B",
            ),
        )
        self._lock = threading.Lock()
        self._loaded_at = 0.0
        self._experiments: dict[str, dict[str, Any]] = {}

    def availability(self) -> dict[str, Any]:
        kubectl_path = shutil.which(KUBECTL) if os.path.sep not in KUBECTL else KUBECTL
        missing = [str(target.kubeconfig) for target in (self.center, *self.clients) if not target.kubeconfig.is_file()]
        try:
            import kubernetes  # type: ignore  # noqa: F401
            client_available = True
        except ImportError:
            client_available = False
        available = (bool(kubectl_path) or client_available) and not missing
        message = "可直接读取中心及 TKE A/B 结果卷" if available else "缺少 Kubernetes 客户端或云端 kubeconfig"
        return {"available": available, "message": message, "missing_kubeconfigs": missing}

    def _kubectl_json_once(self, target: ClusterTarget, script: str) -> Any:
        try:
            from kubernetes import client, config  # type: ignore
            from kubernetes.stream import stream  # type: ignore

            config.load_kube_config(
                config_file=str(target.kubeconfig), persist_config=False
            )
            core = client.CoreV1Api()
            apps = client.AppsV1Api()
            deployment = apps.read_namespaced_deployment(
                target.deployment, target.namespace
            )
            selector = ",".join(
                f"{key}={value}"
                for key, value in (deployment.spec.selector.match_labels or {}).items()
            )
            pods = core.list_namespaced_pod(
                target.namespace, label_selector=selector
            ).items
            pod = next(
                (item for item in pods if item.status.phase == "Running"), None
            )
            if pod is None:
                raise ApiError(
                    HTTPStatus.BAD_GATEWAY,
                    f"读取 {target.site} 失败：deployment 没有运行中的 Pod",
                )
            container_name = deployment.spec.template.spec.containers[0].name
            output = stream(
                core.connect_get_namespaced_pod_exec,
                pod.metadata.name,
                target.namespace,
                command=["python3", "-c", script],
                container=container_name,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _request_timeout=self.exec_timeout_seconds,
            )
            return json.loads(output)
        except ImportError:
            pass
        except ApiError:
            raise
        except json.JSONDecodeError:
            # The Kubernetes Python client's websocket exec path can stringify
            # a JSON response as a Python dict repr (single quotes).  Keep the
            # kubectl path below as a compatible fallback, but accept the
            # representation directly when it is a safe literal.
            try:
                parsed = ast.literal_eval(output)
            except (SyntaxError, ValueError):
                pass
            else:
                if isinstance(parsed, (dict, list)):
                    return parsed
        except Exception as exc:
            # Fall back to kubectl for the development image, while production
            # uses the Python client and therefore needs no kubectl binary.
            if not (shutil.which(KUBECTL) or os.path.isfile(KUBECTL)):
                raise ApiError(HTTPStatus.BAD_GATEWAY, f"读取 {target.site} 失败：{exc}") from exc

        environment = os.environ.copy()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            environment.pop(key, None)
        command = [
            KUBECTL,
            "--kubeconfig",
            str(target.kubeconfig),
            "-n",
            target.namespace,
            "exec",
            f"deployment/{target.deployment}",
            "--",
            "python3",
            "-c",
            script,
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.exec_timeout_seconds,
                env=environment,
            )
            return json.loads(completed.stdout)
        except FileNotFoundError as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "服务器未安装 kubectl，无法读取云端结果") from exc
        except subprocess.TimeoutExpired as exc:
            raise ApiError(HTTPStatus.GATEWAY_TIMEOUT, f"读取 {target.site} 结果超时") from exc
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or "kubectl exec 失败").strip().splitlines()[-1]
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"读取 {target.site} 失败：{message}") from exc
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"{target.site} 返回了无效 JSON") from exc

    def _kubectl_json(self, target: ClusterTarget, script: str) -> Any:
        """Read one remote artifact, retrying transient API/websocket resets.

        Every attempt discovers the deployment pod again.  This matters for TKE
        API-server connection resets and for a pod replacement between retries.
        """
        last_error: ApiError | None = None
        for attempt in range(1, self.exec_retries + 1):
            try:
                return self._kubectl_json_once(target, script)
            except ApiError as exc:
                last_error = exc
                if attempt < self.exec_retries:
                    time.sleep(0.5 * attempt)
        assert last_error is not None
        if self.exec_retries == 1:
            raise last_error
        raise ApiError(
            last_error.status,
            f"{last_error.message}（已重试 {self.exec_retries} 次）",
        ) from last_error

    def _flower_run_statuses(self) -> dict[str, dict[str, Any]]:
        """Read authoritative Run states when a Flower CLI connection is available."""
        flwr_path = shutil.which(FLWR) if os.path.sep not in FLWR else FLWR
        config_path = FLOWER_HOME / ".flwr" / "config.toml"
        if not flwr_path or not config_path.is_file():
            return {}
        environment = os.environ.copy()
        environment["HOME"] = str(FLOWER_HOME)
        environment["PYTHONPATH"] = FLOWER_PYTHONPATH
        try:
            completed = subprocess.run(
                [FLWR, "list", FLOWER_SUPERLINK, "--limit", "100", "--format", "json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
                env=environment,
            )
            payload = json.loads(completed.stdout)
            if not payload.get("success"):
                return {}
            return {
                str(run.get("run-id")): run
                for run in payload.get("runs", [])
                if run.get("run-id") is not None
            }
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
            return {}

    def _load(self, force: bool = False) -> None:
        if not self.availability()["available"]:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, self.availability()["message"])
        with self._lock:
            if not force and self._experiments and time.monotonic() - self._loaded_at < self.ttl_seconds:
                return
            center_records = self._kubectl_json(self.center, CENTER_SCAN_SCRIPT)
            live_runs = self._flower_run_statuses()
            client_records: list[dict[str, Any]] = []
            client_errors: dict[str, str] = {}
            for target in self.clients:
                try:
                    records = self._kubectl_json(target, CLIENT_SCAN_SCRIPT)
                except ApiError as exc:
                    # Client metrics are supplemental.  A temporary failure in
                    # one TKE must not hide center results or the other client.
                    client_errors[target.site] = exc.message
                    continue
                for record in records:
                    record["site"] = target.site
                    client_records.append(record)

            experiments: dict[str, dict[str, Any]] = {}
            for experiment_id, record in center_records.items():
                detail = normalize_experiment(
                    experiment_id,
                    record.get("config", {}),
                    record.get("summary", {}),
                    record.get("state", {}),
                    record.get("evaluation", {}),
                    record.get("federated_rounds", []),
                    record.get("federated_timings", []),
                )
                detail = reconcile_experiment_status(
                    detail, live_runs.get(str(detail.get("run_id")))
                )
                detail["cloud"] = {
                    "clients": match_cloud_records(detail, client_records),
                    "source": "TKE A/B /app/outputs",
                    "read_errors": client_errors,
                }
                experiments[experiment_id] = detail
            self._experiments = experiments
            self._loaded_at = time.monotonic()

    def list(self, force: bool = False) -> list[dict[str, Any]]:
        self._load(force=force)
        experiments = [experiment_list_item(detail) for detail in self._experiments.values()]
        experiments.sort(key=lambda item: item.get("started_at") or "", reverse=True)
        return experiments

    def read(self, experiment_id: str) -> dict[str, Any]:
        self._load()
        if experiment_id not in self._experiments:
            raise ApiError(HTTPStatus.NOT_FOUND, f"云端未找到实验：{experiment_id}")
        return self._experiments[experiment_id]

    def logs(self, experiment_id: str) -> dict[str, Any]:
        """Read one experiment's structured log artifacts from the center PVC."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", experiment_id):
            raise ApiError(HTTPStatus.BAD_REQUEST, "experiment-id 格式非法")
        availability = self.availability()
        if not availability["available"]:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, availability["message"])
        script = CENTER_LOG_SCRIPT.replace(
            "__EXPERIMENT_ID__",
            json.dumps(experiment_id, ensure_ascii=True),
        )
        record = self._kubectl_json(self.center, script)
        if not isinstance(record, dict):
            raise ApiError(HTTPStatus.BAD_GATEWAY, "中心端返回了无效实验日志")
        return build_log_payload(
            experiment_id,
            record.get("config", {}),
            record.get("state", {}),
            record.get("summary", {}),
            record.get("attempts", []),
            record.get("rounds", []),
            record.get("events", []),
            self.supervisor_snapshot(),
        )

    def supervisor_snapshot(self) -> dict[str, Any]:
        availability = self.availability()
        if not availability["available"]:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, availability["message"])
        payload = self._kubectl_json(self.center, SUPERVISOR_STATUS_SCRIPT)
        snapshot = payload if isinstance(payload, dict) else {}
        job = self._latest_supervisor_job()
        file_status = snapshot.get("status") if isinstance(snapshot.get("status"), dict) else {}
        if job:
            snapshot["job"] = job
            # A completed/failed Kubernetes Job is authoritative when the
            # mounted status file is missing or still says running. This keeps
            # a stale control intent (desired_state=running) from looking like
            # a live supervisor.
            if not file_status or job["job_phase"] in {"succeeded", "failed"}:
                snapshot["status"] = {
                    **file_status,
                    "phase": job["phase"],
                    "message": job["message"],
                    "job_name": job["job_name"],
                    "job_phase": job["job_phase"],
                    "updated_at": job.get("updated_at"),
                }
        elif not file_status:
            snapshot["status"] = {
                "phase": "not_started",
                "message": "监督器 Job 尚未启动，请点击“启动监督器”。",
            }
        return snapshot

    def _latest_supervisor_job(self) -> dict[str, Any]:
        """Return the newest supervisor Job state from the center cluster."""
        try:
            from kubernetes import client, config  # type: ignore

            config.load_kube_config(
                config_file=str(self.center.kubeconfig), persist_config=False
            )
            jobs = client.BatchV1Api().list_namespaced_job(self.center.namespace).items
        except Exception:
            # The file protocol remains useful when the API server temporarily
            # refuses Job metadata; callers still receive the raw snapshot.
            return {}

        candidates = [
            job for job in jobs
            if str(getattr(job.metadata, "name", "")).startswith("benchmark-matrix-supervisor")
        ]
        if not candidates:
            return {}
        job = max(
            candidates,
            key=lambda item: getattr(item.metadata, "creation_timestamp", None)
            or datetime.min.replace(tzinfo=timezone.utc),
        )
        status = job.status
        active = int(getattr(status, "active", 0) or 0)
        succeeded = int(getattr(status, "succeeded", 0) or 0)
        failed = int(getattr(status, "failed", 0) or 0)
        if active > 0:
            phase = "starting"
            job_phase = "active"
            message = "监督器 Job 正在运行，等待它写入详细状态。"
        elif succeeded > 0:
            phase = "completed"
            job_phase = "succeeded"
            message = "监督器 Job 已完成；如需继续调度，请点击“启动监督器”。"
        elif failed > 0:
            phase = "failed"
            job_phase = "failed"
            message = "监督器 Job 已失败，请检查 Job 日志后重新启动。"
        else:
            phase = "starting"
            job_phase = "pending"
            message = "监督器 Job 已创建，正在等待调度。"
        created_at = getattr(job.metadata, "creation_timestamp", None)
        return {
            "job_name": job.metadata.name,
            "job_phase": job_phase,
            "phase": phase,
            "message": message,
            "active": active,
            "succeeded": succeeded,
            "failed": failed,
            "updated_at": created_at.isoformat() if created_at else None,
        }

    def start_supervisor(self) -> dict[str, Any]:
        """Create a fresh supervisor Job from the latest center Job template."""
        try:
            from kubernetes import client, config  # type: ignore

            config.load_kube_config(
                config_file=str(self.center.kubeconfig), persist_config=False
            )
            batch = client.BatchV1Api()
            jobs = batch.list_namespaced_job(self.center.namespace).items
            candidates = [
                job for job in jobs
                if str(getattr(job.metadata, "name", "")).startswith("benchmark-matrix-supervisor")
            ]
            active = [
                job for job in candidates
                if int(getattr(job.status, "active", 0) or 0) > 0
                or (
                    int(getattr(job.status, "succeeded", 0) or 0) == 0
                    and int(getattr(job.status, "failed", 0) or 0) == 0
                )
            ]
            if active:
                name = active[0].metadata.name
                raise ApiError(HTTPStatus.CONFLICT, f"监督器已经在运行：{name}")
            if not candidates:
                raise ApiError(HTTPStatus.NOT_FOUND, "没有找到可复用的监督器 Job 模板")
            template_source = max(
                candidates,
                key=lambda item: getattr(item.metadata, "creation_timestamp", None)
                or datetime.min.replace(tzinfo=timezone.utc),
            )
            template = copy.deepcopy(template_source.spec.template)
            template.metadata = client.V1ObjectMeta(
                labels={"app": "benchmark-matrix-supervisor"}
            )
            job = client.V1Job(
                metadata=client.V1ObjectMeta(
                    generate_name="benchmark-matrix-supervisor-",
                    namespace=self.center.namespace,
                    labels={"app": "benchmark-matrix-supervisor"},
                ),
                spec=client.V1JobSpec(
                    backoff_limit=0,
                    template=template,
                ),
            )
            created = batch.create_namespaced_job(self.center.namespace, job)
            return {
                "ok": True,
                "job_name": created.metadata.name,
                "message": "监督器 Job 已启动。",
            }
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"启动监督器失败：{exc}") from exc

    def write_supervisor_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        availability = self.availability()
        if not availability["available"]:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, availability["message"])
        script = SUPERVISOR_CONTROL_WRITE_SCRIPT.replace(
            "__PAYLOAD__", json.dumps(payload, ensure_ascii=True)
        )
        result = self._kubectl_json(self.center, script)
        if not isinstance(result, dict) or not result.get("ok"):
            raise ApiError(HTTPStatus.BAD_GATEWAY, "监督器没有确认控制指令")
        return result


def cloud_signature_matches(detail: dict[str, Any], record: dict[str, Any]) -> bool:
    metrics = record.get("metrics", {})
    parameters = detail.get("parameters", {})
    checks = (
        (str(metrics.get("distributed_strategy", "")).upper(), detail.get("strategy")),
        (metrics.get("finetuning_type"), parameters.get("finetuning_type")),
        (metrics.get("quantization"), parameters.get("quantization")),
        (metrics.get("gradient_checkpointing"), parameters.get("gradient_checkpointing")),
        (metrics.get("ddp_cpu_offload"), parameters.get("ddp_cpu_offload")),
        (finite_number(metrics.get("optimizer_steps")), finite_number(parameters.get("optimizer_steps"))),
        (finite_number(metrics.get("world_size")), finite_number(parameters.get("world_size"))),
    )
    return all(expected is None or actual == expected for actual, expected in checks)


def normalize_cloud_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    detailed = record.get("detailed", {})
    training = detailed.get("training", {}) if isinstance(detailed, dict) else {}
    resources = detailed.get("resources", {}) if isinstance(detailed, dict) else {}
    federated = detailed.get("federated", {}) if isinstance(detailed, dict) else {}
    round_match = re.match(r"train-round-(\d+)-\d+-", record.get("job_id", ""))
    return {
        "site": record.get("site"),
        "job_id": record.get("job_id"),
        "round": int(round_match.group(1)) if round_match else None,
        "timestamp": metrics.get("timestamp") or detailed.get("timestamp"),
        "strategy": str(metrics.get("distributed_strategy", "unknown")).upper(),
        "training_seconds": finite_number(metrics.get("training_only_s"))
        or finite_number(training.get("total_train_time_s")),
        "avg_step_ms": finite_number(training.get("avg_step_time_ms")),
        "avg_forward_ms": positive_number(training.get("avg_forward_ms")),
        "avg_backward_ms": positive_number(training.get("avg_backward_ms")),
        "avg_communication_ms": finite_number(training.get("avg_comm_ms")),
        "avg_optimizer_ms": positive_number(training.get("avg_optimizer_ms")),
        "avg_all_reduce_ms": positive_number(training.get("avg_all_reduce_ms")),
        "avg_all_gather_ms": positive_number(training.get("avg_all_gather_ms")),
        "avg_reduce_scatter_ms": positive_number(training.get("avg_reduce_scatter_ms")),
        "avg_all_to_all_single_ms": positive_number(training.get("avg_all_to_all_single_ms")),
        "all_reduce_bytes": positive_number(training.get("total_all_reduce_bytes")),
        "all_gather_bytes": positive_number(training.get("total_all_gather_bytes")),
        "reduce_scatter_bytes": positive_number(training.get("total_reduce_scatter_bytes")),
        "all_to_all_single_bytes": positive_number(training.get("total_all_to_all_single_bytes")),
        "avg_nccl_comm_ms": positive_number(training.get("avg_nccl_comm_ms"))
        or positive_number(metrics.get("nccl_comm_ms")),
        "nccl_bytes": positive_number(training.get("total_nccl_bytes"))
        or positive_number(training.get("avg_nccl_bytes"))
        or positive_number(metrics.get("nccl_bytes")),
        "nccl_collective_calls": finite_number(training.get("nccl_collective_calls"))
        or finite_number(metrics.get("nccl_collective_calls")),
        "throughput_tokens_s": finite_number(training.get("throughput_tokens_per_s")),
        "gpu_memory_peak_mb": finite_number(resources.get("gpu_memory_peak_mb")),
        "gpu_utilization_avg_pct": finite_number(resources.get("gpu_utilization_avg_pct")),
        "cpu_utilization_avg_pct": finite_number(resources.get("cpu_utilization_avg_pct")),
        "cpu_memory_peak_mb": finite_number(resources.get("cpu_memory_peak_mb")),
        "network_rx_bytes": positive_number(resources.get("network_rx_bytes"))
        or positive_number(metrics.get("network_rx_bytes")),
        "network_tx_bytes": positive_number(resources.get("network_tx_bytes"))
        or positive_number(metrics.get("network_tx_bytes")),
        "network_total_bytes": positive_number(resources.get("network_total_bytes"))
        or positive_number(metrics.get("network_total_bytes")),
        "state_export_type": federated.get("state_export_type")
        or metrics.get("state_export_type"),
        "client_round_seconds": positive_number(
            federated.get("t_total_round_s")
        ) or positive_number(metrics.get("client_round_seconds")),
        "compression_seconds": positive_number(
            federated.get("t_full_update_compression_s")
        ) or positive_number(metrics.get("full_update_compression_seconds")),
        "model_delta_export_seconds": positive_number(
            federated.get("t_model_delta_export_s")
        ) or positive_number(metrics.get("model_delta_export_seconds")),
        "full_state_export_s": positive_number(federated.get("full_state_export_s"))
        or positive_number(metrics.get("full_state_export_s")),
        "sharded_state_export_s": positive_number(federated.get("sharded_state_export_s"))
        or positive_number(metrics.get("sharded_state_export_s")),
        "state_dict_conversion_s": positive_number(federated.get("state_dict_conversion_s"))
        or positive_number(metrics.get("state_dict_conversion_s")),
        "state_serialization_s": positive_number(federated.get("state_serialization_s"))
        or positive_number(metrics.get("state_serialization_s")),
        "state_bytes": positive_number(federated.get("state_bytes"))
        or positive_number(metrics.get("state_bytes")),
        "checkpoint_save_seconds": positive_number(federated.get("checkpoint_save_s"))
        or positive_number(metrics.get("checkpoint_save_s")),
        "checkpoint_bytes": positive_number(federated.get("checkpoint_bytes"))
        or positive_number(metrics.get("checkpoint_bytes")),
        "model_delta_bytes": positive_number(federated.get("model_delta_bytes"))
        or positive_number(metrics.get("model_delta_bytes")),
        "train_loss": finite_number(metrics.get("train_loss")),
        "num_examples": metrics.get("num_examples"),
        "optimizer_steps": metrics.get("optimizer_steps"),
        "steps": training.get("steps", []),
    }


def match_cloud_records(
    detail: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    started = parse_timestamp(detail.get("started_at"))
    finished = parse_timestamp(detail.get("finished_at"))
    completed_round = detail.get("metrics", {}).get("completed_round")
    matched: list[dict[str, Any]] = []
    for record in records:
        if not cloud_signature_matches(detail, record):
            continue
        timestamp = parse_timestamp(record.get("metrics", {}).get("timestamp"))
        if started and timestamp and timestamp < started - timedelta(minutes=2):
            continue
        if finished and timestamp and timestamp > finished + timedelta(minutes=2):
            continue
        normalized = normalize_cloud_record(record)
        if completed_round and normalized["round"] and normalized["round"] > completed_round:
            continue
        matched.append(normalized)

    # One successful result per TKE site and round; keep the newest candidate
    # when stale/retried jobs have the same experiment signature.
    newest: dict[tuple[str, int | None], dict[str, Any]] = {}
    for record in matched:
        key = (record.get("site") or "unknown", record.get("round"))
        previous = newest.get(key)
        if previous is None or (record.get("timestamp") or "") > (previous.get("timestamp") or ""):
            newest[key] = record
    return sorted(newest.values(), key=lambda item: (item.get("round") or 0, item.get("site") or ""))


def normalize_experiment(
    experiment_id: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    state: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
    federated_rounds: list[dict[str, Any]] | None = None,
    federated_timings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_config = config.get("run_config", {}) if isinstance(config, dict) else {}
    metrics = round_metrics(summary)
    federated_rounds = federated_rounds or []
    federated_timings = federated_timings or []
    latest_federated = (
        federated_rounds[-1] if federated_rounds and isinstance(federated_rounds[-1], dict) else {}
    )
    latest_timing = (
        federated_timings[-1] if federated_timings and isinstance(federated_timings[-1], dict) else {}
    )
    world_size = finite_number(metrics.get("distributed_world_size"))
    per_device_batch = finite_number(
        run_config.get("train.training-arguments.per-device-train-batch-size")
    )
    gradient_accumulation = finite_number(
        run_config.get("train.training-arguments.gradient-accumulation-steps")
    )
    effective_batch = None
    if world_size is not None and per_device_batch is not None and gradient_accumulation is not None:
        effective_batch = world_size * per_device_batch * gradient_accumulation

    status = summary.get("status") or state.get("status") or "unknown"
    evaluation_requested = bool(
        run_config.get("train.evaluate-after-fit", False)
        or float(run_config.get("strategy.fraction-evaluate", 0) or 0) > 0
        or (evaluation or {}).get("scope") == "final_global_model"
    )
    strategy = config.get("distributed_strategy") or run_config.get(
        "train.distributed-strategy", "unknown"
    )
    return {
        "id": experiment_id,
        "run_id": identifier(
            config.get("run_id") or summary.get("run_id") or state.get("run_id")
        ),
        "status": status,
        "has_summary": bool(summary),
        "started_at": config.get("started_at") or summary.get("started_at") or state.get("started_at"),
        "finished_at": summary.get("finished_at") or state.get("finished_at"),
        "strategy": str(strategy).upper(),
        "model": config.get("model") or run_config.get("model.name", "—"),
        "parameters": {
            "model": config.get("model") or run_config.get("model.name"),
            "quantization": config.get("quantization", run_config.get("model.quantization")),
            "finetuning_type": config.get("finetuning_type") or run_config.get("model.finetuning-type"),
            "client_count": run_config.get("strategy.min-fit-clients"),
            "world_size": world_size,
            "gpu_layout": (
                f"{int(world_size)} × {GPU_MODEL}" if world_size is not None else GPU_MODEL
            ),
            "sequence_length": run_config.get("train.seq-length"),
            "per_device_batch": per_device_batch,
            "gradient_accumulation": gradient_accumulation,
            "effective_client_batch": effective_batch,
            "optimizer_steps": finite_number(metrics.get("optimizer_steps"))
            or run_config.get("train.training-arguments.max-steps"),
            "learning_rate": run_config.get("train.training-arguments.learning-rate"),
            "gradient_checkpointing": run_config.get(
                "train.training-arguments.gradient-checkpointing",
                run_config.get("model.gradient-checkpointing"),
            ),
            "dataset": config.get("dataset") or run_config.get("dataset.name"),
            "max_train_samples": run_config.get("dataset.max-train-samples"),
            "federated_rounds": config.get("rounds_this_attempt")
            or run_config.get("num-server-rounds"),
            "evaluate_after_fit": run_config.get("train.evaluate-after-fit"),
            "compression": run_config.get("train.full-update-compression"),
            "topk_ratio": run_config.get("train.full-update-topk-ratio"),
            "full_local_initialization": run_config.get("train.full-local-initialization"),
            "strategy": str(strategy).upper(),
            "ddp_cpu_offload": config.get("ddp_cpu_offload", run_config.get("train.ddp-cpu-offload")),
        },
        "metrics": {
            "critical_training_seconds": finite_number(
                metrics.get("critical_path_client_training_seconds")
            ),
            "critical_round_seconds": finite_number(
                metrics.get("critical_path_client_round_seconds")
            ),
            "average_training_seconds": finite_number(metrics.get("client_training_seconds")),
            "average_round_seconds": finite_number(metrics.get("client_round_seconds")),
            "compression_seconds": finite_number(metrics.get("full_update_compression_seconds")),
            "train_loss": finite_number(metrics.get("train_loss")),
            "duration_seconds": finite_number(summary.get("duration_seconds")),
            "wall_clock_seconds": finite_number(summary.get("wall_clock_seconds")),
            "final_evaluation_seconds": finite_number(
                summary.get("final_evaluation_seconds")
            ),
            "federated_cycle_seconds": finite_number(
                latest_federated.get("federated_cycle_s") or latest_timing.get("federated_cycle_s")
            ),
            "server_post_aggregation_seconds": finite_number(
                latest_federated.get("server_post_aggregation_s")
                or latest_timing.get("server_post_aggregation_s")
            ),
            "checkpoint_interval_seconds": (
                (
                    finite_number(
                        latest_federated.get("federated_cycle_s")
                        or latest_timing.get("federated_cycle_s")
                    )
                    + finite_number(
                        latest_federated.get("server_post_aggregation_s")
                        or latest_timing.get("server_post_aggregation_s")
                    )
                )
                if finite_number(
                    latest_federated.get("federated_cycle_s")
                    or latest_timing.get("federated_cycle_s")
                ) is not None
                and finite_number(
                    latest_federated.get("server_post_aggregation_s")
                    or latest_timing.get("server_post_aggregation_s")
                ) is not None
                else None
            ),
            "server_fedavg_aggregation_seconds": finite_number(
                latest_federated.get("server_fedavg_aggregation_s")
                or latest_timing.get("server_fedavg_aggregation_s")
            ),
            "checkpoint_save_seconds": positive_number(
                latest_federated.get("checkpoint_save_s")
                or latest_timing.get("checkpoint_save_s")
            ),
            "checkpoint_bytes": positive_number(
                latest_federated.get("checkpoint_bytes")
                or latest_timing.get("checkpoint_bytes")
            ),
            "initial_state_load_seconds": positive_number(
                summary.get("initial_state_load_s")
            ),
            "server_base_state_load_seconds": positive_number(
                summary.get("server_base_state_load_s")
            ),
            "network_traffic_bytes": positive_number(metrics.get("network_total_bytes")),
            "nccl_traffic_bytes": positive_number(metrics.get("nccl_bytes")),
            "nccl_overhead_ms": positive_number(metrics.get("nccl_comm_ms")),
            "full_state_export_seconds": positive_number(metrics.get("full_state_export_s")),
            "state_serialization_seconds": positive_number(metrics.get("state_serialization_s")),
            "state_export_bytes": positive_number(metrics.get("state_bytes")),
            "completed_round": summary.get("completed_global_round")
            or state.get("latest_completed_round"),
        },
        "quality": normalize_quality_metrics(evaluation or {}),
        "quality_status": {
            "requested": evaluation_requested,
            "available": any(
                value is not None
                for value in normalize_quality_metrics(evaluation or {}).values()
            ),
        },
        "raw": {
            "config": config,
            "summary": summary,
            "state": state,
            "evaluation": evaluation or {},
            "federated_rounds": federated_rounds,
            "federated_timings": federated_timings,
        },
    }


PARAMETER_ROWS = (
    ("model", "模型"),
    ("quantization", "模型量化"),
    ("finetuning_type", "微调方式"),
    ("client_count", "客户端数量"),
    ("gpu_layout", "每客户端 GPU"),
    ("sequence_length", "序列长度"),
    ("per_device_batch", "Per-device batch"),
    ("gradient_accumulation", "梯度累积"),
    ("effective_client_batch", "每客户端有效 batch"),
    ("optimizer_steps", "Optimizer steps"),
    ("learning_rate", "学习率"),
    ("gradient_checkpointing", "梯度检查点"),
    ("dataset", "数据集"),
    ("max_train_samples", "每客户端数据上限"),
    ("federated_rounds", "联邦轮次"),
    ("evaluate_after_fit", "训练后评估"),
    ("compression", "参数传输"),
    ("topk_ratio", "Top-K ratio"),
    ("full_local_initialization", "首轮本地初始化"),
    ("strategy", "分布策略"),
    ("ddp_cpu_offload", "CPU optimizer offload"),
)

METRIC_ROWS = (
    ("critical_round_seconds", "客户端关键路径整轮耗时", "seconds", "lower"),
    ("critical_training_seconds", "客户端关键路径训练耗时", "seconds", "lower"),
    ("average_round_seconds", "两客户端平均整轮耗时", "seconds", "lower"),
    ("average_training_seconds", "两客户端平均训练耗时", "seconds", "lower"),
    ("compression_seconds", "平均 Top-K 压缩耗时", "seconds", "lower"),
    ("federated_cycle_seconds", "联邦周期：下发/训练/上传/聚合", "seconds", "lower"),
    ("server_post_aggregation_seconds", "中心聚合后处理", "seconds", "lower"),
    ("checkpoint_interval_seconds", "全局 checkpoint 间隔", "seconds", "lower"),
    ("server_fedavg_aggregation_seconds", "纯 FedAvg 聚合计算", "seconds", "lower"),
    ("checkpoint_save_seconds", "Checkpoint 保存耗时", "seconds", "lower"),
    ("initial_state_load_seconds", "初始状态/恢复加载", "seconds", "lower"),
    ("server_base_state_load_seconds", "中心基座状态加载", "seconds", "lower"),
    ("train_loss", "平均训练 Loss", "number", "neutral"),
    ("duration_seconds", "Run 端到端耗时（可能含排队）", "seconds", "context"),
)

QUALITY_METRIC_ROWS = (
    ("validation_loss", "Held-out Validation Loss", "number", "lower"),
    ("perplexity", "Held-out PPL", "number", "lower"),
    ("rouge_l_f1", "ROUGE-L F1", "ratio", "higher"),
    ("bertscore_f1", "BERTScore F1", "ratio", "higher"),
    ("accuracy", "Accuracy", "ratio", "higher"),
    ("macro_f1", "Macro-F1", "ratio", "higher"),
    ("exact_match", "Exact Match", "ratio", "higher"),
    ("empty_prediction_rate", "空回答比例", "ratio", "lower"),
    ("average_generated_tokens", "平均生成 Token", "number", "context"),
)

CLOUD_METRIC_ROWS = (
    ("critical_training_seconds", "TKE 客户端训练关键路径", "seconds", "lower"),
    ("tke_a_training_seconds", "TKE-A 训练耗时", "seconds", "lower"),
    ("tke_b_training_seconds", "TKE-B 训练耗时", "seconds", "lower"),
    ("average_training_seconds", "TKE A/B 平均训练耗时", "seconds", "lower"),
    ("critical_round_seconds", "TKE 客户端整轮关键路径", "seconds", "lower"),
    ("tke_a_round_seconds", "TKE-A 客户端整轮", "seconds", "lower"),
    ("tke_b_round_seconds", "TKE-B 客户端整轮", "seconds", "lower"),
    ("average_round_seconds", "TKE A/B 平均整轮", "seconds", "lower"),
    ("critical_compression_seconds", "TKE Top-K 压缩关键路径", "seconds", "lower"),
    ("average_compression_seconds", "TKE A/B 平均 Top-K 压缩", "seconds", "lower"),
    ("average_step_ms", "平均 optimizer step 耗时", "milliseconds", "lower"),
    ("average_throughput", "平均吞吐量", "tokens_s", "higher"),
    ("server_fedavg_aggregation_seconds", "中心 FedAvg 聚合计算", "seconds", "lower"),
    ("checkpoint_save_seconds", "中心 Checkpoint 保存", "seconds", "lower"),
    ("checkpoint_bytes", "中心 Checkpoint 大小", "bytes", "context"),
    ("gpu_memory_peak_mb", "客户端 GPU 显存峰值", "memory_mb", "context"),
    ("cpu_memory_peak_mb", "客户端 CPU 内存峰值", "memory_mb", "context"),
    ("cpu_utilization_avg_pct", "客户端 CPU 平均利用率", "percent", "context"),
    ("train_loss", "客户端平均训练 Loss", "number", "neutral"),
)

# These fields are deliberately limited to intra-cluster work.  WAN update
# transport, model-delta export, and server FedAvg timings are a separate
# cross-centre evaluation stream and are not presented as local performance.
AGGREGATION_PERFORMANCE_ROWS = (
    ("critical_training_seconds", "集群内训练关键路径", "seconds", "lower"),
    ("average_training_seconds", "集群内平均总训练时间", "seconds", "lower"),
    ("average_step_ms", "平均单 Step 时间", "milliseconds", "lower"),
    ("average_forward_ms", "平均前向传播时间", "milliseconds", "lower"),
    ("average_backward_ms", "平均反向传播时间", "milliseconds", "lower"),
    ("average_communication_ms", "平均集群内通信时间", "milliseconds", "lower"),
    ("avg_all_reduce_ms", "All-Reduce 时间", "milliseconds", "lower"),
    ("avg_all_gather_ms", "All-Gather 时间", "milliseconds", "lower"),
    ("avg_reduce_scatter_ms", "Reduce-Scatter 时间", "milliseconds", "lower"),
    ("average_optimizer_ms", "平均优化器更新时间", "milliseconds", "lower"),
    ("average_throughput", "平均训练吞吐量", "tokens_s", "higher"),
)

AGGREGATION_RESOURCE_ROWS = (
    ("gpu_memory_peak_mb", "GPU 显存峰值", "memory_mb", "context"),
    ("gpu_utilization_avg_pct", "GPU 平均利用率", "percent", "context"),
    ("cpu_memory_peak_mb", "CPU 内存峰值", "memory_mb", "context"),
    ("cpu_utilization_avg_pct", "CPU 平均利用率", "percent", "context"),
)

AGGREGATION_NETWORK_ROWS = (
    ("network_traffic_bytes", "训练期间 Pod 网络流量", "bytes", "context"),
    ("nccl_traffic_bytes", "NCCL 集合通信流量", "bytes", "context"),
    ("average_nccl_comm_ms", "NCCL 集合通信开销", "milliseconds", "lower"),
    ("all_reduce_bytes", "All-Reduce 流量", "bytes", "context"),
    ("all_gather_bytes", "All-Gather 流量", "bytes", "context"),
    ("reduce_scatter_bytes", "Reduce-Scatter 流量", "bytes", "context"),
)

STATE_EXPORT_ROWS = (
    ("full_state_export_seconds", "Full-state export", "seconds", "lower"),
    ("sharded_state_export_seconds", "Sharded-state export", "seconds", "lower"),
    ("state_dict_conversion_seconds", "状态转换耗时", "seconds", "lower"),
    ("state_serialization_seconds", "状态序列化耗时", "seconds", "lower"),
    ("state_export_bytes", "导出状态大小", "bytes", "context"),
    ("checkpoint_save_seconds", "Checkpoint 保存耗时", "seconds", "lower"),
    ("checkpoint_bytes", "Checkpoint 大小", "bytes", "context"),
)


def parameter_comparison_rows(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for key, label in PARAMETER_ROWS:
        left_value = left["parameters"].get(key)
        right_value = right["parameters"].get(key)
        expected_difference = key in {"strategy", "ddp_cpu_offload"}
        rows.append(
            {
                "key": key,
                "label": label,
                "left": left_value,
                "right": right_value,
                "same": left_value == right_value,
                "expected_difference": expected_difference,
            }
        )
    return rows


def metric_comparison_rows(
    rows: tuple[tuple[str, str, str, str], ...],
    left_metrics: dict[str, Any],
    right_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    compared = []
    for key, label, unit, interpretation in rows:
        left_value = left_metrics.get(key)
        right_value = right_metrics.get(key)
        delta = None
        percent = None
        winner = None
        if left_value is not None and right_value is not None:
            delta = left_value - right_value
            if right_value:
                percent = abs(delta) / abs(right_value) * 100
            if interpretation == "lower" and delta != 0:
                winner = "left" if delta < 0 else "right"
            elif interpretation == "higher" and delta != 0:
                winner = "left" if delta > 0 else "right"
        compared.append(
            {
                "key": key,
                "label": label,
                "left": left_value,
                "right": right_value,
                "unit": unit,
                "interpretation": interpretation,
                "delta": delta,
                "percent": percent,
                "winner": winner,
            }
        )
    return compared


def has_partial_results(detail: dict[str, Any]) -> bool:
    """Return whether an experiment has persisted, comparable progress."""
    metrics = detail.get("metrics", {}) if isinstance(detail, dict) else {}
    completed_round = finite_number(metrics.get("completed_round"))
    if completed_round is not None and completed_round > 0:
        return True
    if federated_timing_records(detail):
        return True
    clients = detail.get("cloud", {}).get("clients", [])
    return any(
        (_round_number(record) or 0) > 0
        for record in clients
        if isinstance(record, dict)
    )


def comparison_scope(left: dict[str, Any], right: dict[str, Any]) -> str:
    """Classify comparison as formal, partial, or unavailable."""
    formal = all(
        detail.get("has_summary") and detail.get("status") == "completed"
        for detail in (left, right)
    )
    if formal:
        return "formal"
    if has_partial_results(left) and has_partial_results(right):
        return "partial"
    return "unavailable"


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    parameter_rows = parameter_comparison_rows(left, right)
    metric_rows = metric_comparison_rows(METRIC_ROWS, left["metrics"], right["metrics"])
    primary = next(row for row in metric_rows if row["key"] == "critical_round_seconds")
    return {
        "source": "center",
        "verdict_scope": "客户端关键路径整轮耗时",
        "comparison_scope": comparison_scope(left, right),
        "left": left,
        "right": right,
        "parameters": parameter_rows,
        "metrics": metric_rows,
        "quality_metrics": metric_comparison_rows(
            QUALITY_METRIC_ROWS, left["quality"], right["quality"]
        ),
        "timing": {
            "left": federated_timing_records(left),
            "right": federated_timing_records(right),
        },
        "primary": primary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def cloud_aggregate(detail: dict[str, Any]) -> dict[str, Any]:
    records = detail.get("cloud", {}).get("clients", [])
    if not records:
        return {}
    latest_round = max((record.get("round") or 0 for record in records), default=0)
    clients = [record for record in records if (record.get("round") or 0) == latest_round]
    by_site = {record.get("site"): record for record in clients}
    training_values = [record.get("training_seconds") for record in clients]
    round_values = [record.get("client_round_seconds") for record in clients]
    compression_values = [record.get("compression_seconds") for record in clients]
    return {
        "critical_training_seconds": max(
            (value for value in training_values if value is not None), default=None
        ),
        "tke_a_training_seconds": by_site.get("TKE-A", {}).get("training_seconds"),
        "tke_b_training_seconds": by_site.get("TKE-B", {}).get("training_seconds"),
        "average_training_seconds": average(training_values),
        "critical_round_seconds": max(
            (value for value in round_values if value is not None), default=None
        ),
        "tke_a_round_seconds": by_site.get("TKE-A", {}).get("client_round_seconds"),
        "tke_b_round_seconds": by_site.get("TKE-B", {}).get("client_round_seconds"),
        "average_round_seconds": average(round_values),
        "critical_compression_seconds": max(
            (value for value in compression_values if value is not None), default=None
        ),
        "average_compression_seconds": average(compression_values),
        "average_step_ms": average([record.get("avg_step_ms") for record in clients]),
        "average_forward_ms": average([record.get("avg_forward_ms") for record in clients]),
        "average_backward_ms": average([record.get("avg_backward_ms") for record in clients]),
        "average_communication_ms": average(
            [record.get("avg_communication_ms") for record in clients]
        ),
        "average_optimizer_ms": average([record.get("avg_optimizer_ms") for record in clients]),
        "average_nccl_comm_ms": average([record.get("avg_nccl_comm_ms") for record in clients]),
        "nccl_traffic_bytes": sum(
            record.get("nccl_bytes") or 0 for record in clients
        ) or None,
        "network_traffic_bytes": sum(
            record.get("network_total_bytes") or 0 for record in clients
        ) or None,
        "average_throughput": average(
            [record.get("throughput_tokens_s") for record in clients]
        ),
        "gpu_memory_peak_mb": max(
            (
                record.get("gpu_memory_peak_mb")
                for record in clients
                if record.get("gpu_memory_peak_mb") is not None
            ),
            default=None,
        ),
        "gpu_utilization_avg_pct": average(
            [record.get("gpu_utilization_avg_pct") for record in clients]
        ),
        "cpu_memory_peak_mb": max(
            (
                record.get("cpu_memory_peak_mb")
                for record in clients
                if record.get("cpu_memory_peak_mb") is not None
            ),
            default=None,
        ),
        "cpu_utilization_avg_pct": average(
            [record.get("cpu_utilization_avg_pct") for record in clients]
        ),
        "full_state_export_seconds": max(
            (record.get("full_state_export_s") for record in clients if record.get("full_state_export_s") is not None),
            default=None,
        ),
        "sharded_state_export_seconds": max(
            (record.get("sharded_state_export_s") for record in clients if record.get("sharded_state_export_s") is not None),
            default=None,
        ),
        "state_dict_conversion_seconds": max(
            (record.get("state_dict_conversion_s") for record in clients if record.get("state_dict_conversion_s") is not None),
            default=None,
        ),
        "state_serialization_seconds": max(
            (record.get("state_serialization_s") for record in clients if record.get("state_serialization_s") is not None),
            default=None,
        ),
        "state_export_bytes": max(
            (record.get("state_bytes") for record in clients if record.get("state_bytes") is not None),
            default=None,
        ),
        "train_loss": average([record.get("train_loss") for record in clients]),
        "server_fedavg_aggregation_seconds": detail.get("metrics", {}).get(
            "server_fedavg_aggregation_seconds"
        ),
        "checkpoint_save_seconds": detail.get("metrics", {}).get(
            "checkpoint_save_seconds"
        ),
        "checkpoint_bytes": detail.get("metrics", {}).get("checkpoint_bytes"),
    }


def _round_number(record: dict[str, Any]) -> int | None:
    value = record.get("round")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def federated_timing_records(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Combine center round timings with the matching TKE client timings."""
    raw = detail.get("raw", {}) if isinstance(detail, dict) else {}
    center_records = raw.get("federated_rounds", []) if isinstance(raw, dict) else []
    center_by_round: dict[int, dict[str, Any]] = {}
    for record in center_records:
        if not isinstance(record, dict):
            continue
        round_number = _round_number(record)
        if round_number is None:
            round_number = _round_number({"round": record.get("server_round")})
        if round_number is not None:
            center_by_round[round_number] = record

    clients = detail.get("cloud", {}).get("clients", [])
    clients_by_round: dict[int, list[dict[str, Any]]] = {}
    for record in clients if isinstance(clients, list) else []:
        if not isinstance(record, dict):
            continue
        round_number = _round_number(record)
        if round_number is not None:
            clients_by_round.setdefault(round_number, []).append(record)

    round_numbers = sorted(set(center_by_round) | set(clients_by_round))
    rows: list[dict[str, Any]] = []
    for round_number in round_numbers:
        client_records = clients_by_round.get(round_number, [])
        # Flower emits a server_round=0 callback for initialization. It is not
        # a federated training/aggregation round and has no client timings.
        if round_number == 0 and not client_records:
            continue
        by_site = {record.get("site"): record for record in client_records}
        training_values = [record.get("training_seconds") for record in client_records]
        client_round_values = [
            record.get("client_round_seconds") for record in client_records
        ]
        compression_values = [
            record.get("compression_seconds") for record in client_records
        ]
        center = center_by_round.get(round_number, {})
        federated_cycle = finite_number(center.get("federated_cycle_s"))
        server_post = finite_number(center.get("server_post_aggregation_s"))
        rows.append({
            "round": round_number,
            "tke_a_training_seconds": by_site.get("TKE-A", {}).get("training_seconds"),
            "tke_b_training_seconds": by_site.get("TKE-B", {}).get("training_seconds"),
            "critical_training_seconds": max(
                (value for value in training_values if value is not None), default=None
            ),
            "tke_a_client_round_seconds": by_site.get("TKE-A", {}).get(
                "client_round_seconds"
            ),
            "tke_b_client_round_seconds": by_site.get("TKE-B", {}).get(
                "client_round_seconds"
            ),
            "critical_client_round_seconds": max(
                (value for value in client_round_values if value is not None),
                default=None,
            ),
            "tke_a_compression_seconds": by_site.get("TKE-A", {}).get(
                "compression_seconds"
            ),
            "tke_b_compression_seconds": by_site.get("TKE-B", {}).get(
                "compression_seconds"
            ),
            "average_compression_seconds": average(compression_values),
            "federated_cycle_seconds": federated_cycle,
            "server_post_aggregation_seconds": server_post,
            "checkpoint_interval_seconds": (
                federated_cycle + server_post
                if federated_cycle is not None and server_post is not None
                else None
            ),
            "server_fedavg_aggregation_seconds": finite_number(
                center.get("server_fedavg_aggregation_s")
            ),
            "checkpoint_save_seconds": finite_number(center.get("checkpoint_save_s")),
            "checkpoint_bytes": positive_number(center.get("checkpoint_bytes")),
            "timestamp": center.get("timestamp"),
        })
    return rows


def aggregation_observability(detail: dict[str, Any]) -> dict[str, Any]:
    """Summarize latest TKE results as observed by the aggregation server."""
    records = detail.get("cloud", {}).get("clients", [])
    latest_round = max((record.get("round") or 0 for record in records), default=0)
    latest = [record for record in records if (record.get("round") or 0) == latest_round]
    aggregate = cloud_aggregate(detail)
    server_metrics = detail.get("metrics", {})
    return {
        "observed_round": latest_round or None,
        "cluster_count": len(latest),
        "source": "聚合端读取 TKE A/B 已完成训练结果",
        "coverage": {
            "network": any((record.get("network_total_bytes") or 0) > 0 for record in latest),
            "nccl": any((record.get("nccl_bytes") or 0) > 0 for record in latest),
            "state_export": any((record.get("full_state_export_s") or 0) > 0 for record in latest),
        },
        "performance": aggregate,
        "resources": aggregate,
        "server": {
            "federated_cycle_seconds": server_metrics.get("federated_cycle_seconds"),
            "server_post_aggregation_seconds": server_metrics.get(
                "server_post_aggregation_seconds"
            ),
            "server_fedavg_aggregation_seconds": server_metrics.get(
                "server_fedavg_aggregation_seconds"
            ),
            "checkpoint_save_seconds": server_metrics.get("checkpoint_save_seconds"),
            "checkpoint_bytes": server_metrics.get("checkpoint_bytes"),
        },
        "cross_centre_update": {
            "implemented": False,
            "message": "跨智算中心联邦更新指标暂未纳入本看板。",
        },
    }


def compare_cloud(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    metric_rows = metric_comparison_rows(
        CLOUD_METRIC_ROWS, cloud_aggregate(left), cloud_aggregate(right)
    )
    primary = next(row for row in metric_rows if row["key"] == "critical_training_seconds")
    return {
        "source": "cloud",
        "verdict_scope": "TKE 客户端训练关键路径",
        "comparison_scope": comparison_scope(left, right),
        "left": left,
        "right": right,
        "parameters": parameter_comparison_rows(left, right),
        "metrics": metric_rows,
        "quality_metrics": metric_comparison_rows(
            QUALITY_METRIC_ROWS, left["quality"], right["quality"]
        ),
        "timing": {
            "left": federated_timing_records(left),
            "right": federated_timing_records(right),
        },
        "aggregation": {
            "left": aggregation_observability(left),
            "right": aggregation_observability(right),
            "performance_metrics": metric_comparison_rows(
                AGGREGATION_PERFORMANCE_ROWS, cloud_aggregate(left), cloud_aggregate(right)
            ),
            "resource_metrics": metric_comparison_rows(
                AGGREGATION_RESOURCE_ROWS, cloud_aggregate(left), cloud_aggregate(right)
            ),
            "network_metrics": metric_comparison_rows(
                AGGREGATION_NETWORK_ROWS, cloud_aggregate(left), cloud_aggregate(right)
            ),
            "state_export_metrics": metric_comparison_rows(
                STATE_EXPORT_ROWS, cloud_aggregate(left), cloud_aggregate(right)
            ),
        },
        "primary": primary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ExperimentDashboard/1.0"

    @property
    def store(self) -> ExperimentStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def cloud_store(self) -> KubectlCloudStore:
        return self.server.cloud_store  # type: ignore[attr-defined]

    @property
    def auth(self) -> DashboardAuth:
        return self.server.auth  # type: ignore[attr-defined]

    def require_auth(self, path: str) -> bool:
        # Kubernetes probes must stay unauthenticated. All UI and data/control
        # routes remain protected when the TKE Secret enables Basic Auth.
        if path == "/api/health" or self.auth.accepts(self.headers.get("Authorization")):
            return True
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="Federated Lab", charset="UTF-8"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        return False

    def selected_store(self, query: dict[str, list[str]]) -> Any:
        source = query.get("source", ["center"])[0]
        if source == "center":
            return self.store
        if source == "cloud":
            return self.cloud_store
        raise ApiError(HTTPStatus.BAD_REQUEST, f"不支持的数据源：{source}")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, relative_path: str, head_only: bool = False) -> None:
        requested = (STATIC_ROOT / relative_path).resolve()
        try:
            requested.relative_to(STATIC_ROOT.resolve())
        except ValueError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "资源不存在") from exc
        if not requested.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "资源不存在")
        body = requested.read_bytes()
        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            parsed = urlparse(self.path)
            if not self.require_auth(parsed.path):
                return
            if parsed.path in {"/", "/index.html"}:
                self.send_static("index.html", head_only=True)
                return
            if parsed.path.startswith("/static/"):
                self.send_static(parsed.path.removeprefix("/static/"), head_only=True)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, "页面不存在")
        except ApiError as exc:
            self.send_json({"error": exc.message}, exc.status)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            parsed = urlparse(self.path)
            if not self.require_auth(parsed.path):
                return
            if parsed.path != "/api/supervisor":
                raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")
            query = parse_qs(parsed.query)
            if query.get("source", ["cloud"])[0] != "cloud":
                raise ApiError(HTTPStatus.BAD_REQUEST, "监督器控制只允许使用 cloud 数据源")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, "请求体长度非法") from exc
            if length <= 0 or length > 16 * 1024:
                raise ApiError(HTTPStatus.BAD_REQUEST, "控制请求体为空或过大")
            try:
                request = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, "请求体不是有效 JSON") from exc
            action = str(request.get("action", "configure")).lower()
            snapshot = self.cloud_store.supervisor_snapshot()
            current = snapshot.get("control", {}) if isinstance(snapshot, dict) else {}
            control = normalize_supervisor_control(current, request)
            result = self.cloud_store.write_supervisor_control(control)
            if action == "start":
                result = {**result, "start": self.cloud_store.start_supervisor()}
            self.send_json({"control": control, "result": result, "supervisor": self.cloud_store.supervisor_snapshot()})
        except ApiError as exc:
            self.send_json({"error": exc.message}, exc.status)
        except Exception as exc:
            self.send_json({"error": f"监督器控制失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            parsed = urlparse(self.path)
            if not self.require_auth(parsed.path):
                return
            query = parse_qs(parsed.query)
            if parsed.path == "/api/health":
                self.send_json(
                    {
                        "status": "ok",
                        "results_root": str(self.store.root),
                        "results_root_exists": self.store.root.is_dir(),
                        "cloud": self.cloud_store.availability(),
                        "swanlab": self.server.swanlab_sync.status(),
                    }
                )
                return
            if parsed.path == "/api/sources":
                self.send_json(
                    {
                        "sources": [
                            {
                                "id": "center",
                                "label": "中心聚合结果",
                                "available": self.store.root.is_dir(),
                                "description": "读取 ServerApp experiment_config/summary/state",
                            },
                            {
                                "id": "cloud",
                                "label": "TKE 云端结果",
                                "description": "实时读取中心以及 TKE A/B 客户端详细指标",
                                **self.cloud_store.availability(),
                            },
                        ]
                    }
                )
                return
            if parsed.path == "/api/supervisor":
                source = query.get("source", ["cloud"])[0]
                if source != "cloud":
                    raise ApiError(HTTPStatus.BAD_REQUEST, "监督器状态只允许使用 cloud 数据源")
                self.send_json(self.cloud_store.supervisor_snapshot())
                return
            if parsed.path == "/api/experiments":
                selected = self.selected_store(query)
                force = query.get("refresh", ["0"])[0] == "1"
                experiments = selected.list(force=force) if selected is self.cloud_store else selected.list()
                self.send_json({"experiments": experiments})
                return
            if parsed.path.startswith("/api/experiments/") and parsed.path.endswith("/logs"):
                experiment_id = unquote(parsed.path.removeprefix("/api/experiments/").removesuffix("/logs").rstrip("/"))
                self.send_json(self.selected_store(query).logs(experiment_id))
                return
            if parsed.path.startswith("/api/experiments/"):
                experiment_id = unquote(parsed.path.removeprefix("/api/experiments/"))
                self.send_json(self.selected_store(query).read(experiment_id))
                return
            if parsed.path == "/api/compare":
                left_id = query.get("left", [""])[0]
                right_id = query.get("right", [""])[0]
                if left_id == right_id:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "请选择两个不同的实验")
                selected = self.selected_store(query)
                left = selected.read(left_id)
                right = selected.read(right_id)
                self.send_json(compare_cloud(left, right) if selected is self.cloud_store else compare(left, right))
                return
            if parsed.path in {"/", "/index.html"}:
                self.send_static("index.html")
                return
            if parsed.path.startswith("/static/"):
                self.send_static(parsed.path.removeprefix("/static/"))
                return
            raise ApiError(HTTPStatus.NOT_FOUND, "页面不存在")
        except ApiError as exc:
            self.send_json({"error": exc.message}, exc.status)
        except Exception as exc:  # keep malformed result files from killing the service
            self.send_json({"error": f"服务器读取实验结果失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


class DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        store: ExperimentStore,
        cloud_store: KubectlCloudStore | None = None,
    ):
        super().__init__(address, DashboardHandler)
        self.auth = DashboardAuth()
        self.store = store
        self.cloud_store = cloud_store or KubectlCloudStore()
        self.swanlab_sync = SwanLabSync(store)
        self.swanlab_sync.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated experiment comparison dashboard")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing one subdirectory per experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = DashboardServer((args.host, args.port), ExperimentStore(args.results_root.resolve()))
    print(f"Experiment dashboard listening on http://{args.host}:{args.port}")
    print(f"Reading experiment results from {args.results_root.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.swanlab_sync.stop()
        server.server_close()


if __name__ == "__main__":
    main()
