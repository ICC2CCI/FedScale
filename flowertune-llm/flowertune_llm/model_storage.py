"""Small, deterministic S3 artifact helpers for object-store federation.

Flower transports only the metadata represented by :class:`ModelArtifact`.
The checkpoint bytes are uploaded directly by the client and downloaded by an
aggregation Worker through an S3-compatible object store (MinIO/COS/S3).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


class ObjectStoreError(RuntimeError):
    """Raised when an artifact cannot safely be published or consumed."""


def _validate_identifier(value: str, name: str) -> str:
    value = str(value).strip()
    if not _IDENTIFIER.fullmatch(value):
        raise ObjectStoreError(f"invalid {name}: {value!r}")
    return value


def _positive_int_env(name: str, default: int) -> int:
    """Read a bounded transport setting early, before a long transfer starts."""
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ObjectStoreError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ObjectStoreError(f"{name} must be a positive integer, got {raw!r}")
    return value


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a checkpoint without loading it into process memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelArtifact:
    """The immutable metadata Flower may carry for one model checkpoint."""

    uri: str
    sha256: str
    size: int
    experiment_id: str
    round: int
    role: str
    num_examples: int = 0

    def __post_init__(self) -> None:
        _validate_identifier(self.experiment_id, "experiment_id")
        _validate_identifier(self.role, "role")
        if int(self.round) < 0:
            raise ObjectStoreError("artifact round cannot be negative")
        if int(self.size) <= 0:
            raise ObjectStoreError("artifact size must be positive")
        if int(self.num_examples) < 0:
            raise ObjectStoreError("num_examples cannot be negative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ObjectStoreError("artifact sha256 must be a lowercase SHA-256 hex digest")
        parse_s3_uri(self.uri)

    def to_config(self) -> dict[str, str | int]:
        """Return ConfigRecord-compatible scalar values."""
        return {
            "model-uri": self.uri,
            "model-sha256": self.sha256,
            "model-size": int(self.size),
            "artifact-round": int(self.round),
            "artifact-role": self.role,
            "artifact-experiment-id": self.experiment_id,
            "num-examples": int(self.num_examples),
        }

    @classmethod
    def from_config(cls, values: dict[str, Any]) -> "ModelArtifact":
        return cls(
            uri=str(values["model-uri"]),
            sha256=str(values["model-sha256"]),
            size=int(values["model-size"]),
            experiment_id=str(values["artifact-experiment-id"]),
            round=int(values["artifact-round"]),
            role=str(values["artifact-role"]),
            num_examples=int(values.get("num-examples", 0)),
        )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(str(uri))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ObjectStoreError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def model_key(experiment_id: str, round_number: int, role: str) -> str:
    """Use one predictable key per role; retries safely overwrite no other role."""
    experiment_id = _validate_identifier(experiment_id, "experiment_id")
    role = _validate_identifier(role, "role")
    if int(round_number) < 0:
        raise ObjectStoreError("round cannot be negative")
    return f"{experiment_id}/round-{int(round_number)}/{role}.pt"


def round_state_key(experiment_id: str, round_number: int) -> str:
    experiment_id = _validate_identifier(experiment_id, "experiment_id")
    if int(round_number) < 0:
        raise ObjectStoreError("round cannot be negative")
    return f"{experiment_id}/round-{int(round_number)}/round_state.json"


class ModelStorage:
    """S3-compatible checkpoint store with verified, atomic local downloads."""

    def __init__(self, client: Any, bucket: str, retries: int = 3) -> None:
        if not bucket or "/" in bucket:
            raise ObjectStoreError("S3 bucket must be a non-empty bucket name")
        self.client = client
        self.bucket = bucket
        self.retries = max(1, int(retries))

    @classmethod
    def from_env(cls) -> "ModelStorage":
        """Create a store from deployment secrets without importing boto3 in tests."""
        endpoint = os.environ.get("S3_ENDPOINT", "").strip()
        bucket = os.environ.get("S3_BUCKET", "").strip()
        access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
        secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
        if not all((endpoint, bucket, access_key, secret_key)):
            raise ObjectStoreError(
                "S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, and S3_SECRET_KEY are required"
            )
        retries = _positive_int_env("S3_RETRIES", 3)
        connect_timeout = _positive_int_env("S3_CONNECT_TIMEOUT_SECONDS", 10)
        read_timeout = _positive_int_env("S3_READ_TIMEOUT_SECONDS", 300)
        ca_bundle = os.environ.get("S3_CA_BUNDLE", "").strip()
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - exercised in runtime image
            raise ObjectStoreError("boto3 must be installed for object-store transport") from exc
        client_options: dict[str, Any] = {
            "endpoint_url": endpoint,
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": os.environ.get("S3_REGION", "us-east-1"),
            "config": Config(
                s3={"addressing_style": os.environ.get("S3_ADDRESSING_STYLE", "path")},
                retries={"max_attempts": max(3, retries), "mode": "standard"},
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                tcp_keepalive=True,
            ),
        }
        if ca_bundle:
            client_options["verify"] = ca_bundle
        client = boto3.client(
            "s3",
            **client_options,
        )
        return cls(client, bucket, retries=retries)

    def uri_for_key(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def _retry(self, operation, description: str):
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return operation()
            except Exception as exc:  # boto-compatible clients expose varied errors
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(5, attempt))
        raise ObjectStoreError(f"{description} failed after {self.retries} attempts: {last_error}") from last_error

    def head_artifact(self, uri: str, *, expected_sha256: str | None = None, expected_size: int | None = None) -> dict[str, Any]:
        bucket, key = parse_s3_uri(uri)
        if bucket != self.bucket:
            raise ObjectStoreError(f"artifact bucket {bucket!r} is not configured bucket {self.bucket!r}")
        response = self._retry(
            lambda: self.client.head_object(Bucket=bucket, Key=key), f"head {uri}"
        )
        size = int(response.get("ContentLength", -1))
        metadata = {str(k).lower(): str(v) for k, v in response.get("Metadata", {}).items()}
        actual_sha256 = metadata.get("sha256", "")
        if expected_size is not None and size != int(expected_size):
            raise ObjectStoreError(f"artifact size mismatch for {uri}: got {size}, expected {expected_size}")
        if expected_sha256 is not None and actual_sha256 and actual_sha256 != expected_sha256:
            raise ObjectStoreError(f"artifact object metadata SHA-256 mismatch for {uri}")
        return {"size": size, "metadata": metadata}

    def artifact_from_uri(
        self, uri: str, *, experiment_id: str, round_number: int, role: str, num_examples: int = 0
    ) -> ModelArtifact:
        """Rehydrate verified artifact metadata from a deterministic S3 object."""
        head = self.head_artifact(uri)
        sha256 = str(head["metadata"].get("sha256", ""))
        if not sha256:
            raise ObjectStoreError(f"artifact {uri} has no SHA-256 object metadata")
        return ModelArtifact(
            uri=uri,
            sha256=sha256,
            size=int(head["size"]),
            experiment_id=experiment_id,
            round=round_number,
            role=role,
            num_examples=num_examples,
        )

    def upload_model(
        self,
        local_path: str | Path,
        *,
        experiment_id: str,
        round_number: int,
        role: str,
        num_examples: int = 0,
    ) -> ModelArtifact:
        local_path = Path(local_path)
        if not local_path.is_file():
            raise ObjectStoreError(f"checkpoint is missing: {local_path}")
        key = model_key(experiment_id, round_number, role)
        size = local_path.stat().st_size
        sha256 = sha256_file(local_path)
        metadata = {
            "sha256": sha256,
            "experiment-id": _validate_identifier(experiment_id, "experiment_id"),
            "round": str(int(round_number)),
            "role": _validate_identifier(role, "role"),
        }
        self._retry(
            lambda: self.client.upload_file(
                str(local_path), self.bucket, key, ExtraArgs={"Metadata": metadata}
            ),
            f"upload {local_path}",
        )
        uri = self.uri_for_key(key)
        self.head_artifact(uri, expected_sha256=sha256, expected_size=size)
        return ModelArtifact(
            uri=uri,
            sha256=sha256,
            size=size,
            experiment_id=experiment_id,
            round=int(round_number),
            role=role,
            num_examples=int(num_examples),
        )

    def download_model(self, artifact: ModelArtifact, destination: str | Path) -> Path:
        """Download to a temporary file, verify bytes, then atomically publish it."""
        bucket, key = parse_s3_uri(artifact.uri)
        if bucket != self.bucket:
            raise ObjectStoreError(f"artifact bucket {bucket!r} is not configured bucket {self.bucket!r}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        self.head_artifact(
            artifact.uri, expected_sha256=artifact.sha256, expected_size=artifact.size
        )
        self._retry(
            lambda: self.client.download_file(bucket, key, str(temporary)),
            f"download {artifact.uri}",
        )
        actual_size = temporary.stat().st_size if temporary.exists() else -1
        actual_sha256 = sha256_file(temporary) if temporary.exists() else ""
        if actual_size != artifact.size or actual_sha256 != artifact.sha256:
            temporary.unlink(missing_ok=True)
            raise ObjectStoreError(
                f"download verification failed for {artifact.uri}: "
                f"size={actual_size}, sha256={actual_sha256}"
            )
        os.replace(temporary, destination)
        return destination

    def put_round_state(self, experiment_id: str, round_number: int, payload: dict[str, Any]) -> str:
        key = round_state_key(experiment_id, round_number)
        body = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
        self._retry(
            lambda: self.client.put_object(
                Bucket=self.bucket, Key=key, Body=body, ContentType="application/json"
            ),
            f"write round state {key}",
        )
        return self.uri_for_key(key)

    def get_round_state(self, experiment_id: str, round_number: int) -> dict[str, Any] | None:
        key = round_state_key(experiment_id, round_number)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception:
            return None
        try:
            return json.loads(response["Body"].read().decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ObjectStoreError(f"round state {key} is malformed") from exc


def artifact_state(status: str, artifact: ModelArtifact | None = None, **extra: Any) -> dict[str, Any]:
    """Create a compact, durable round-state document."""
    payload: dict[str, Any] = {"status": status, **extra}
    if artifact is not None:
        payload["global"] = asdict(artifact)
    return payload
