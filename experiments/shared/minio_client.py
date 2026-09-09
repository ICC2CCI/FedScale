"""MinIO / S3 读写封装。"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class MinIOClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: Optional[bool] = None,
    ) -> None:
        from minio import Minio

        endpoint = endpoint.strip()
        if endpoint.startswith("https://"):
            host = endpoint[len("https://") :]
            use_secure = True if secure is None else secure
        elif endpoint.startswith("http://"):
            host = endpoint[len("http://") :]
            use_secure = False if secure is None else secure
        else:
            host = endpoint
            use_secure = False if secure is None else secure

        self.bucket = bucket
        self.client = Minio(host, access_key=access_key, secret_key=secret_key, secure=use_secure)
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(
            self.bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except Exception:
            return False

    def put_json(self, key: str, obj: Dict[str, Any]) -> None:
        data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.put_bytes(key, data, content_type="application/json")

    def get_json(self, key: str) -> Dict[str, Any]:
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def put_torch(self, key: str, obj: Any) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = tmp.name
        try:
            torch.save(obj, path)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                self.client.put_object(self.bucket, key, f, length=size)
        finally:
            Path(path).unlink(missing_ok=True)

    def get_torch(self, key: str, map_location: Union[str, torch.device] = "cpu") -> Any:
        data = self.get_bytes(key)
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)
