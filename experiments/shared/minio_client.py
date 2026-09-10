"""MinIO / S3 读写封装（长超时 + multipart + 重试）。"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

logger = logging.getLogger("minio_client")

# 大于该阈值走 multipart，降低单连接长时间卡住的概率
_MULTIPART_THRESHOLD = 32 * 1024 * 1024
_PART_SIZE = 64 * 1024 * 1024


class MinIOClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: Optional[bool] = None,
        read_timeout_s: float = 1800.0,
        max_retries: int = 5,
    ) -> None:
        from minio import Minio
        import urllib3

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
        self.max_retries = max(1, int(max_retries))
        # 1Gbps 跨机上传 block delta / 全局 state 可能超过默认 5min read timeout
        http_client = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=30.0, read=float(read_timeout_s)),
            retries=False,  # 应用层自己做带退避的重试，避免与 multipart 叠加混乱
            maxsize=10,
        )
        self.client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=use_secure,
            http_client=http_client,
        )
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def _retry(self, op_name: str, fn):
        last: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(60.0, 2.0 ** (attempt - 1))
                logger.warning(
                    "%s failed attempt %s/%s: %s; retry in %.1fs",
                    op_name,
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
        assert last is not None
        raise last

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        def _do() -> None:
            kwargs = {
                "bucket_name": self.bucket,
                "object_name": key,
                "data": io.BytesIO(data),
                "length": len(data),
                "content_type": content_type,
            }
            if len(data) >= _MULTIPART_THRESHOLD:
                kwargs["part_size"] = _PART_SIZE
            self.client.put_object(**kwargs)

        self._retry(f"put_bytes:{key}", _do)

    def get_bytes(self, key: str) -> bytes:
        def _do() -> bytes:
            resp = self.client.get_object(self.bucket, key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()

        return self._retry(f"get_bytes:{key}", _do)

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

    def object_size(self, key: str) -> int:
        return int(self.client.stat_object(self.bucket, key).size)

    def put_torch(self, key: str, obj: Any) -> int:
        """上传 torch 对象，返回字节数。"""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = tmp.name
        try:
            torch.save(obj, path)
            size = os.path.getsize(path)

            def _do() -> None:
                with open(path, "rb") as f:
                    kwargs = {
                        "bucket_name": self.bucket,
                        "object_name": key,
                        "data": f,
                        "length": size,
                    }
                    if size >= _MULTIPART_THRESHOLD:
                        kwargs["part_size"] = _PART_SIZE
                    self.client.put_object(**kwargs)

            self._retry(f"put_torch:{key}", _do)
            return int(size)
        finally:
            Path(path).unlink(missing_ok=True)

    def get_torch(self, key: str, map_location: Union[str, torch.device] = "cpu") -> Any:
        data = self.get_bytes(key)
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)

    def get_torch_with_size(
        self, key: str, map_location: Union[str, torch.device] = "cpu"
    ) -> Tuple[Any, int]:
        data = self.get_bytes(key)
        obj = torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)
        return obj, len(data)
