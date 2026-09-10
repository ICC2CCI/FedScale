"""MinIO / S3 读写封装。

针对跨机上传偶发「假死」（TCP 半开 / 僵死 keep-alive 连接）：
- 单请求短 read timeout + TCP keepalive，尽快发现死连接
- 硬超时后立刻 clear 连接池打断卡住的 socket（不再被 executor join 拖到 read_timeout）
- multipart 串行上传，失败语义清晰；重建客户端后重试
"""
from __future__ import annotations

import io
import json
import logging
import os
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

logger = logging.getLogger("minio_client")

# 16MiB part：单 part 在 GbE 上应秒级完成；卡住时由短 read timeout 触发失败
_MULTIPART_THRESHOLD = 16 * 1024 * 1024
_PART_SIZE = 16 * 1024 * 1024
# 单 HTTP 请求（含单个 part）读超时；假死时不必等几百秒
_DEFAULT_READ_TIMEOUT_S = 45.0
# 整对象一次 attempt 的墙钟上限（正常 ~250MiB 上传 <10s）
_DEFAULT_ATTEMPT_TIMEOUT_S = 90.0


class MinIOClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: Optional[bool] = None,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
        max_retries: int = 6,
        attempt_timeout_s: float = _DEFAULT_ATTEMPT_TIMEOUT_S,
    ) -> None:
        self._endpoint_raw = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure_arg = secure
        self.bucket = bucket
        self.max_retries = max(1, int(max_retries))
        self.attempt_timeout_s = float(attempt_timeout_s)
        self.read_timeout_s = float(read_timeout_s)
        self.client = None  # set by _rebuild_client
        self._http = None
        self._rebuild_client()
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def _parse_endpoint(self) -> Tuple[str, bool]:
        endpoint = self._endpoint_raw.strip()
        secure = self._secure_arg
        if endpoint.startswith("https://"):
            host = endpoint[len("https://") :]
            use_secure = True if secure is None else secure
        elif endpoint.startswith("http://"):
            host = endpoint[len("http://") :]
            use_secure = False if secure is None else secure
        else:
            host = endpoint
            use_secure = False if secure is None else secure
        return host, use_secure

    def _clear_http(self, http) -> None:
        """强制关掉连接池里的 socket，打断卡住的 send/recv。"""
        if http is None:
            return
        try:
            http.clear()
        except Exception as exc:  # noqa: BLE001
            logger.warning("http pool clear failed: %s", exc)

    def _rebuild_client(self) -> None:
        """丢弃旧连接池，重建 Minio 客户端（上传假死后必须换连接）。"""
        from minio import Minio
        import urllib3
        from urllib3.connection import HTTPConnection

        old_http = self._http
        host, use_secure = self._parse_endpoint()

        # TCP keepalive：半开连接可在几十秒内被内核发现，而不是干等到 read timeout
        socket_options = list(HTTPConnection.default_socket_options)
        socket_options.append((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1))
        if hasattr(socket, "TCP_KEEPIDLE"):
            socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30))
        if hasattr(socket, "TCP_KEEPINTVL"):
            socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10))
        if hasattr(socket, "TCP_KEEPCNT"):
            socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3))

        http_client = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=10.0, read=float(self.read_timeout_s)),
            retries=False,
            maxsize=2,
            socket_options=socket_options,
        )
        self._http = http_client
        self.client = Minio(
            host,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=use_secure,
            http_client=http_client,
        )
        # 先挂上新 client，再清旧池，避免并发读到半清状态
        self._clear_http(old_http)
        logger.info(
            "MinIO client (re)built endpoint=%s read_timeout=%.0fs attempt_timeout=%.0fs keepalive=on",
            host,
            self.read_timeout_s,
            self.attempt_timeout_s,
        )

    def _run_attempt(self, fn, timeout_s: float):
        """跑一次 MinIO 操作；超时后立刻 clear 连接池，不 join 卡住的工作线程。"""
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(fn)
        try:
            result = fut.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            # 关键：立刻掐断旧 socket，否则 shutdown(wait=True) 会再等到 read_timeout
            hung_http = self._http
            try:
                self._rebuild_client()
            except Exception as rebuild_exc:  # noqa: BLE001
                logger.warning("rebuild after hard timeout failed: %s", rebuild_exc)
                self._clear_http(hung_http)
            pool.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"MinIO op exceeded hard timeout {timeout_s:.0f}s") from exc
        except Exception:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
            return result

    def _retry(self, op_name: str, fn, attempt_timeout_s: Optional[float] = None):
        last: Optional[Exception] = None
        timeout_s = float(attempt_timeout_s if attempt_timeout_s is not None else self.attempt_timeout_s)
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._run_attempt(fn, timeout_s)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(8.0, 1.0 * (1.5 ** (attempt - 1)))
                logger.warning(
                    "%s failed attempt %s/%s: %s; rebuild client, retry in %.1fs",
                    op_name,
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_s,
                )
                try:
                    self._rebuild_client()
                except Exception as rebuild_exc:  # noqa: BLE001
                    logger.warning("rebuild client failed: %s", rebuild_exc)
                time.sleep(sleep_s)
        assert last is not None
        raise last

    def _put_object_kwargs(self, key: str, data, length: int, content_type: Optional[str] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "bucket_name": self.bucket,
            "object_name": key,
            "data": data,
            "length": length,
            # 串行 part：单连接假死时语义更清晰，避免 3 路并行放大卡住面
            "num_parallel_uploads": 1,
        }
        if content_type is not None:
            kwargs["content_type"] = content_type
        if length >= _MULTIPART_THRESHOLD:
            kwargs["part_size"] = _PART_SIZE
        return kwargs

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        def _do() -> None:
            self.client.put_object(**self._put_object_kwargs(key, io.BytesIO(data), len(data), content_type))

        # 正常路径按 5 MiB/s 估；假死由短 read timeout / 硬超时打断
        size_timeout = max(self.attempt_timeout_s, (len(data) / (5 * 1024 * 1024)) + 30.0)
        self._retry(f"put_bytes:{key}", _do, attempt_timeout_s=size_timeout)

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
            size_timeout = max(self.attempt_timeout_s, (size / (5 * 1024 * 1024)) + 30.0)

            def _do() -> None:
                with open(path, "rb") as f:
                    self.client.put_object(**self._put_object_kwargs(key, f, size))

            self._retry(f"put_torch:{key}", _do, attempt_timeout_s=size_timeout)
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
