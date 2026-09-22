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
# 单 HTTP 请求读超时。45s 在全量 ~6.5GiB PUT 上会把慢 ACK 当成断连
# （BASE-S2-3B Round 6：约 5 分钟 RemoteDisconnected，重试耗尽后整轮退出）。
_DEFAULT_READ_TIMEOUT_S = 180.0
# 整对象一次 attempt 的墙钟上限（小对象）。大对象另按体积放宽。
_DEFAULT_ATTEMPT_TIMEOUT_S = 90.0
# 超过该体积则分段 PUT 再 compose 成原 key，避免一条连接扛数 GiB。
_CHUNK_PUT_BYTES = 128 * 1024 * 1024


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
        self._op_pool: Optional[ThreadPoolExecutor] = None
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
            maxsize=8,
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

    def _ensure_op_pool(self) -> ThreadPoolExecutor:
        if self._op_pool is None:
            self._op_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minio-op")
        return self._op_pool

    def _drop_op_pool(self) -> None:
        pool = self._op_pool
        self._op_pool = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _run_attempt(self, fn, timeout_s: float):
        """跑一次 MinIO 操作；超时后立刻 clear 连接池，不 join 卡住的工作线程。"""
        pool = self._ensure_op_pool()
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            hung_http = self._http
            try:
                self._rebuild_client()
            except Exception as rebuild_exc:  # noqa: BLE001
                logger.warning("rebuild after hard timeout failed: %s", rebuild_exc)
                self._clear_http(hung_http)
            self._drop_op_pool()
            raise TimeoutError(f"MinIO op exceeded hard timeout {timeout_s:.0f}s") from exc
        except Exception:
            self._drop_op_pool()
            raise

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
        self._retry(f"put_bytes:{key}", _do, attempt_timeout_s=self._size_timeout_s(len(data)))

    def _put_file(self, key: str, path: str, size: int) -> None:
        """上传本地文件。大于 128MiB 时分段 PUT，再 compose 成同一个 key。"""
        if size <= _CHUNK_PUT_BYTES:
            def _do() -> None:
                with open(path, "rb") as f:
                    self.client.put_object(**self._put_object_kwargs(key, f, size))

            self._retry(f"put_torch:{key}", _do, attempt_timeout_s=self._size_timeout_s(size))
            return

        n_parts = (size + _CHUNK_PUT_BYTES - 1) // _CHUNK_PUT_BYTES
        part_keys: list[str] = []
        logger.info(
            "chunked put %s size=%.1fMiB parts=%s",
            key, size / (1024 * 1024), n_parts,
        )
        try:
            with open(path, "rb") as f:
                for i in range(n_parts):
                    chunk = f.read(_CHUNK_PUT_BYTES)
                    if not chunk:
                        break
                    part_key = f"{key}.part-{i:04d}"
                    self.put_bytes(part_key, chunk)
                    part_keys.append(part_key)
            from minio.commonconfig import ComposeSource

            sources = [ComposeSource(self.bucket, pk) for pk in part_keys]

            def _compose() -> None:
                self.client.compose_object(self.bucket, key, sources)

            self._retry(f"compose:{key}", _compose, attempt_timeout_s=180.0)
        finally:
            for pk in part_keys:
                try:
                    self.client.remove_object(self.bucket, pk)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("remove chunk %s failed: %s", pk, exc)

    def _size_timeout_s(self, size_bytes: int) -> float:
        """按体积估 attempt 硬超时（与 put 一致：约 5 MiB/s + 30s 余量）。

        dense 全量 delta / global_state 可达数 GiB；默认 90s 只够小对象，
        不按 size 放宽会在 get 中途被硬超时掐死（BASE-S2-3B R1 复现）。
        """
        size = max(0, int(size_bytes))
        return max(self.attempt_timeout_s, (size / (5 * 1024 * 1024)) + 30.0)

    def get_bytes(self, key: str) -> bytes:
        size_timeout = self.attempt_timeout_s
        try:
            size_timeout = self._size_timeout_s(self.object_size(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("stat before get_bytes:%s failed (%s); using default timeout", key, exc)

        def _do() -> bytes:
            resp = self.client.get_object(self.bucket, key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()

        return self._retry(f"get_bytes:{key}", _do, attempt_timeout_s=size_timeout)

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

    def put_torch(self, key: str, obj: Any, *, via_memory: bool = False) -> int:
        """上传 torch 对象，返回字节数。

        via_memory=True：序列化到内存再 PUT，避免临界路径上的临时文件写盘
        （适合 global_delta 这类百兆级对象）。全量 global_state 仍走 tempfile，
        以免 7B 级 state 再复制一份到 RAM。
        """
        if via_memory:
            buf = io.BytesIO()
            torch.save(obj, buf)
            data = buf.getvalue()
            self.put_bytes(key, data)
            return int(len(data))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = tmp.name
        try:
            torch.save(obj, path)
            size = os.path.getsize(path)
            self._put_file(key, path, size)
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
