"""Resilient Flower 1.28 heartbeat sender for the cross-cloud REST path."""

import logging
import random
import signal
import threading
from collections.abc import Callable

import grpc
import requests

from flwr.common.constant import (
    HEARTBEAT_BASE_MULTIPLIER,
    HEARTBEAT_CALL_TIMEOUT,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_RANDOM_RANGE,
)
from flwr.common.retry_invoker import RetryInvoker, exponential
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.heartbeat_pb2 import SendAppHeartbeatRequest
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub


_LOGGER = logging.getLogger(__name__)


class HeartbeatFailure(Exception):
    """Exception raised when a heartbeat fails."""


class HeartbeatSender:
    """Send heartbeats and survive transient REST/Skupper disconnects."""

    def __init__(self, heartbeat_fn: Callable[[], bool]) -> None:
        self.heartbeat_fn = heartbeat_fn
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._retry_invoker = RetryInvoker(
            lambda: exponential(max_delay=20),
            HeartbeatFailure,
            max_tries=None,
            max_time=None,
            wait_function=self._stop_event.wait,
        )

    def start(self) -> None:
        if self._thread.is_alive():
            raise RuntimeError("Heartbeat sender is already running.")
        if self._stop_event.is_set():
            raise RuntimeError("Cannot start a stopped heartbeat sender.")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive() and not self._stop_event.is_set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._retry_invoker.invoke(self._heartbeat)
            except (requests.RequestException, OSError, ConnectionError) as exc:
                # Flower 1.28 only retries HeartbeatFailure.  A transient REST
                # disconnect otherwise escapes the background thread and the
                # SuperLink later marks this node unavailable.  Keep the same
                # sender alive and let the next request establish a connection.
                _LOGGER.warning(
                    "Transient heartbeat transport error; retrying: %s", exc
                )
                self._stop_event.wait(5.0)
                continue

            rd = random.uniform(*HEARTBEAT_RANDOM_RANGE)
            next_interval = HEARTBEAT_DEFAULT_INTERVAL - HEARTBEAT_CALL_TIMEOUT
            next_interval *= HEARTBEAT_BASE_MULTIPLIER + rd
            self._stop_event.wait(next_interval)

    def _heartbeat(self) -> None:
        if not self._stop_event.is_set() and not self.heartbeat_fn():
            raise HeartbeatFailure


def make_app_heartbeat_fn_grpc(
    stub: ServerAppIoStub | ClientAppIoStub, token: str
) -> Callable[[], bool]:
    req = SendAppHeartbeatRequest(token=token)

    def fn() -> bool:
        try:
            res = stub.SendAppHeartbeat(req)
        except grpc.RpcError as exc:
            if exc.code() in {
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            }:
                return False
            raise
        if not res.success:
            signal.raise_signal(signal.SIGINT)
        return True

    return fn
