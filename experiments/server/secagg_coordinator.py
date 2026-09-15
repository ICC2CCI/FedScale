"""Server 端 SecAgg 协调器。

职责：
    1. 收集 client 的 X25519 公钥并 relay 给其他 client
    2. 收集 client 上传的 masked window (z_k)
    3. 收集 client 上传的 self_master
    4. 聚合: sum_z → 移除 self_mask → dequant → agg_delta
    5. 掉线处理: 剩余 >= q_min 则聚合，否则 abort

2-client 无掉线场景（当前实现）：
    z_A = q_A + R_AB + B_A
    z_B = q_B - R_AB + B_B
    sum_z = z_A + z_B = (q_A + q_B) + (B_A + B_B)   (R_AB 抵消)
    sum_q = sum_z - (B_A + B_B) = q_A + q_B
    agg_delta = dequant(sum_q) / N

安全性：
    - server 不知道 R_AB（DH 保护）
    - server 知道 self_master → 知道 B_k，但只能算出 q_k + R_AB，无法分离
    - 只有 pairwise + self 都泄露才能还原 q_k → 安全
"""
from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

from shared.fixed_point import (
    aggregate_zq,
    dequantize_from_zq,
    unmask_aggregate,
)
from shared.protocol import (
    SecAggPlan,
    WindowDescriptor,
    compute_window_layout_hash,
)
from shared.secagg_crypto import (
    compute_self_mask,
    compute_session_id,
    compute_cohort_hash,
)

logger = logging.getLogger(__name__)


class SecAggCoordinator:
    """Server 端 SecAgg 协调器（per-round, per-attempt）。"""

    def __init__(
        self,
        round_idx: int,
        successful_round_index: int,
        client_ids: List[int],
        windows: List[WindowDescriptor],
        layout_hash: str,
        mask_hash: str,
        secagg_plan: SecAggPlan,
        job_id: str = "fedscale-job",
        base_model_version: str = "v0",
    ) -> None:
        self.round_idx = round_idx
        self.client_ids = sorted(client_ids)
        self.windows = windows
        self.secagg_plan = secagg_plan
        self.lock = threading.Lock()

        # 计算 session_id 和 cohort_hash
        self.cohort_hash = compute_cohort_hash(self.client_ids)
        self.session_id = compute_session_id(
            job_id=job_id,
            round_idx=round_idx,
            successful_round_index=successful_round_index,
            base_model_version=base_model_version,
            cohort_hash=self.cohort_hash,
            layout_hash=layout_hash,
            mask_hash=mask_hash,
        )
        self.secagg_plan.secagg_session_id = self.session_id
        self.secagg_plan.cohort_hash = self.cohort_hash
        self.secagg_plan.mask_hash = mask_hash

        # window_plan_hash
        wph_data = b"window_plan|" + b"|".join(
            w.window_layout_hash.encode() for w in windows
        )
        self.secagg_plan.window_plan_hash = hashlib.sha256(wph_data).hexdigest()

        # 状态
        self.public_keys: Dict[int, bytes] = {}      # client_id → pk_raw (32 bytes)
        self.masked_windows: Dict[int, Dict[int, torch.Tensor]] = {}  # window_id → {client_id → z_kw}
        self.self_masters: Dict[int, bytes] = {}      # client_id → self_master (32 bytes)
        self.survivors: Optional[List[int]] = None    # 冻结后的幸存者集合

        logger.info(
            "SecAggCoordinator: round=%s session=%s cohort=%s windows=%s clients=%s",
            round_idx, self.session_id[:16], self.cohort_hash[:16],
            len(windows), self.client_ids,
        )

    # ----------------------------------------------------------------------- #
    # Phase 1: 公钥收集 + relay
    # ----------------------------------------------------------------------- #

    def submit_public_key(self, client_id: int, pk_raw: bytes) -> Dict[str, Any]:
        """client 提交自己的 X25519 公钥。"""
        with self.lock:
            self.public_keys[client_id] = pk_raw
            logger.info(
                "SecAgg: received public key from client %s (%s/%s)",
                client_id, len(self.public_keys), len(self.client_ids),
            )
            all_received = len(self.public_keys) >= len(self.client_ids)

        if all_received:
            return {"status": "ready", "public_keys": self._get_peer_keys()}
        return {"status": "waiting", "received": len(self.public_keys)}

    def _get_peer_keys(self) -> Dict[str, str]:
        """返回所有 client 的公钥（hex 编码，用于 JSON 传输）。"""
        return {
            str(cid): pk.hex()
            for cid, pk in self.public_keys.items()
        }

    def get_peer_keys(self) -> Dict[str, Any]:
        """client 获取所有其他 client 的公钥。"""
        with self.lock:
            if len(self.public_keys) >= len(self.client_ids):
                return {"status": "ready", "public_keys": self._get_peer_keys()}
            return {"status": "waiting", "received": len(self.public_keys)}

    # ----------------------------------------------------------------------- #
    # Phase 3: 收集 masked windows
    # ----------------------------------------------------------------------- #

    def submit_masked_window(
        self,
        client_id: int,
        window_id: int,
        z_kw: torch.Tensor,
    ) -> Dict[str, Any]:
        """client 提交一个 masked window (z_kw)。"""
        with self.lock:
            window_map = self.masked_windows.setdefault(window_id, {})
            window_map[client_id] = z_kw
            n_received = len(window_map)
            logger.info(
                "SecAgg: masked window %s from client %s (%s/%s)",
                window_id, client_id, n_received, len(self.client_ids),
            )
            all_received = n_received >= len(self.client_ids)

        if all_received:
            return {"status": "ready", "window_id": window_id}
        return {"status": "waiting", "received": n_received}

    # ----------------------------------------------------------------------- #
    # Phase 4: 收集 self_master + 聚合 unmask
    # ----------------------------------------------------------------------- #

    def submit_self_master(self, client_id: int, self_master: bytes) -> None:
        """client 提交 self_master（32 bytes）。

        2-client 无掉线场景：server 用 self_master 生成 self_mask，从 sum_z 中减去。
        安全性：server 不知道 pairwise mask（DH 保护），所以即使知道 self_master
        也只能算出 q_k + R_kl，无法分离 q_k。
        """
        with self.lock:
            self.self_masters[client_id] = self_master
            logger.info(
                "SecAgg: self_master from client %s (%s/%s)",
                client_id, len(self.self_masters), len(self.client_ids),
            )

    def all_self_masters_received(self) -> bool:
        with self.lock:
            return len(self.self_masters) >= len(self.client_ids)

    def freeze_survivors(self) -> List[int]:
        """冻结幸存者集合 U*（所有提交了 masked window + self_master 的 client）。

        2-client 无掉线: U* = all client_ids
        掉线: U* = 完成所有 window 的 client（简化：用提交了 self_master 的）
        """
        with self.lock:
            # 检查哪些 client 提交了所有 window
            survivors = []
            for cid in self.client_ids:
                has_all_windows = all(
                    cid in self.masked_windows.get(w.window_id, {})
                    for w in self.windows
                )
                has_self_master = cid in self.self_masters
                if has_all_windows and has_self_master:
                    survivors.append(cid)
            self.survivors = sorted(survivors)
            logger.info(
                "SecAgg: survivors=%s (total=%s, q_min=%s)",
                self.survivors, len(self.client_ids), self.secagg_plan.q_min,
            )
            return list(self.survivors)

    def aggregate_and_unmask(self) -> Dict[str, List[Tuple[int, int, torch.Tensor]]]:
        """聚合所有 window：sum_z → 移除 self_mask → dequant → agg_delta。

        返回: {key_name: [(start, end, delta_slice), ...]}（与 block_delta 格式一致）

        如果 |U*| < q_min，抛出 RuntimeError。
        """
        if self.survivors is None:
            self.freeze_survivors()

        if len(self.survivors) < self.secagg_plan.q_min:
            raise RuntimeError(
                f"SecAgg: abort, survivors={len(self.survivors)} < q_min={self.secagg_plan.q_min}"
            )

        q = self.secagg_plan.modulus_q
        scale = self.secagg_plan.quantization_scale
        n_survivors = len(self.survivors)

        # 按 window 聚合
        agg_delta: Dict[str, List[Tuple[int, int, torch.Tensor]]] = {}
        for window in self.windows:
            wid = window.window_id

            # 1. 收集所有 survivor 的 z_kw
            z_list = []
            with self.lock:
                window_map = self.masked_windows.get(wid, {})
                for cid in self.survivors:
                    if cid in window_map:
                        z_list.append(window_map[cid])

            if len(z_list) < self.secagg_plan.q_min:
                raise RuntimeError(
                    f"SecAgg: window {wid} has only {len(z_list)} masked values"
                )

            # 2. 在 Z_q 中求和
            sum_z = aggregate_zq(z_list, q)

            # 3. 生成并求和 self_mask
            sum_B = torch.zeros(window.vector_length, dtype=torch.int64)
            with self.lock:
                for cid in self.survivors:
                    sm = self.self_masters.get(cid)
                    if sm is None:
                        continue
                    B_k = compute_self_mask(
                        self_master=sm,
                        session_id=self.session_id,
                        attempt_id=self.secagg_plan.attempt_id,
                        client_id=cid,
                        window_id=wid,
                        window_layout_hash=window.window_layout_hash,
                        vector_len=window.vector_length,
                        q=q,
                    )
                    sum_B = (sum_B + B_k) % q

            # 4. 移除 self_mask
            sum_q = unmask_aggregate(sum_z, sum_B, q)

            # 5. dequant → delta
            delta_slice = dequantize_from_zq(sum_q, scale, q)

            # 6. 加权平均（等权：delta / N）
            delta_slice = delta_slice / float(n_survivors)

            agg_delta.setdefault(window.key_name, []).append(
                (window.start, window.end, delta_slice)
            )

        logger.info(
            "SecAgg: aggregation complete, %s windows, %s survivors",
            len(self.windows), n_survivors,
        )
        return agg_delta

    # ----------------------------------------------------------------------- #
    # 状态查询
    # ----------------------------------------------------------------------- #

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "round_idx": self.round_idx,
                "session_id": self.session_id,
                "attempt_id": self.secagg_plan.attempt_id,
                "public_keys_received": len(self.public_keys),
                "total_clients": len(self.client_ids),
                "masked_windows": {
                    str(wid): len(wm) for wid, wm in self.masked_windows.items()
                },
                "self_masters_received": len(self.self_masters),
                "survivors": self.survivors,
                "q_min": self.secagg_plan.q_min,
            }
