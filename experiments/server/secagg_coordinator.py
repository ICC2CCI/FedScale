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
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch

from shared.fixed_point import (
    ROUND1_PUBLIC_SCALE,
    SCALE_COVERAGE,
    aggregate_zq,
    compute_window_scale,
    dequantize_from_zq,
    generate_rademacher_signs,
    hadamard_work_length,
    inverse_hadamard_transform,
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
        # finalize 前只登记对象键，HTTP 通知路径不拉 MinIO
        self.masked_window_refs: Dict[int, Dict[int, Dict[str, Any]]] = {}  # wid → cid → {z_key, vector_len}
        self.masked_blobs: Dict[int, Dict[str, Any]] = {}  # client_id → {z_key, window_ids}
        self.self_masters: Dict[int, bytes] = {}      # client_id → self_master (32 bytes)
        self.survivors: Optional[List[int]] = None    # 冻结后的幸存者集合
        # window_id → {client_id → amax}，非 Hadamard 路径：收齐后取 max_k 算 per-window scale
        self.window_amax: Dict[int, Dict[int, float]] = {}
        # client_id → 全局 amax（Hadamard 路径只收这一个标量，不泄露 per-window L∞）
        self.client_global_amax: Dict[int, float] = {}
        self.global_scale: Optional[float] = None
        # per-window scale：收齐所有 client amax 后才就绪
        self.window_scales: Dict[int, float] = {}
        self.scales_ready: bool = False
        self._init_window_scales()

        logger.info(
            "SecAggCoordinator: round=%s session=%s cohort=%s windows=%s clients=%s",
            round_idx, self.session_id[:16], self.cohort_hash[:16],
            len(windows), self.client_ids,
        )

    def session_payload(self) -> Dict[str, Any]:
        """下发给 client 的 session 字段；self-mask 必须用同一套值。"""
        return {
            "secagg_session_id": self.session_id,
            "cohort_hash": self.cohort_hash,
            "attempt_id": int(self.secagg_plan.attempt_id),
        }

    def _init_window_scales(self) -> None:
        """初始化为 fallback（公开常数）；收齐 client amax 后由 _finalize_window_scales 覆盖。

        fallback 只在 client 还没上报 amax 时使用（例如 key-announce 尚未完成）。
        不读任何单 client 私有统计作为最终 scale。
        """
        fallback = float(self.secagg_plan.quantization_scale)
        if fallback <= 0.0:
            fallback = ROUND1_PUBLIC_SCALE
        for window in self.windows:
            self.window_scales[window.window_id] = fallback
        self.secagg_plan.window_scales = {
            str(k): float(v) for k, v in self.window_scales.items()
        }
        if self.window_scales:
            vals = list(self.window_scales.values())
            logger.info(
                "SecAgg: window scales fallback n=%s min=%e max=%e (waiting for client amax)",
                len(vals), min(vals), max(vals),
            )

    def _apply_uniform_scale(self, scale: float) -> None:
        scale = float(scale)
        self.global_scale = scale
        for window in self.windows:
            self.window_scales[window.window_id] = scale
        self.secagg_plan.quantization_scale = scale
        self.secagg_plan.window_scales = {
            str(k): float(v) for k, v in self.window_scales.items()
        }
        self.scales_ready = True

    def _finalize_global_scale(self) -> None:
        """Hadamard 路径：所有 window 共用一个 scale，只依赖全局 amax。

        scale = max_k(global_amax_k) / Q_max * SCALE_COVERAGE
        不读、不下发 per-window L∞。
        """
        values = [float(v) for v in self.client_global_amax.values() if v is not None]
        if not values:
            self.scales_ready = True
            return
        amax = max(values)
        scale = compute_window_scale(amax, self.secagg_plan.q_max, coverage=SCALE_COVERAGE)
        self._apply_uniform_scale(scale)
        logger.info(
            "SecAgg: global scale finalized (hadamard, no per-window amax) amax=%e scale=%e n_windows=%s",
            amax, scale, len(self.windows),
        )

    def _finalize_window_scales(self) -> None:
        """非 Hadamard：收齐所有 client 的 per-window amax 后，取 max_k 算当轮 scale。

        scale_b = max_k(|delta_{k,b}|) / Q_max * SCALE_COVERAGE
        SCALE_COVERAGE=1.05 → amax 映射到 0.95*Q_max，不 clip。
        所有 client 用同一套 scale（整数域求和的前提）。
        """
        q_max = self.secagg_plan.q_max
        for window in self.windows:
            wid = window.window_id
            amax_per_client = self.window_amax.get(wid, {})
            if not amax_per_client:
                continue
            amax = max(amax_per_client.values())
            self.window_scales[wid] = compute_window_scale(amax, q_max, coverage=SCALE_COVERAGE)
        self.secagg_plan.window_scales = {
            str(k): float(v) for k, v in self.window_scales.items()
        }
        self.scales_ready = True
        vals = list(self.window_scales.values())
        logger.info(
            "SecAgg: window scales finalized (current-round amax) n=%s min=%e max=%e median=%e",
            len(vals), min(vals), max(vals), sorted(vals)[len(vals) // 2],
        )

    # ----------------------------------------------------------------------- #
    # Phase 1: 公钥收集 + relay
    # ----------------------------------------------------------------------- #

    def submit_public_key(
        self,
        client_id: int,
        pk_raw: bytes,
        window_amax: Optional[Dict[Any, float]] = None,
        global_amax: Optional[float] = None,
    ) -> Dict[str, Any]:
        """client 提交 X25519 公钥，并可选上报 scale 统计。

        - 非 Hadamard：window_amax = {str(window_id): max|delta|}，泄露每 window 的 L∞。
        - Hadamard：只收 global_amax 一个标量，所有 window 共用 scale，不泄露 per-window L∞。
        """
        with self.lock:
            self.public_keys[client_id] = pk_raw
            n_keys = len(self.public_keys)
            if global_amax is not None:
                self.client_global_amax[client_id] = float(global_amax)
            # 非 Hadamard 才保留 per-window amax。Hadamard 即使误传也不入库，避免 L∞ 画像。
            if window_amax and not self.secagg_plan.hadamard_enabled:
                for wid_str, amax_val in window_amax.items():
                    try:
                        wid = int(wid_str)
                    except (TypeError, ValueError):
                        continue
                    self.window_amax.setdefault(wid, {})[client_id] = float(amax_val)
            elif window_amax and self.secagg_plan.hadamard_enabled and client_id not in self.client_global_amax:
                # 旧 client 在 Hadamard 下仍报了 per-window amax：只取 max 当全局值，丢弃分窗口明细
                try:
                    collapsed = max(float(v) for v in window_amax.values())
                    self.client_global_amax[client_id] = collapsed
                except ValueError:
                    pass
            all_received = n_keys >= len(self.client_ids)
            if all_received and not self.scales_ready:
                if self.secagg_plan.hadamard_enabled:
                    if self.client_global_amax:
                        self._finalize_global_scale()
                    else:
                        self.scales_ready = True
                elif any(self.window_amax.get(w.window_id) for w in self.windows):
                    self._finalize_window_scales()
                else:
                    self.scales_ready = True
            scales_ready = self.scales_ready
            window_scales = dict(self.window_scales)
            global_scale = self.global_scale
            logger.info(
                "SecAgg: received public key from client %s (%s/%s) scales_ready=%s hadamard=%s global_scale=%s",
                client_id, n_keys, len(self.client_ids), scales_ready,
                self.secagg_plan.hadamard_enabled,
                f"{global_scale:e}" if global_scale is not None else "n/a",
            )

        result: Dict[str, Any] = {
            "received": n_keys,
            "scales_ready": scales_ready,
        }
        result.update(self.session_payload())
        if all_received:
            result["status"] = "ready"
            result["public_keys"] = self._get_peer_keys()
        else:
            result["status"] = "waiting"
        result["window_scales"] = {str(k): float(v) for k, v in window_scales.items()}
        if global_scale is not None:
            result["global_scale"] = float(global_scale)
        return result

    def _get_peer_keys(self) -> Dict[str, str]:
        """返回所有 client 的公钥（hex 编码，用于 JSON 传输）。"""
        return {
            str(cid): pk.hex()
            for cid, pk in self.public_keys.items()
        }

    def get_peer_keys(self) -> Dict[str, Any]:
        """client 获取所有其他 client 的公钥，以及已对齐的 per-window scale。"""
        with self.lock:
            scales_payload = {str(k): float(v) for k, v in self.window_scales.items()}
            if len(self.public_keys) >= len(self.client_ids):
                result: Dict[str, Any] = {
                    "status": "ready",
                    "public_keys": self._get_peer_keys(),
                    "scales_ready": self.scales_ready,
                    "received": len(self.public_keys),
                }
                result.update(self.session_payload())
                if self.scales_ready:
                    result["window_scales"] = scales_payload
                    if self.global_scale is not None:
                        result["global_scale"] = float(self.global_scale)
                return result
            waiting = {
                "status": "waiting",
                "received": len(self.public_keys),
                "scales_ready": self.scales_ready,
            }
            waiting.update(self.session_payload())
            return waiting

    # ----------------------------------------------------------------------- #
    # Phase 3: 收集 masked windows
    # ----------------------------------------------------------------------- #

    def submit_masked_window(
        self,
        client_id: int,
        window_id: int,
        z_kw: torch.Tensor,
    ) -> Dict[str, Any]:
        """client 提交一个 masked window (z_kw) 的内存张量。"""
        with self.lock:
            window_map = self.masked_windows.setdefault(window_id, {})
            window_map[client_id] = z_kw
            n_received = len(window_map)
            all_received = n_received >= len(self.client_ids)

        if all_received:
            return {"status": "ready", "window_id": window_id}
        return {"status": "waiting", "received": n_received}

    def register_masked_window_key(
        self,
        client_id: int,
        window_id: int,
        z_key: str,
        vector_len: int,
    ) -> Dict[str, Any]:
        """只登记 MinIO key，finalize 时再 GET。"""
        with self.lock:
            cmap = self.masked_window_refs.setdefault(int(window_id), {})
            cmap[int(client_id)] = {"z_key": str(z_key), "vector_len": int(vector_len)}
            n_received = len(cmap)
            all_received = n_received >= len(self.client_ids)
        return {
            "status": "ready" if all_received else "waiting",
            "window_id": int(window_id),
            "received": n_received,
        }

    def register_masked_blob(
        self,
        client_id: int,
        z_key: str,
        window_ids: List[int],
    ) -> None:
        """本轮该 client 全部 window 打在一个 blob 里。"""
        ids = [int(w) for w in window_ids]
        with self.lock:
            self.masked_blobs[int(client_id)] = {
                "z_key": str(z_key),
                "window_ids": ids,
                "window_id_set": set(ids),
            }
        logger.info(
            "SecAgg: masked blob from client %s key=%s n_windows=%s",
            client_id, z_key, len(ids),
        )

    def _client_has_all_windows_locked(self, cid: int) -> bool:
        blob = self.masked_blobs.get(cid)
        if blob is not None:
            id_set = blob.get("window_id_set") or set(blob.get("window_ids") or [])
            return all(w.window_id in id_set for w in self.windows)
        for w in self.windows:
            wid = w.window_id
            if cid in self.masked_windows.get(wid, {}):
                continue
            if cid in self.masked_window_refs.get(wid, {}):
                continue
            return False
        return True

    def materialize_masked_windows(self, load_bytes) -> None:
        """从 MinIO（或测试 fake）把登记的 z 载入 masked_windows。load_bytes(key)->bytes。"""
        from shared.fixed_point import unpack_zq
        from shared.secagg_blob import unpack_masked_windows_blob

        bits = int(self.secagg_plan.modulus_bits)
        with self.lock:
            blobs = {cid: dict(info) for cid, info in self.masked_blobs.items()}
            refs = {
                wid: {cid: dict(meta) for cid, meta in cmap.items()}
                for wid, cmap in self.masked_window_refs.items()
            }

        for cid, info in blobs.items():
            data = load_bytes(info["z_key"])
            for wid, vlen, payload in unpack_masked_windows_blob(data):
                z_kw = unpack_zq(payload, int(vlen), bits)
                self.submit_masked_window(int(cid), int(wid), z_kw)

        for wid, cmap in refs.items():
            for cid, meta in cmap.items():
                with self.lock:
                    already = cid in self.masked_windows.get(wid, {})
                if already:
                    continue
                data = load_bytes(meta["z_key"])
                z_kw = unpack_zq(data, int(meta["vector_len"]), bits)
                self.submit_masked_window(int(cid), int(wid), z_kw)

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
                has_all_windows = self._client_has_all_windows_locked(cid)
                has_self_master = cid in self.self_masters
                if has_all_windows and has_self_master:
                    survivors.append(cid)
            self.survivors = sorted(survivors)
            logger.info(
                "SecAgg: survivors=%s (total=%s, q_min=%s)",
                self.survivors, len(self.client_ids), self.secagg_plan.q_min,
            )
            return list(self.survivors)

    @staticmethod
    def _unmask_worker_count(n_windows: int) -> int:
        """并行 unmask worker 数：默认 min(8, CPU)，可用 SECAGG_UNMASK_WORKERS 覆盖。

        按 window 并行，不按模型/层名写死。
        """
        raw = (os.environ.get("SECAGG_UNMASK_WORKERS") or "").strip()
        if raw:
            n = max(1, int(raw))
        else:
            n = min(8, os.cpu_count() or 4)
        return max(1, min(n, int(n_windows) or 1))

    def _unmask_one_window(
        self,
        window: WindowDescriptor,
        survivors: List[int],
        n_survivors: int,
        q: int,
        hadamard: bool,
        self_masters: Dict[int, bytes],
        z_list: List[torch.Tensor],
    ) -> Tuple[str, int, int, torch.Tensor, float, float]:
        """单窗：sum_z → 去 self_mask → dequant → iHadamard。线程安全（无共享可变状态）。"""
        wid = window.window_id
        orig_len = int(window.vector_length)
        work_len = hadamard_work_length(orig_len, hadamard)

        if len(z_list) < self.secagg_plan.q_min:
            raise RuntimeError(
                f"SecAgg: window {wid} has only {len(z_list)} masked values"
            )
        for z in z_list:
            if int(z.numel()) != work_len:
                raise RuntimeError(
                    f"SecAgg: window {wid} z_len={int(z.numel())} != work_len={work_len}"
                )

        sum_z = aggregate_zq(z_list, q)
        sum_B = torch.zeros(work_len, dtype=torch.int64)
        for cid in survivors:
            sm = self_masters.get(cid)
            if sm is None:
                continue
            B_k = compute_self_mask(
                self_master=sm,
                session_id=self.session_id,
                attempt_id=self.secagg_plan.attempt_id,
                client_id=cid,
                window_id=wid,
                window_layout_hash=window.window_layout_hash,
                vector_len=work_len,
                q=q,
            )
            sum_B = (sum_B + B_k) % q

        sum_q = unmask_aggregate(sum_z, sum_B, q)
        scale = self.secagg_plan.get_window_scale(wid)
        delta_slice = dequantize_from_zq(sum_q, scale, q)
        work_amax = float(delta_slice.abs().max().item()) if delta_slice.numel() else 0.0

        if hadamard and work_len > 0:
            seed = self.secagg_plan.hadamard_seed + wid
            signs = generate_rademacher_signs(work_len, seed)
            delta_slice = inverse_hadamard_transform(delta_slice, signs)
        delta_slice = delta_slice.reshape(-1)[:orig_len].contiguous()
        delta_slice = delta_slice / float(n_survivors)
        orig_amax = float(delta_slice.abs().max().item()) if delta_slice.numel() else 0.0
        return window.key_name, int(window.start), int(window.end), delta_slice, work_amax, orig_amax

    def aggregate_and_unmask(self) -> Dict[str, List[Tuple[int, int, torch.Tensor]]]:
        """聚合所有 window：sum_z → 移除 self_mask → dequant → agg_delta。

        返回: {key_name: [(start, end, delta_slice), ...]}（与 block_delta 格式一致）

        如果 |U*| < q_min，抛出 RuntimeError。
        Window 互相独立，默认按 CPU 并行（SHAKE + iHadamard 释放 GIL）。
        """
        if self.survivors is None:
            self.freeze_survivors()

        if len(self.survivors) < self.secagg_plan.q_min:
            raise RuntimeError(
                f"SecAgg: abort, survivors={len(self.survivors)} < q_min={self.secagg_plan.q_min}"
            )

        q = self.secagg_plan.modulus_q
        n_survivors = len(self.survivors)
        hadamard = bool(self.secagg_plan.hadamard_enabled)
        survivors = list(self.survivors)

        with self.lock:
            self_masters = dict(self.self_masters)
            jobs: List[Tuple[WindowDescriptor, List[torch.Tensor]]] = []
            for window in self.windows:
                window_map = self.masked_windows.get(window.window_id, {})
                z_list = [window_map[cid] for cid in survivors if cid in window_map]
                jobs.append((window, z_list))

        workers = self._unmask_worker_count(len(jobs))
        results: List[Optional[Tuple[str, int, int, torch.Tensor, float, float]]] = [
            None
        ] * len(jobs)
        t0 = time.monotonic()
        prev_threads = torch.get_num_threads()
        try:
            # 并行时把 BLAS 收到 1，避免 8 worker × 全核 OpenMP 过订阅
            if workers > 1:
                torch.set_num_threads(1)
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="secagg-unmask",
                ) as ex:
                    futs = {
                        ex.submit(
                            self._unmask_one_window,
                            window,
                            survivors,
                            n_survivors,
                            q,
                            hadamard,
                            self_masters,
                            z_list,
                        ): i
                        for i, (window, z_list) in enumerate(jobs)
                    }
                    for fut in as_completed(futs):
                        results[futs[fut]] = fut.result()
            else:
                for i, (window, z_list) in enumerate(jobs):
                    results[i] = self._unmask_one_window(
                        window, survivors, n_survivors, q, hadamard,
                        self_masters, z_list,
                    )
        finally:
            torch.set_num_threads(prev_threads)

        orig_amaxs: List[float] = []
        work_amaxs: List[float] = []
        agg_delta: Dict[str, List[Tuple[int, int, torch.Tensor]]] = {}
        for item in results:
            assert item is not None
            key_name, start, end, delta_slice, work_amax, orig_amax = item
            work_amaxs.append(work_amax)
            orig_amaxs.append(orig_amax)
            agg_delta.setdefault(key_name, []).append((start, end, delta_slice))

        orig_max = max(orig_amaxs) if orig_amaxs else 0.0
        work_max = max(work_amaxs) if work_amaxs else 0.0
        scale_ref = float(self.secagg_plan.quantization_scale or 0.0)
        clip_bound = float(self.secagg_plan.q_max) * scale_ref if scale_ref > 0 else 0.0
        logger.info(
            "SecAgg: aggregation complete windows=%s survivors=%s hadamard=%s "
            "workers=%s unmask_s=%.2f work_amax=%e orig_amax=%e clip_bound=%e session=%s",
            len(self.windows), n_survivors, hadamard, workers, time.monotonic() - t0,
            work_max, orig_max, clip_bound, self.session_id[:16],
        )
        if clip_bound > 0 and orig_max > 0.05 and orig_max > 0.25 * clip_bound:
            logger.warning(
                "SecAgg: orig_amax=%e close to Q_max*scale=%e — self-mask 可能未抵消",
                orig_max, clip_bound,
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
                "scales_ready": self.scales_ready,
                "n_window_scales": len(self.window_scales),
            }
