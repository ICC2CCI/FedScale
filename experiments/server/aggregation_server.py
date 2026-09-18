"""Central Server：REST API + S3R12v3 block FedAvg 聚合。"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.block_scheduler import BlockScheduler
from shared.block_selection import apply_block_delta, resolve_transfer_dtype, _slice_to_fp32
from shared.block_crypto import (
    SEC_PAYLOAD_VERSION,
    decode_sec_block_payload,
    is_sec_payload,
)
from shared.minio_client import MinIOClient
from shared.protocol import (
    DEFAULT_BUCKET,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_COVERAGE_H,
    DEFAULT_MEMORY_DECAY,
    DEFAULT_NUM_CLIENTS,
    DEFAULT_NUM_ROUNDS,
    DEFAULT_SEED,
    DEFAULT_TRANSFER_DTYPE,
    RoundPlan,
    SecAggPlan,
    WindowDescriptor,
    build_window_descriptors,
    agg_block_done_key,
    agg_block_key,
    epoch_seed_from_plan,
    global_delta_key,
    global_state_key,
    plan_key,
    selected_from_jsonable,
    upload_block_key,
    upload_blocks_key,
)
from shared.run_config import apply_to_args, load_run_config, snapshot_effective_config
from shared.state_dict_utils import cpu_state, floating_elem_count
from shared.fixed_point import (
    DEFAULT_MODULUS_BITS,
    DEFAULT_Q,
    DEFAULT_Q_MAX,
    DEFAULT_SCALE,
    ROUND1_PUBLIC_SCALE,
    compute_q_max,
)
from server.secagg_coordinator import SecAggCoordinator

logger = logging.getLogger("aggregation_server")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _arg_was_set_cli(flag: str) -> bool:
    """检查 sys.argv 中是否显式出现了某个 --flag（支持 - 与 _）。"""
    import sys

    target = flag.replace("_", "-")
    for a in sys.argv[1:]:
        name = a.lstrip("-").split("=", 1)[0]
        if name == flag or name == target:
            return True
    return False


def _round_s(x: float) -> float:
    return round(float(x), 3)


class UploadCompleteBody(BaseModel):
    num_examples: int = Field(ge=1)
    train_loss: float = 0.0
    # 在线 eval（客户端本地算完上报；可选）
    eval_loss: Optional[float] = None
    # 客户端在 upload 完成前可上报的分段耗时（秒）
    timings: Dict[str, float] = Field(default_factory=dict)
    block_energies: List[float] = Field(default_factory=list)


class ClientTimingBody(BaseModel):
    """聚合完成后客户端补报完整 round 耗时。"""

    timings: Dict[str, float] = Field(default_factory=dict)


class BlockUploadBody(BaseModel):
    """流式 per-block pipeline：client 上传一个 block 后通知 server 的 body。"""
    num_examples: int = 1
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    block_energies: List[float] = Field(default_factory=list)


class AggregationServer:
    def __init__(
        self,
        minio: MinIOClient,
        num_clients: int,
        num_rounds: int,
        seed: int,
        coverage_h: int,
        results_dir: Path,
        init_state: Dict[str, torch.Tensor],
        transfer_dtype_name: str = DEFAULT_TRANSFER_DTYPE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        memory_decay: float = DEFAULT_MEMORY_DECAY,
        slots_per_round: int = 1,
        write_full_global_every_n_rounds: int = 1,
        client_upload_timeout_s: float = 1800.0,
        min_clients_to_aggregate: int = 0,
        round_deadline_s: float = 0.0,
        auth_token: str = "",
        resume_from_round: int = 0,
        run_config_snapshot: Optional[Dict[str, Any]] = None,
        minio_retention_recent_uploads: int = 0,
        selected_client_ids: Optional[List[int]] = None,
        delta_norm_clip: float = 0.0,
        delta_norm_reject: float = 0.0,
        compressor: str = "public_random",
        rho: float = 0.0,
        sec_upload_privacy: bool = False,
        always_on_threshold: int = 4096,
        secagg_enabled: bool = False,
        secagg_modulus_bits: int = DEFAULT_MODULUS_BITS,
        secagg_scale: float = 0.0,
        secagg_stochastic_rounding: bool = False,
        secagg_q_min: int = 0,
        secagg_hadamard: bool = False,
    ) -> None:
        self.minio = minio
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.seed = seed
        self.coverage_h = coverage_h
        self.slots_per_round = max(1, int(slots_per_round))
        self.block_size = block_size
        self.memory_decay = float(memory_decay)
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.global_state = cpu_state(init_state)
        self.transfer_dtype = resolve_transfer_dtype(
            transfer_dtype_name, ref_state=self.global_state
        )
        logger.info(
            "transfer_dtype=%s (arg=%s)",
            self.transfer_dtype,
            transfer_dtype_name,
        )
        self.scheduler = BlockScheduler(
            self.global_state,
            seed=seed,
            coverage_h=coverage_h,
            block_size=block_size,
            slots_per_round=self.slots_per_round,
            compressor=compressor,
            rho=rho,
            always_on_threshold=always_on_threshold,
        )
        self.compressor = self.scheduler.compressor
        self.rho = self.scheduler.rho
        # IO-1：全量 global_state 写盘频率
        # 1=每轮都写（旧行为/联调兜底）；N>1=每 N 轮写一次；0=只写 round-0 与最终一轮
        self.write_full_global_every_n_rounds = int(write_full_global_every_n_rounds)
        # SYNC-1：最低参与人数；0 表示必须等齐 num_clients（旧行为）
        self.min_clients_to_aggregate = int(min_clients_to_aggregate or 0)
        self.effective_min_clients = (
            self.min_clients_to_aggregate
            if self.min_clients_to_aggregate > 0
            else self.num_clients
        )
        # SYNC-2：本轮聚合宽限上限（超时后用已上传子集聚合，前提满足最低人数）
        self.round_deadline_s = float(round_deadline_s or 0.0)
        # SEC-0：控制面共享 token；空字符串=不鉴权
        self.auth_token = auth_token or ""
        # RES-1：断点续训
        self.resume_from_round = int(resume_from_round or 0)
        self.run_config_snapshot = run_config_snapshot or {}
        # OPS-2：MinIO 生命周期；0=不清理
        self.minio_retention_recent_uploads = int(minio_retention_recent_uploads or 0)
        # OPS-3：Client 选择（每轮抽样）。None=全部参与（旧行为）；list=只这些 client
        self.selected_client_ids: Optional[List[int]] = (
            list(selected_client_ids) if selected_client_ids is not None else None
        )
        # SEC-4：异常更新防护。delta_norm_clip>0 时裁剪；delta_norm_reject>0 时范数超过即剔除该 client
        self.delta_norm_clip = float(delta_norm_clip or 0.0)
        self.delta_norm_reject = float(delta_norm_reject or 0.0)
        # SEC-1/2/3：上传隐私（gidx 化 + 加密 + 末 block 填充）
        # server 端只做"解码"：从 SEC payload 还原出 block_delta。
        # 兼容旧格式：若 payload 不是 SEC 格式，走旧路径。
        self.sec_upload_privacy = bool(sec_upload_privacy)

        # B-6: SecAgg (Windowed Secure Aggregation v2)
        self.secagg_enabled = bool(secagg_enabled)
        self.secagg_modulus_bits = int(secagg_modulus_bits)
        self.secagg_q = 1 << self.secagg_modulus_bits
        self.secagg_q_max = compute_q_max(self.secagg_modulus_bits, n_clients=max(num_clients, 2))
        self.secagg_scale = float(secagg_scale)  # 0=per-window 当轮 amax；>0=固定全局 scale
        self.secagg_stochastic_rounding = bool(secagg_stochastic_rounding)
        self.secagg_q_min = int(secagg_q_min) if secagg_q_min > 0 else num_clients
        self.secagg_hadamard = bool(secagg_hadamard)
        # per-round SecAgg coordinator
        self.secagg_coordinators: Dict[int, SecAggCoordinator] = {}
        if self.secagg_enabled:
            logger.info(
                "SecAgg enabled: modulus_bits=%s q=%s q_max=%s scale_mode=%s q_min=%s",
                self.secagg_modulus_bits, self.secagg_q, self.secagg_q_max,
                ("fixed:%g" % self.secagg_scale) if self.secagg_scale > 0 else "per-window-current-round-amax",
                self.secagg_q_min,
            )

        self.current_round = 1
        # A-6: successful_round_index — only incremented after successful aggregation + commit
        # (spec section 2.1: only count as successful round if all conditions met)
        # Failed/aborted rounds do NOT increment this, so slot/mask stays the same on retry.
        self.successful_round_index = 0
        self.lock = threading.Lock()
        self.round_uploads: Dict[int, Dict[int, UploadCompleteBody]] = {}
        self.round_results: Dict[int, Dict[str, Any]] = {}
        self.round_plans: Dict[int, RoundPlan] = {}
        self.round_log: List[Dict[str, Any]] = []
        self.round_meta: Dict[int, Dict[str, Any]] = {}
        self._aggregating = False
        self._secagg_finalizing: set = set()  # round_idx 正在后台 finalize，避免重复聚合
        # 保护 in-memory global_state：apply delta 与异步 put_torch 互斥
        self._global_io_lock = threading.Lock()
        # 流式 per-block pipeline：round_idx -> {block_idx -> set(client_ids)} 已上传的 block
        self.round_block_uploads: Dict[int, Dict[int, set]] = {}
        # round_idx -> {block_idx -> agg_block_delta} 已聚合的 block 结果
        self.round_block_agg: Dict[int, Dict[int, Any]] = {}
        # round_idx -> {client_id -> num_examples} 流式上传的 metadata
        self.round_client_meta: Dict[int, Dict[int, Dict[str, Any]]] = {}
        # 本轮打开后，若超过该时间仍未收齐客户端上传，则标记失败（避免对端无限等待）
        self.client_upload_timeout_s = float(client_upload_timeout_s)

        # RES-1：恢复模式
        if self.resume_from_round > 1:
            self._resume_state()
        else:
            # 写入 round-0 初始全局状态 + 第 1 轮 plan
            if not self.minio.exists(global_state_key(0)):
                logger.info("Uploading initial global_state/round-0/state.pt")
                self.minio.put_torch(global_state_key(0), self.global_state)
            self.current_round = 1
            self._ensure_plan(1)
            self._mark_round_open(1)
        self._write_run_meta()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def _resume_state(self) -> None:
        """从 resume_from_round 恢复 global_state / round_log / current_round。"""
        target = int(self.resume_from_round)
        if target < 1:
            target = 1
        # 找到最近一次全量 checkpoint（target 本身或更早）
        restored_round = -1
        for r in range(target, 0, -1):
            if r == 0 or self.minio.exists(global_state_key(r)):
                if r > 0:
                    self.global_state = self.minio.get_torch(global_state_key(r), map_location="cpu")
                    if isinstance(self.global_state, dict) and "block_delta" in self.global_state:
                        self.global_state = self.global_state["block_delta"]
                    logger.info("RES-1: restored global_state from round-%s full checkpoint", r)
                restored_round = r
                break
        if restored_round < 0:
            raise RuntimeError(
                f"RES-1: cannot resume from round {target}; no full checkpoint found in MinIO"
            )
        # 顺序 apply delta 直到 target-1（即恢复到 resume_from_round - 1 的全局状态）
        for r in range(restored_round + 1, target):
            dkey = global_delta_key(r)
            if not self.minio.exists(dkey):
                raise RuntimeError(f"RES-1: missing delta {dkey} while resuming to round {target}")
            payload = self.minio.get_torch(dkey, map_location="cpu")
            if isinstance(payload, dict) and "block_delta" in payload:
                from shared.block_selection import add_block_delta

                add_block_delta(self.global_state, payload["block_delta"])
                logger.info("RES-1: applied delta round-%s", r)
        self.current_round = target
        # 恢复 round_log（本地文件优先，否则从 metrics.jsonl 重建）
        rl_path = self.results_dir / "round_log.json"
        if rl_path.exists():
            try:
                self.round_log = json.loads(rl_path.read_text(encoding="utf-8"))
                self.round_log = [e for e in self.round_log if int(e.get("round", -1)) < target]
            except Exception:
                self.round_log = []
        self._ensure_plan(target)
        self._mark_round_open(target)
        logger.info("RES-1: resume complete, current_round=%s", self.current_round)

    def _write_run_meta(self) -> None:
        """写出 run_meta.json，记录实际生效的超参（CFG-1/CFG-2 验收）。"""
        meta = {
            "num_clients": self.num_clients,
            "num_rounds": self.num_rounds,
            "coverage_h": self.coverage_h,
            "slots_per_round": self.slots_per_round,
            "seed": self.seed,
            "transfer_dtype": str(self.transfer_dtype).replace("torch.", ""),
            "block_size": self.block_size,
            "memory_decay": self.memory_decay,
            "compressor": self.compressor,
            "rho": self.rho,
            "write_full_global_every_n_rounds": self.write_full_global_every_n_rounds,
            "min_clients_to_aggregate": self.effective_min_clients,
            "client_upload_timeout_s": self.client_upload_timeout_s,
            "resume_from_round": self.resume_from_round,
            "effective_config": self.run_config_snapshot,
        }
        (self.results_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _mark_round_open(self, round_idx: int) -> None:
        meta = self.round_meta.setdefault(round_idx, {})
        if "opened_mono" not in meta:
            meta["opened_mono"] = time.monotonic()
            meta["opened_at"] = _utc_now_iso()
            meta["client_upload_mono"] = {}
            meta["client_timings"] = {}

    def _find_log_entry(self, round_idx: int) -> Optional[Dict[str, Any]]:
        for entry in self.round_log:
            if int(entry.get("round", -1)) == round_idx:
                return entry
        return None

    def _ensure_plan(self, round_idx: int) -> RoundPlan:
        if round_idx in self.round_plans:
            return self.round_plans[round_idx]
        plan = self.scheduler.plan_for_round(round_idx)
        self.round_plans[round_idx] = plan
        self.minio.put_json(plan_key(round_idx), plan.to_dict())
        logger.info(
            "Prepared plan round=%s epoch=%s slot=%s blocks=%s ratio=%.2f%%",
            plan.round,
            plan.epoch,
            plan.slot,
            plan.n_selected_blocks,
            plan.upload_ratio * 100,
        )
        return plan

    def _round_key_for(self, round_idx: int) -> bytes:
        """SEC-2：从 plan 的 (seed, epoch) 派生 per-round 密钥。"""
        plan = self._ensure_plan(round_idx)
        from shared.block_crypto import derive_round_key

        epoch_seed = epoch_seed_from_plan(plan.seed, plan.epoch)
        return derive_round_key(epoch_seed, round_idx)

    def _decode_sec_to_block_delta(
        self, payload: Any, round_idx: int
    ) -> Any:
        """SEC-1/2/3：把 SEC 格式 payload 还原成 block_delta dict。

        输入可能是：
        - SEC pipeline 格式 (v=1, enc_gidx)：单个 block
        - SEC batch 格式 (v=1, sec_blocks)：所有 block 的 list
        - 旧格式：{block_delta: {key_name: [(s,e,slice),...]}, ...}

        返回统一格式：block_delta dict = {key_name: [(start, end, slice), ...]}
        （兼容 apply_block_delta 的输入）
        """
        # 旧格式：直接取 block_delta
        if isinstance(payload, dict) and "block_delta" in payload and not is_sec_payload(payload):
            return payload["block_delta"]
        # SEC batch 格式：sec_blocks 列表
        if isinstance(payload, dict) and payload.get("v") == 1 and "sec_blocks" in payload:
            round_key = self._round_key_for(round_idx)
            block_delta: Dict[str, list] = {}
            for sec_item in payload["sec_blocks"]:
                gidx, slice_data, real_len, is_int8, scale = decode_sec_block_payload(sec_item, round_key)
                kn, s, e = self.scheduler.gidx_map.get(int(gidx), ("", 0, 0))
                if not kn:
                    continue
                if is_int8:
                    item = (s, e, slice_data, scale) if scale is not None else (s, e, slice_data)
                else:
                    item = (s, e, slice_data)
                block_delta.setdefault(kn, []).append(item)
            return block_delta
        # 单个 SEC payload（pipeline per-block 上传）
        if is_sec_payload(payload):
            round_key = self._round_key_for(round_idx)
            gidx, slice_data, real_len, is_int8, scale = decode_sec_block_payload(payload, round_key)
            kn, s, e = self.scheduler.gidx_map.get(int(gidx), ("", 0, 0))
            if not kn:
                logger.warning("SEC: gidx %s not in gidx_map", gidx)
                return {}
            # 还原 int8 元组格式（与旧格式一致）
            if is_int8:
                item = (s, e, slice_data, scale) if scale is not None else (s, e, slice_data)
            else:
                item = (s, e, slice_data)
            return {kn: [item]}
        # 其他情况：原样返回（可能是 batch 模式的旧 payload）
        return payload

    def status(self) -> Dict[str, Any]:
        with self.lock:
            finished = self.current_round > self.num_rounds
            return {
                "round": min(self.current_round, self.num_rounds),
                "num_rounds": self.num_rounds,
                "num_clients": self.num_clients,
                "status": "finished" if finished else "running",
                "aggregating": self._aggregating,
                "floating_elems": floating_elem_count(self.global_state),
                "successful_round_index": self.successful_round_index,
            }

    def get_plan(self, round_idx: int) -> Dict[str, Any]:
        if round_idx < 1 or round_idx > self.num_rounds:
            raise HTTPException(404, f"round {round_idx} out of range")
        with self.lock:
            if round_idx > self.current_round:
                raise HTTPException(409, f"round {round_idx} not open yet (current={self.current_round})")
            plan = self._ensure_plan(round_idx).to_dict()
            # OPS-3：Client 选择。None=全部参与；list=只这些 client 训练，其余 skip
            plan["selected_client_ids"] = (
                list(self.selected_client_ids) if self.selected_client_ids is not None else None
            )
            return plan

    def upload_complete(self, round_idx: int, client_id: int, body: UploadCompleteBody) -> Dict[str, Any]:
        if client_id < 0 or client_id >= self.num_clients:
            raise HTTPException(400, f"client_id must be in [0, {self.num_clients})")
        if round_idx < 1 or round_idx > self.num_rounds:
            raise HTTPException(404, f"round {round_idx} out of range")

        key = upload_blocks_key(round_idx, client_id)
        if not self.minio.exists(key):
            raise HTTPException(400, f"missing MinIO object: {key}")

        # SYNC-2：迟到上传（本轮已聚合）直接拒绝，提示下一轮
        with self.lock:
            if round_idx < self.current_round:
                raise HTTPException(
                    409,
                    f"round {round_idx} already finished (current={self.current_round}); upload next round",
                )
            if round_idx != self.current_round:
                raise HTTPException(409, f"expected round {self.current_round}, got {round_idx}")

        should_aggregate = False
        with self.lock:
            self._mark_round_open(round_idx)
            bucket = self.round_uploads.setdefault(round_idx, {})
            bucket[client_id] = body
            meta = self.round_meta[round_idx]
            meta["client_upload_mono"][client_id] = time.monotonic()
            if body.timings:
                meta["client_timings"][client_id] = dict(body.timings)
            logger.info(
                "Upload complete round=%s client=%s (%s/%s, min=%s) loss=%.4f n=%s timings=%s",
                round_idx,
                client_id,
                len(bucket),
                self.num_clients,
                self.effective_min_clients,
                body.train_loss,
                body.num_examples,
                {k: _round_s(v) for k, v in (body.timings or {}).items()},
            )
            # SYNC-1：满足最低人数即可聚合；min=0 表示旧行为（等齐 num_clients）
            ready = len(bucket) >= self.effective_min_clients
            if (
                ready
                and round_idx not in self.round_results
                and not self._aggregating
            ):
                self._aggregating = True
                should_aggregate = True
                meta["partial"] = len(bucket) < self.num_clients
                if meta["partial"]:
                    meta["participated_clients"] = sorted(bucket.keys())
                    meta["missing_clients"] = [
                        cid for cid in range(self.num_clients) if cid not in bucket
                    ]

        if should_aggregate:
            threading.Thread(target=self._aggregate_round, args=(round_idx,), daemon=True).start()
        return {"accepted": True, "received": len(self.round_uploads.get(round_idx, {}))}

    def report_client_timing(self, round_idx: int, client_id: int, body: ClientTimingBody) -> Dict[str, Any]:
        if client_id < 0 or client_id >= self.num_clients:
            raise HTTPException(400, f"client_id must be in [0, {self.num_clients})")
        if round_idx < 1 or round_idx > self.num_rounds:
            raise HTTPException(404, f"round {round_idx} out of range")
        with self.lock:
            meta = self.round_meta.setdefault(round_idx, {"client_timings": {}})
            prev = dict(meta.get("client_timings", {}).get(client_id, {}))
            prev.update({k: float(v) for k, v in body.timings.items()})
            meta.setdefault("client_timings", {})[client_id] = prev
            entry = self._find_log_entry(round_idx)
            if entry is not None:
                timing = entry.setdefault("timing_s", {})
                clients = timing.setdefault("clients", {})
                clients[str(client_id)] = {k: _round_s(v) for k, v in prev.items()}
                self._write_round_log()
            logger.info(
                "Client timing round=%s client=%s timings=%s",
                round_idx,
                client_id,
                {k: _round_s(v) for k, v in prev.items()},
            )
        return {"accepted": True}

    def block_uploaded(self, round_idx: int, client_id: int, block_idx: int,
                        num_examples: int, train_loss: float,
                        eval_loss: Optional[float] = None,
                        block_energies: Optional[List[float]] = None) -> Dict[str, Any]:
        """流式 per-block pipeline：client 上传一个 block 后通知 server。

        server 检查该 block 的所有 client 是否都已上传：
        - 是 → 立即 FedAvg 该 block → 写 agg_block_key → client 可下载
        - 否 → 记录，继续等其他 client 的该 block
        """
        if client_id < 0 or client_id >= self.num_clients:
            raise HTTPException(400, f"client_id must be in [0, {self.num_clients})")
        if round_idx < 1 or round_idx > self.num_rounds:
            raise HTTPException(404, f"round {round_idx} out of range")

        key = upload_block_key(round_idx, client_id, block_idx)
        if not self.minio.exists(key):
            raise HTTPException(400, f"missing MinIO object: {key}")

        # 记录该 client 的 metadata
        with self.lock:
            if round_idx < self.current_round:
                raise HTTPException(409, f"round {round_idx} already finished")
            block_uploads = self.round_block_uploads.setdefault(round_idx, {})
            uploaded_clients = block_uploads.setdefault(block_idx, set())
            uploaded_clients.add(client_id)
            # 记录 client metadata
            client_meta = self.round_client_meta.setdefault(round_idx, {})
            client_meta[client_id] = {
                "num_examples": num_examples,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }

            # 检查该 block 是否所有 client 都已上传
            all_uploaded = len(uploaded_clients) >= self.effective_min_clients
            if not all_uploaded:
                logger.info(
                    "Pipeline block %s round %s: client %s uploaded (%s/%s), waiting",
                    block_idx, round_idx, client_id,
                    len(uploaded_clients), self.effective_min_clients,
                )
                return {"aggregated": False, "waiting": len(uploaded_clients)}

        # 所有 client 的该 block 都已上传 → 立即聚合
        agg_result = self._aggregate_single_block(round_idx, block_idx)

        return {"aggregated": True, "agg_key": agg_block_key(round_idx, block_idx)}

    # ----------------------------------------------------------------------- #
    # B-7a: SecAgg pipeline 集成
    # ----------------------------------------------------------------------- #

    def secagg_block_uploaded(
        self,
        round_idx: int,
        client_id: int,
        block_idx: int,
        z_hex: str = "",
        vector_len: int = 0,
        num_examples: int = 1,
        train_loss: float = 0.0,
        eval_loss: Optional[float] = None,
        block_energies: Optional[List[float]] = None,
        z_key: str = "",
    ) -> Dict[str, Any]:
        """B-7a: SecAgg pipeline — client 通知一个 masked window 已上传。

        z_key：只登记对象键，finalize 再 GET（PERF-1）。
        z_hex：兼容旧客户端，仍在 handler 里解码。
        """
        if client_id < 0 or client_id >= self.num_clients:
            raise HTTPException(400, f"client_id must be in [0, {self.num_clients})")
        if round_idx < 1 or round_idx > self.num_rounds:
            raise HTTPException(404, f"round {round_idx} out of range")

        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            raise HTTPException(400, "SecAgg not enabled")

        if z_key:
            coord.register_masked_window_key(client_id, block_idx, z_key, vector_len)
        elif z_hex:
            from shared.fixed_point import unpack_zq
            z_data = bytes.fromhex(z_hex)
            z_kw = unpack_zq(z_data, vector_len, self.secagg_modulus_bits)
            coord.submit_masked_window(client_id, block_idx, z_kw)
        else:
            raise HTTPException(400, "missing z_key or z_hex")

        # 记录 client metadata
        with self.lock:
            if round_idx < self.current_round:
                raise HTTPException(409, f"round {round_idx} already finished")
            block_uploads = self.round_block_uploads.setdefault(round_idx, {})
            uploaded_clients = block_uploads.setdefault(block_idx, set())
            uploaded_clients.add(client_id)
            client_meta = self.round_client_meta.setdefault(round_idx, {})
            client_meta[client_id] = {
                "num_examples": num_examples,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }
            all_uploaded = len(uploaded_clients) >= self.effective_min_clients

        if not all_uploaded:
            logger.info(
                "SecAgg block %s round %s: client %s uploaded z_k (%s/%s), waiting",
                block_idx, round_idx, client_id,
                len(uploaded_clients), self.effective_min_clients,
            )
            return {"aggregated": False, "waiting": len(uploaded_clients)}

        # 所有 client 的该 block 都已上传 → 检查是否所有 self_master 也已提交
        n_windows = self._ensure_plan(round_idx).n_selected_blocks
        all_windows_uploaded = len(block_uploads) >= n_windows
        logger.info(
            "SecAgg: all clients uploaded z_k for block %s round %s, checking self_masters",
            block_idx, round_idx,
        )
        return {
            "aggregated": False,
            "waiting_self_masters": True,
            "all_windows_uploaded": all_windows_uploaded,
        }

    def secagg_blob_uploaded(
        self,
        round_idx: int,
        client_id: int,
        z_key: str,
        window_ids: List[int],
        num_examples: int = 1,
        train_loss: float = 0.0,
        eval_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """整轮 masked windows 一个 blob：只登记 z_key，不在 HTTP 里 GET。"""
        if client_id < 0 or client_id >= self.num_clients:
            raise HTTPException(400, f"client_id must be in [0, {self.num_clients})")
        if round_idx < 1 or round_idx > self.num_rounds:
            raise HTTPException(404, f"round {round_idx} out of range")
        if not z_key:
            raise HTTPException(400, "missing z_key")

        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            raise HTTPException(400, "SecAgg not enabled")

        ids = [int(w) for w in window_ids]
        coord.register_masked_blob(client_id, z_key, ids)

        plan = self._ensure_plan(round_idx)
        n_windows = plan.n_selected_blocks
        with self.lock:
            if round_idx < self.current_round:
                raise HTTPException(409, f"round {round_idx} already finished")
            block_uploads = self.round_block_uploads.setdefault(round_idx, {})
            for wid in ids:
                block_uploads.setdefault(wid, set()).add(client_id)
            client_meta = self.round_client_meta.setdefault(round_idx, {})
            client_meta[client_id] = {
                "num_examples": num_examples,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }
            all_windows_uploaded = len(block_uploads) >= n_windows

        logger.info(
            "SecAgg blob round %s client %s n_windows=%s all_registered=%s",
            round_idx, client_id, len(ids), all_windows_uploaded,
        )
        return {
            "aggregated": False,
            "all_windows_uploaded": all_windows_uploaded,
        }

    def secagg_schedule_finalize(self, round_idx: int) -> None:
        """在后台 finalize，避免卡住 self-master / masked-window 的 HTTP 响应。"""
        def _run() -> None:
            try:
                self.secagg_try_finalize_round(round_idx)
            except Exception:
                logger.exception("SecAgg: background finalize failed round=%s", round_idx)

        threading.Thread(
            target=_run, name=f"secagg-finalize-{round_idx}", daemon=True,
        ).start()

    def secagg_try_finalize_round(self, round_idx: int) -> bool:
        """B-7a: 尝试完成 SecAgg 轮次。

        条件：所有 window 的 z_k 都已上传 + 所有 self_master 都已提交。
        如果条件满足：聚合 unmask → 写每个 block 的 agg_block_key → 写 delta → 推进 round。
        返回 True 如果完成，False 如果还在等待。
        """
        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            return False

        plan = self._ensure_plan(round_idx)
        n_windows = plan.n_selected_blocks

        with self.lock:
            if round_idx in self.round_results:
                return True
            if round_idx in self._secagg_finalizing:
                return False
            block_uploads = self.round_block_uploads.get(round_idx, {})
            n_blocks_uploaded = len(block_uploads)
            n_self_masters = len(coord.self_masters)
            all_windows_done = n_blocks_uploaded >= n_windows
            all_self_masters = n_self_masters >= self.effective_min_clients
            if not (all_windows_done and all_self_masters):
                return False
            self._secagg_finalizing.add(round_idx)

        try:
            return self._secagg_run_finalize(round_idx, coord, plan, n_windows)
        except Exception:
            with self.lock:
                self._secagg_finalizing.discard(round_idx)
                self._aggregating = False
            raise

    def _async_put_full_global(self, round_idx: int) -> None:
        """全量 global_state 不挡 client 等聚合；与下一轮 apply 用同一把锁。"""
        t0 = time.monotonic()
        try:
            with self._global_io_lock:
                nbytes = self.minio.put_torch(
                    global_state_key(round_idx), self.global_state,
                )
            logger.info(
                "SecAgg: async full global_state round=%s MiB=%.1f in %.1fs",
                round_idx, nbytes / (1024 * 1024), time.monotonic() - t0,
            )
        except Exception:
            logger.exception(
                "SecAgg: async full global_state failed round=%s", round_idx,
            )

    def _secagg_run_finalize(
        self, round_idx: int, coord: Any, plan: Any, n_windows: int,
    ) -> bool:
        """已抢到 finalize 名额后的实际聚合与落盘。"""
        logger.info(
            "SecAgg: finalizing round %s windows=%s (all z_k + self_masters received)",
            round_idx, n_windows,
        )
        t0 = time.monotonic()
        try:
            coord.materialize_masked_windows(self.minio.get_bytes)
            t_mat = time.monotonic()
            coord.freeze_survivors()
            agg_delta = coord.aggregate_and_unmask()
            t_unmask = time.monotonic()
        except RuntimeError as e:
            logger.error("SecAgg aggregation failed: %s", e)
            with self.lock:
                self.round_results[round_idx] = {
                    "round": round_idx,
                    "done": False,
                    "message": "secagg_failed",
                    "reason": str(e),
                }
                self._aggregating = False
                self._secagg_finalizing.discard(round_idx)
            return True

        # 只写一份 global_delta（client 一次 GET）；不再按 window 落 N 个 agg_block。
        with self.lock:
            client_meta = dict(self.round_client_meta.get(round_idx, {}))
            participated = sorted(client_meta.keys())
        losses = [float(client_meta[c].get("train_loss", 0)) for c in participated]
        eval_losses = [float(client_meta[c]["eval_loss"]) for c in participated
                       if client_meta[c].get("eval_loss") is not None]

        # 把 agg_delta 转成 block_delta 格式（与现有 pipeline 一致）
        agg_delta_fp16: Dict[str, list] = {}
        for kn, blocks in agg_delta.items():
            agg_delta_fp16[kn] = [
                (s, e, sd.to(dtype=self.transfer_dtype)) for s, e, sd in blocks
            ]

        # 写 delta payload（内存序列化，避免临界路径 tempfile）
        delta_payload = {
            "round": int(round_idx),
            "from_round": int(round_idx - 1),
            "to_round": int(round_idx),
            "block_delta": agg_delta_fp16,
            "n_selected_blocks": int(plan.n_selected_blocks),
            "selected_elems": int(plan.selected_elems),
            "secagg": True,
        }
        delta_bytes = self.minio.put_torch(
            global_delta_key(round_idx), delta_payload, via_memory=True,
        )
        t_delta = time.monotonic()

        # 内存里先 apply；全量写盘放到后台，client 只等 global_delta + done
        wrote_full_global = self._should_write_full_global(round_idx)
        if wrote_full_global and self.global_state is not None:
            from shared.block_selection import add_block_delta
            with self._global_io_lock:
                add_block_delta(self.global_state, agg_delta_fp16)

        # 构建结果
        avg_train = sum(losses) / max(len(losses), 1)
        avg_eval = (sum(eval_losses) / len(eval_losses)) if eval_losses else None
        result = {
            "round": round_idx,
            "done": True,
            "eval_loss": round(float(avg_eval), 6) if avg_eval is not None else None,
            "avg_train_loss": round(avg_train, 6),
            "upload_ratio": round(plan.upload_ratio, 6),
            "n_selected_blocks": plan.n_selected_blocks,
            "selected_elems": plan.selected_elems,
            "message": "secagg_aggregated",
            "secagg": True,
            "wrote_full_global": wrote_full_global,
            "global_delta_key": global_delta_key(round_idx),
            "global_state_key": global_state_key(round_idx) if wrote_full_global else None,
        }
        entry = {
            "round": round_idx,
            "epoch": plan.epoch,
            "slot": plan.slot,
            "avg_train_loss": result["avg_train_loss"],
            "eval_loss": result["eval_loss"],
            "pct_of_total": round(plan.upload_ratio * 100, 2),
            "n_selected_blocks": plan.n_selected_blocks,
            "secagg": True,
        }
        with self.lock:
            self.round_results[round_idx] = result
            self.round_log.append(entry)
            self._write_round_log()
            self._append_metrics_jsonl(entry)
            self.successful_round_index += 1
            if round_idx < self.num_rounds:
                self.current_round = round_idx + 1
                self._ensure_plan(self.current_round)
                self._mark_round_open(self.current_round)
            else:
                self.current_round = self.num_rounds + 1
            self._aggregating = False
            self._secagg_finalizing.discard(round_idx)

        if wrote_full_global and self.global_state is not None:
            threading.Thread(
                target=self._async_put_full_global,
                args=(round_idx,),
                name=f"secagg-put-global-{round_idx}",
                daemon=True,
            ).start()

        logger.info(
            "SecAgg: round %s complete, avg_train=%.4f delta_MiB=%.1f "
            "materialize=%.1fs unmask=%.1fs put_delta=%.1fs wait_s=%.1fs async_full=%s",
            round_idx, avg_train, delta_bytes / (1024 * 1024),
            t_mat - t0, t_unmask - t_mat, t_delta - t_unmask, t_delta - t0,
            wrote_full_global,
        )
        return True

    def _aggregate_single_block(self, round_idx: int, block_idx: int) -> Dict[str, Any]:
        """流式：聚合单个 block（所有 client 的该 block FedAvg），写出 agg_block_key。"""
        plan = self._ensure_plan(round_idx)
        # 从 block_list 找到该 block 的 key_name, start, end
        block_info = None
        for b in plan.block_list:
            if int(b[0]) == block_idx:
                block_info = b
                break
        if block_info is None:
            logger.error("Pipeline: block_idx %s not in plan block_list", block_idx)
            return {}

        _, key_name, start, end = block_info
        selected_for_block = {key_name: [(start, end)]}

        # 下载所有 client 的该 block
        with self.lock:
            uploaded = sorted(self.round_block_uploads.get(round_idx, {}).get(block_idx, set()))
            client_meta = dict(self.round_client_meta.get(round_idx, {}))

        deltas = []
        weights = []
        for cid in uploaded:
            bkey = upload_block_key(round_idx, cid, block_idx)
            payload = self.minio.get_torch(bkey, map_location="cpu")
            # SEC-1/2/3：兼容 SEC 格式（v=1）与旧格式
            if is_sec_payload(payload):
                bd = self._decode_sec_to_block_delta(payload, round_idx)
                deltas.append(bd)
                weights.append(float(payload.get("num_examples", client_meta.get(cid, {}).get("num_examples", 1))))
            elif isinstance(payload, dict) and "block_delta" in payload:
                deltas.append(payload["block_delta"])
                weights.append(float(payload.get("num_examples", client_meta.get(cid, {}).get("num_examples", 1))))
            else:
                deltas.append(payload)
                weights.append(float(client_meta.get(cid, {}).get("num_examples", 1)))

        # FedAvg 该单个 block
        agg_delta = apply_block_delta(
            self.global_state,
            deltas,
            weights,
            selected_for_block,
            out_dtype=self.transfer_dtype,
        )

        # 写出该 block 的聚合 delta
        agg_payload = {
            "round": int(round_idx),
            "block_idx": int(block_idx),
            "block_delta": agg_delta,
        }
        self.minio.put_torch(agg_block_key(round_idx, block_idx), agg_payload)

        with self.lock:
            self.round_block_agg.setdefault(round_idx, {})[block_idx] = agg_delta
            n_done = len(self.round_block_agg.get(round_idx, {}))
            plan = self.round_plans.get(round_idx)
        logger.info(
            "Pipeline: aggregated block %s/%s round %s (key=%s [%s:%s])",
            n_done, plan.n_selected_blocks if plan else "?", round_idx, key_name, start, end,
        )

        # 所有 block 聚合完成 → 触发 round 完成（组装 delta + 推进 current_round）
        if plan and n_done >= plan.n_selected_blocks:
            logger.info("Pipeline: all blocks aggregated for round %s, finalizing", round_idx)
            self._finalize_pipeline_round(round_idx)
        return agg_payload

    def _finalize_pipeline_round(self, round_idx: int) -> None:
        """流式：所有 block 聚合完成后，组装完整 delta、写 MinIO、推进 current_round。"""
        threading.Thread(target=self._aggregate_round, args=(round_idx,), daemon=True).start()

    def check_block_agg_done(self, round_idx: int) -> Dict[str, Any]:
        """client 轮询本轮聚合是否完成。

        非 SecAgg 流式路径看 per-block `round_block_agg`；
        SecAgg 只写一份 global_delta，以 round_results.done 为准。
        """
        with self.lock:
            plan = self.round_plans.get(round_idx)
            n_total = int(plan.n_selected_blocks) if plan is not None else 0
            result = self.round_results.get(round_idx) or {}
            if result.get("done"):
                return {"done": True, "n_done": n_total, "n_total": n_total}
            n_done = len(self.round_block_agg.get(round_idx, {}))
            return {
                "done": bool(n_total and n_done >= n_total),
                "n_done": n_done,
                "n_total": n_total,
            }

    def get_result(self, round_idx: int) -> Dict[str, Any]:
        with self.lock:
            if round_idx in self.round_results:
                return self.round_results[round_idx]
            if round_idx > self.current_round:
                raise HTTPException(404, f"round {round_idx} not started")
            return {
                "round": round_idx,
                "done": False,
                "message": "waiting for clients / aggregation",
            }

    def _watchdog_loop(self) -> None:
        while True:
            try:
                self._check_upload_timeouts()
            except Exception:
                logger.exception("upload timeout watchdog error")
            time.sleep(5.0)

    def _check_upload_timeouts(self) -> None:
        with self.lock:
            round_idx = self.current_round
            if round_idx < 1 or round_idx > self.num_rounds:
                return
            if round_idx in self.round_results or self._aggregating:
                return
            meta = self.round_meta.get(round_idx) or {}
            opened = meta.get("opened_mono")
            if opened is None:
                return
            elapsed = time.monotonic() - float(opened)
            # 宽限期：优先用 round_deadline_s，否则 client_upload_timeout_s
            deadline = self.round_deadline_s if self.round_deadline_s > 0 else self.client_upload_timeout_s
            if elapsed < deadline:
                return
            bucket = self.round_uploads.get(round_idx, {})
            missing = [cid for cid in range(self.num_clients) if cid not in bucket]
            # SYNC-1：超时后若满足最低人数，用已上传子集做部分聚合
            if len(bucket) >= self.effective_min_clients and len(bucket) < self.num_clients:
                logger.warning(
                    "Round %s partial aggregation after %.0fs; received=%s missing=%s",
                    round_idx,
                    elapsed,
                    sorted(bucket.keys()),
                    missing,
                )
                meta["partial"] = True
                meta["participated_clients"] = sorted(bucket.keys())
                meta["missing_clients"] = missing
                self._aggregating = True
                threading.Thread(
                    target=self._aggregate_round, args=(round_idx,), daemon=True
                ).start()
                return
            if len(bucket) >= self.num_clients:
                return
            logger.error(
                "Round %s upload timeout after %.0fs; received=%s missing=%s (below min=%s)",
                round_idx,
                elapsed,
                sorted(bucket.keys()),
                missing,
                self.effective_min_clients,
            )
            self.round_results[round_idx] = {
                "round": round_idx,
                "done": False,
                "message": "aggregation_failed",
                "reason": "client_upload_timeout",
                "missing_clients": missing,
                "elapsed_s": _round_s(elapsed),
            }
            self._aggregating = False

    def _should_write_full_global(self, round_idx: int) -> bool:
        """IO-1：决定本轮是否写全量 global_state。"""
        n = self.write_full_global_every_n_rounds
        if n == 1:
            return True
        if n <= 0:
            # 只写 round-0 与最终一轮
            return round_idx == self.num_rounds
        return round_idx % n == 0 or round_idx == self.num_rounds

    def _aggregate_round(self, round_idx: int) -> None:
        try:
            t_agg0 = time.monotonic()
            plan = self._ensure_plan(round_idx)
            selected = selected_from_jsonable(plan.selected_by_key)

            # 流式 per-block pipeline：如果所有 block 已通过 pipeline 聚合完成，
            # 直接组装 agg_delta，不再重新下载/聚合
            with self.lock:
                block_agg = dict(self.round_block_agg.get(round_idx, {}))
            if block_agg and len(block_agg) >= plan.n_selected_blocks:
                logger.info("Pipeline: all %s blocks already aggregated via pipeline, assembling", plan.n_selected_blocks)
                # 组装完整 agg_delta（合并所有 block）
                agg_delta = {}
                for bidx in sorted(block_agg.keys()):
                    bd = block_agg[bidx]
                    for kn, blocks in bd.items():
                        agg_delta.setdefault(kn, []).extend(blocks)

                # 从 client_meta 取 losses
                with self.lock:
                    client_meta = dict(self.round_client_meta.get(round_idx, {}))
                    participated = sorted(client_meta.keys())
                losses = [float(client_meta[c].get("train_loss", 0)) for c in participated]
                eval_losses = [float(client_meta[c]["eval_loss"]) for c in participated
                               if client_meta[c].get("eval_loss") is not None]
                weights = [float(client_meta[c].get("num_examples", 1)) for c in participated]
                metas = {}  # pipeline 模式不用 metas
                upload_sizes_b = [0] * len(participated)
                t_download = 0.0
                t_apply = 0.0
            else:
                # 批量模式（回退）
                metas = self.round_uploads[round_idx]
                weights: List[float] = []
                deltas = []
                losses = []
                eval_losses: List[float] = []

                # 流式聚合：每收到一个 client 上传就立即下载（不等其他 client），
                # 下载与下一个 client 的训练/上传并行，减少串行等待
                participated = sorted(metas.keys())
                t_dl0 = time.monotonic()
                upload_sizes_b: List[int] = []
                for cid in participated:
                    key = upload_blocks_key(round_idx, cid)
                    try:
                        upload_sizes_b.append(self.minio.object_size(key))
                    except Exception:
                        upload_sizes_b.append(0)
                    t_block0 = time.monotonic()
                    payload = self.minio.get_torch(key, map_location="cpu")
                    t_block = time.monotonic() - t_block0
                    # SEC-1/2/3：兼容 SEC 格式（v=1, pipeline 单 block 或 batch sec_blocks）与旧格式
                    if isinstance(payload, dict) and payload.get("v") == 1:
                        deltas.append(self._decode_sec_to_block_delta(payload, round_idx))
                        weights.append(float(payload.get("num_examples", metas[cid].num_examples)))
                        losses.append(float(payload.get("train_loss", metas[cid].train_loss)))
                        ev = payload.get("eval_loss", metas[cid].eval_loss)
                        if ev is not None:
                            eval_losses.append(float(ev))
                    elif isinstance(payload, dict) and "block_delta" in payload:
                        deltas.append(payload["block_delta"])
                        weights.append(float(payload.get("num_examples", metas[cid].num_examples)))
                        losses.append(float(payload.get("train_loss", metas[cid].train_loss)))
                        ev = payload.get("eval_loss", metas[cid].eval_loss)
                        if ev is not None:
                            eval_losses.append(float(ev))
                    else:
                        deltas.append(payload)
                        weights.append(float(metas[cid].num_examples))
                        losses.append(float(metas[cid].train_loss))
                        if metas[cid].eval_loss is not None:
                            eval_losses.append(float(metas[cid].eval_loss))
                    logger.info(
                        "Streaming: downloaded client %s blocks.pt in %.2fs (accumulated %s/%s)",
                        cid, t_block, len(deltas), len(participated),
                    )
                t_download = time.monotonic() - t_dl0

                # SEC-4：异常更新防护——计算各 client delta 范数，超阈值剔除/裁剪
                sec4_rejected: List[int] = []
                if self.delta_norm_reject > 0 or self.delta_norm_clip > 0:
                    kept_deltas: List = []
                    kept_weights: List[float] = []
                    kept_losses: List[float] = []
                    kept_participated: List[int] = []
                    for i, cid in enumerate(participated):
                        norm = 0.0
                        for blocks in deltas[i].values():
                            for item in blocks:
                                norm += float(_slice_to_fp32(item).pow(2).sum().item())
                        norm = norm ** 0.5
                        if self.delta_norm_reject > 0 and norm > self.delta_norm_reject:
                            logger.warning(
                                "SEC-4: reject client %s round %s: delta norm %.4f > %.4f",
                                cid, round_idx, norm, self.delta_norm_reject,
                            )
                            sec4_rejected.append(cid)
                            continue
                        if self.delta_norm_clip > 0 and norm > self.delta_norm_clip:
                            scale = self.delta_norm_clip / (norm + 1e-12)
                            for kn in list(deltas[i].keys()):
                                new_blocks = []
                                for it in deltas[i][kn]:
                                    s, e = it[0], it[1]
                                    sf32 = _slice_to_fp32(it) * scale
                                    orig_dtype = it[2].dtype
                                    if len(it) > 3:
                                        new_blocks.append((s, e, sf32.to(orig_dtype), it[3]))
                                    else:
                                        new_blocks.append((s, e, sf32.to(orig_dtype)))
                                deltas[i][kn] = new_blocks
                            logger.info(
                                "SEC-4: clip client %s round %s: norm %.4f -> %.4f",
                                cid, round_idx, norm, self.delta_norm_clip,
                            )
                        kept_deltas.append(deltas[i])
                        kept_weights.append(weights[i])
                        kept_losses.append(losses[i] if i < len(losses) else 0.0)
                        kept_participated.append(cid)
                    if sec4_rejected:
                        deltas = kept_deltas
                        weights = kept_weights
                        losses = kept_losses
                        participated = kept_participated
                        logger.warning("SEC-4: rejected clients %s for round %s", sec4_rejected, round_idx)

                t_apply0 = time.monotonic()
                if self.compressor == "dense":
                    selected = {}
                    seen: Dict[str, set] = {}
                    for delta in deltas:
                        for kn, blocks in delta.items():
                            bucket = selected.setdefault(kn, [])
                            seen_key = seen.setdefault(kn, set())
                            for item in blocks:
                                pair = (int(item[0]), int(item[1]))
                                if pair not in seen_key:
                                    seen_key.add(pair)
                                    bucket.append(pair)
                    logger.info(
                        "compressor=%s using union of uploaded blocks keys=%s",
                        self.compressor,
                        len(selected),
                    )
                agg_delta = apply_block_delta(
                    self.global_state,
                    deltas,
                    weights,
                    selected,
                    out_dtype=self.transfer_dtype,
                )
                t_apply = time.monotonic() - t_apply0

            # 公共：写出 delta + 结果（pipeline 模式 agg_delta 已组装好，batch 模式刚算完）

            t_up0 = time.monotonic()
            delta_payload = {
                "round": int(round_idx),
                "from_round": int(round_idx - 1),
                "to_round": int(round_idx),
                "block_delta": agg_delta,
                "n_selected_blocks": int(plan.n_selected_blocks),
                "selected_elems": int(plan.selected_elems),
            }
            delta_bytes = self.minio.put_torch(global_delta_key(round_idx), delta_payload)
            # IO-1：按频率写全量 global_state
            wrote_full_global = self._should_write_full_global(round_idx)
            global_bytes = 0
            if wrote_full_global:
                global_bytes = self.minio.put_torch(global_state_key(round_idx), self.global_state)
            else:
                logger.info("IO-1: skip writing full global_state for round %s", round_idx)
            t_upload_global = time.monotonic() - t_up0
            t_agg_total = time.monotonic() - t_agg0

            with self.lock:
                meta = self.round_meta.setdefault(round_idx, {})
                opened_mono = float(meta.get("opened_mono", t_agg0))
                upload_monos = dict(meta.get("client_upload_mono", {}))
                client_timings = {
                    str(cid): {k: _round_s(v) for k, v in timings.items()}
                    for cid, timings in dict(meta.get("client_timings", {})).items()
                }

            first_upload = min(upload_monos.values()) if upload_monos else t_agg0
            last_upload = max(upload_monos.values()) if upload_monos else t_agg0
            finished_mono = time.monotonic()
            transfer = {
                "upload_blocks_MiB_per_client": {
                    str(participated[i]): _round_s(sz / (1024 * 1024)) for i, sz in enumerate(upload_sizes_b)
                },
                "upload_blocks_MiB_total": _round_s(sum(upload_sizes_b) / (1024 * 1024)),
                "global_delta_MiB": _round_s(delta_bytes / (1024 * 1024)),
                "global_state_MiB": _round_s(global_bytes / (1024 * 1024)),
                "selected_elems_M": _round_s(plan.selected_elems / 1e6),
            }
            timing_s = {
                "opened_at": meta.get("opened_at"),
                "finished_at": _utc_now_iso(),
                "round_wall_s": _round_s(finished_mono - opened_mono),
                "wait_all_clients_s": _round_s(last_upload - opened_mono),
                "client_upload_gap_s": _round_s(last_upload - first_upload),
                "agg_download_uploads_s": _round_s(t_download),
                "agg_apply_s": _round_s(t_apply),
                "agg_upload_global_s": _round_s(t_upload_global),
                "agg_total_s": _round_s(t_agg_total),
                "transfer": transfer,
                "clients": client_timings,
            }

            avg_train = sum(losses) / max(len(losses), 1)
            client_losses = {str(participated[i]): round(float(losses[i]), 6) for i in range(len(losses))}
            avg_eval = (sum(eval_losses) / len(eval_losses)) if eval_losses else None
            client_eval = {}
            for cid in participated:
                # pipeline 模式 metas 为空，从 client_meta 取；batch 模式从 metas 取
                if metas:
                    ev = metas[cid].eval_loss
                else:
                    ev = self.round_client_meta.get(round_idx, {}).get(cid, {}).get("eval_loss")
                if ev is not None:
                    client_eval[str(cid)] = round(float(ev), 6)
            is_partial = bool(meta.get("partial")) and len(participated) < self.num_clients
            participated_clients = sorted(meta.get("participated_clients", participated))
            missing_clients = sorted(
                meta.get("missing_clients", [c for c in range(self.num_clients) if c not in participated])
            )
            result = {
                "round": round_idx,
                "done": True,
                "eval_loss": round(float(avg_eval), 6) if avg_eval is not None else None,
                "client_eval_loss": client_eval,
                "avg_train_loss": round(avg_train, 6),
                "client_train_loss": client_losses,
                "upload_ratio": round(plan.upload_ratio, 6),
                "n_selected_blocks": plan.n_selected_blocks,
                "selected_elems": plan.selected_elems,
                "message": "aggregated",
                "partial": is_partial,
                "participated_clients": participated_clients,
                "missing_clients": missing_clients if is_partial else [],
                "wrote_full_global": wrote_full_global,
                "global_delta_key": global_delta_key(round_idx),
                "global_state_key": global_state_key(round_idx) if wrote_full_global else None,
                "timing_s": timing_s,
            }
            entry = {
                "round": round_idx,
                "epoch": plan.epoch,
                "slot": plan.slot,
                "avg_train_loss": result["avg_train_loss"],
                "client_train_loss": client_losses,
                "eval_loss": result["eval_loss"],
                "client_eval_loss": client_eval,
                "pct_of_total": round(plan.upload_ratio * 100, 2),
                "n_selected_blocks": plan.n_selected_blocks,
                "selected_elems_M": round(plan.selected_elems / 1e6, 3),
                "wrote_full_global": wrote_full_global,
                "partial": is_partial,
                "participated_clients": participated_clients,
                "missing_clients": missing_clients if is_partial else [],
                "timing_s": timing_s,
            }
            with self.lock:
                self.round_results[round_idx] = result
                self.round_log.append(entry)
                self._write_round_log()
                self._append_metrics_jsonl(entry)
                # A-6: only increment successful_round_index after successful commit
                self.successful_round_index += 1
                if round_idx < self.num_rounds:
                    self.current_round = round_idx + 1
                    self._ensure_plan(self.current_round)
                    self._mark_round_open(self.current_round)
                else:
                    self.current_round = self.num_rounds + 1
                self._aggregating = False
            # OPS-2：清理旧轮 uploads（锁外执行，避免阻塞聚合推进）
            self._cleanup_old_uploads(round_idx)
            logger.info(
                "Aggregated round %s avg_train=%.4f ratio=%.2f%% transfer=%s timing=%s",
                round_idx,
                avg_train,
                plan.upload_ratio * 100,
                transfer,
                {
                    k: timing_s[k]
                    for k in (
                        "round_wall_s",
                        "wait_all_clients_s",
                        "client_upload_gap_s",
                        "agg_download_uploads_s",
                        "agg_apply_s",
                        "agg_upload_global_s",
                        "agg_total_s",
                    )
                },
            )
        except Exception:
            logger.exception("Aggregation failed for round %s", round_idx)
            with self.lock:
                self.round_results[round_idx] = {
                    "round": round_idx,
                    "done": False,
                    "message": "aggregation_failed",
                }
                self._aggregating = False

    def _write_round_log(self) -> None:
        path = self.results_dir / "round_log.json"
        path.write_text(json.dumps(self.round_log, indent=2), encoding="utf-8")

    def _append_metrics_jsonl(self, entry: Dict[str, Any]) -> None:
        path = self.results_dir / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _cleanup_old_uploads(self, completed_round: int) -> None:
        """OPS-2：删除早于 (completed - retention) 轮的 uploads/ 对象。0=不清理。"""
        k = self.minio_retention_recent_uploads
        if k <= 0:
            return
        cutoff = completed_round - k
        if cutoff < 1:
            return
        try:
            prefix = f"uploads/round-{cutoff}/"
            objs = self.minio.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            removed = 0
            for obj in objs:
                self.minio.client.remove_object(self.bucket, obj.object_name)
                removed += 1
            if removed:
                logger.info("OPS-2: removed %s upload objects for round %s (retention=%s)", removed, cutoff, k)
        except Exception:
            logger.warning("OPS-2: cleanup for round %s failed", cutoff, exc_info=True)

    # ----------------------------------------------------------------------- #
    # B-6: SecAgg coordinator management
    # ----------------------------------------------------------------------- #

    def _get_or_create_secagg_coordinator(self, round_idx: int) -> Optional[SecAggCoordinator]:
        """获取或创建 per-round SecAgg coordinator。"""
        if not self.secagg_enabled:
            return None
        with self.lock:
            if round_idx in self.secagg_coordinators:
                return self.secagg_coordinators[round_idx]
            plan = self._ensure_plan(round_idx)
            fallback_scale = self.secagg_scale if self.secagg_scale > 0 else ROUND1_PUBLIC_SCALE
            windows = build_window_descriptors(plan.block_list)
            # 固定 scale 模式：每 window 都用固定值；per-window 模式：留空，
            # 由 coordinator 收齐 client amax 后 _finalize_window_scales 填充。
            window_scales: Dict[str, float] = {}
            if self.secagg_scale > 0:
                for w in windows:
                    window_scales[str(w.window_id)] = float(self.secagg_scale)
            secagg_plan = SecAggPlan(
                q_min=self.secagg_q_min,
                quantization_scale=fallback_scale,
                window_scales=window_scales,
                modulus_bits=self.secagg_modulus_bits,
                modulus_q=self.secagg_q,
                q_max=self.secagg_q_max,
                stochastic_rounding=self.secagg_stochastic_rounding,
                hadamard_enabled=self.secagg_hadamard,
                hadamard_seed=round_idx,
            )
            client_ids = list(range(self.num_clients))
            if self.selected_client_ids is not None:
                client_ids = list(self.selected_client_ids)
            coord = SecAggCoordinator(
                round_idx=round_idx,
                successful_round_index=self.successful_round_index,
                client_ids=client_ids,
                windows=windows,
                layout_hash=plan.layout_hash,
                mask_hash=plan.mask_hash,
                secagg_plan=secagg_plan,
            )
            self.secagg_coordinators[round_idx] = coord
            return coord

    def secagg_key_announce(
        self,
        round_idx: int,
        client_id: int,
        pk_hex: str,
        window_amax: Optional[Dict[str, float]] = None,
        global_amax: Optional[float] = None,
    ) -> Dict[str, Any]:
        """B-6: client 提交 X25519 公钥 + scale 统计。

        Hadamard：只收 global_amax；否则收 per-window amax。
        """
        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            raise HTTPException(400, "SecAgg not enabled")
        pk_raw = bytes.fromhex(pk_hex)
        return coord.submit_public_key(
            client_id, pk_raw, window_amax=window_amax, global_amax=global_amax,
        )

    def secagg_get_peer_keys(self, round_idx: int) -> Dict[str, Any]:
        """B-6: client 获取所有 peer 的公钥。"""
        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            raise HTTPException(400, "SecAgg not enabled")
        return coord.get_peer_keys()

    def secagg_submit_masked_window(
        self, round_idx: int, client_id: int, window_id: int,
        z_data: bytes, vector_len: int,
    ) -> Dict[str, Any]:
        """B-6: client 提交一个 masked window (z_k packed bytes)。"""
        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            raise HTTPException(400, "SecAgg not enabled")
        from shared.fixed_point import unpack_zq
        z_kw = unpack_zq(z_data, vector_len, self.secagg_modulus_bits)
        return coord.submit_masked_window(client_id, window_id, z_kw)

    def secagg_submit_self_master(self, round_idx: int, client_id: int, sm_hex: str) -> Dict[str, Any]:
        """B-6: client 提交 self_master。"""
        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            raise HTTPException(400, "SecAgg not enabled")
        sm_raw = bytes.fromhex(sm_hex)
        coord.submit_self_master(client_id, sm_raw)
        return {"accepted": True}

    def secagg_get_status(self, round_idx: int) -> Dict[str, Any]:
        """B-6: 查询 SecAgg 状态。"""
        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            return {"enabled": False}
        return {"enabled": True, **coord.get_status()}

    def secagg_get_aggregated_delta(self, round_idx: int) -> Dict[str, Any]:
        """B-6: 获取 SecAgg 聚合后的 delta（用于 pipeline 模式）。"""
        coord = self._get_or_create_secagg_coordinator(round_idx)
        if coord is None:
            raise HTTPException(400, "SecAgg not enabled")
        if not coord.all_self_masters_received():
            return {"done": False, "status": "waiting for self_masters"}
        agg_delta = coord.aggregate_and_unmask()
        return {"done": True, "agg_delta": agg_delta}


def load_initial_state(args: argparse.Namespace, minio: MinIOClient) -> Dict[str, torch.Tensor]:
    key0 = global_state_key(0)
    if args.init_state:
        path = Path(args.init_state)
        logger.info("Loading init state from %s", path)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        raise ValueError(f"init state must be a state_dict tensor map: {path}")

    if minio.exists(key0):
        logger.info("Loading init state from MinIO %s", key0)
        obj = minio.get_torch(key0, map_location="cpu")
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        raise ValueError(f"MinIO object {key0} is not a state_dict")

    if args.model_path:
        from transformers import AutoModelForCausalLM

        logger.info("Loading model on CPU from %s", args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        del model
        return state

    raise SystemExit(
        "No initial state. Provide --init-state PATH.pt, or upload "
        f"{key0} via scripts/bootstrap_initial_state.py, or pass --model-path."
    )


def build_app(server: AggregationServer) -> FastAPI:
    app = FastAPI(title="FedScale Aggregation Server", version="0.1.0")

    # SEC-0：控制面鉴权。auth_token 非空时所有 /api/** 要求 Bearer token；/health 不鉴权
    from fastapi import Depends, Header, Request
    from fastapi.responses import JSONResponse

    async def _auth(authorization: Optional[str] = Header(default=None)):
        if not server.auth_token:
            return
        expected = f"Bearer {server.auth_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid or missing auth token")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/round/current")
    def current_round(_: None = Depends(_auth)) -> Dict[str, Any]:
        return server.status()

    @app.get("/api/round/{round_idx}/plan")
    def round_plan(round_idx: int, _: None = Depends(_auth)) -> Dict[str, Any]:
        return server.get_plan(round_idx)

    @app.post("/api/round/{round_idx}/client/{client_id}/upload-complete")
    def upload_complete(
        round_idx: int, client_id: int, body: UploadCompleteBody, _: None = Depends(_auth)
    ) -> Dict[str, Any]:
        return server.upload_complete(round_idx, client_id, body)

    @app.post("/api/round/{round_idx}/client/{client_id}/timing")
    def client_timing(
        round_idx: int, client_id: int, body: ClientTimingBody, _: None = Depends(_auth)
    ) -> Dict[str, Any]:
        return server.report_client_timing(round_idx, client_id, body)

    @app.get("/api/round/{round_idx}/result")
    def round_result(round_idx: int, _: None = Depends(_auth)) -> Dict[str, Any]:
        return server.get_result(round_idx)

    # 流式 per-block pipeline API
    @app.post("/api/round/{round_idx}/client/{client_id}/block/{block_idx}/uploaded")
    def block_uploaded(
        round_idx: int, client_id: int, block_idx: int,
        body: BlockUploadBody,
        _: None = Depends(_auth),
    ) -> Dict[str, Any]:
        return server.block_uploaded(
            round_idx, client_id, block_idx,
            body.num_examples, body.train_loss, body.eval_loss, body.block_energies,
        )

    @app.get("/api/round/{round_idx}/block-status")
    def block_status(round_idx: int, _: None = Depends(_auth)) -> Dict[str, Any]:
        return server.check_block_agg_done(round_idx)

    # B-6: SecAgg REST API
    @app.post("/api/round/{round_idx}/client/{client_id}/secagg/key-announce")
    def secagg_key_announce(
        round_idx: int, client_id: int, body: dict, _: None = Depends(_auth),
    ) -> Dict[str, Any]:
        return server.secagg_key_announce(
            round_idx, client_id, body.get("pk_hex", ""),
            window_amax=body.get("window_amax"),
            global_amax=body.get("global_amax"),
        )

    @app.get("/api/round/{round_idx}/secagg/peer-keys")
    def secagg_peer_keys(round_idx: int, _: None = Depends(_auth)) -> Dict[str, Any]:
        return server.secagg_get_peer_keys(round_idx)

    @app.post("/api/round/{round_idx}/client/{client_id}/secagg/masked-window/{window_id}")
    def secagg_submit_masked_window(
        round_idx: int, client_id: int, window_id: int,
        body: dict, _: None = Depends(_auth),
    ) -> Dict[str, Any]:
        z_hex = body.get("z_hex", "")
        z_key = body.get("z_key", "")
        vector_len = int(body.get("vector_len", 0))
        num_examples = int(body.get("num_examples", 1))
        train_loss = float(body.get("train_loss", 0.0))
        eval_loss = body.get("eval_loss")
        if eval_loss is not None:
            eval_loss = float(eval_loss)
        block_energies = body.get("block_energies", [])
        result = server.secagg_block_uploaded(
            round_idx, client_id, window_id, z_hex, vector_len,
            num_examples, train_loss, eval_loss, block_energies,
            z_key=z_key,
        )
        # 全部 window 齐了才后台 finalize；不要每个 block 都开线程
        if result.get("all_windows_uploaded"):
            server.secagg_schedule_finalize(round_idx)
        return result

    @app.post("/api/round/{round_idx}/client/{client_id}/secagg/masked-blob")
    def secagg_submit_masked_blob(
        round_idx: int, client_id: int,
        body: dict, _: None = Depends(_auth),
    ) -> Dict[str, Any]:
        result = server.secagg_blob_uploaded(
            round_idx, client_id,
            z_key=str(body.get("z_key") or ""),
            window_ids=list(body.get("window_ids") or []),
            num_examples=int(body.get("num_examples", 1)),
            train_loss=float(body.get("train_loss", 0.0)),
            eval_loss=(float(body["eval_loss"]) if body.get("eval_loss") is not None else None),
        )
        if result.get("all_windows_uploaded"):
            server.secagg_schedule_finalize(round_idx)
        return result

    @app.post("/api/round/{round_idx}/client/{client_id}/secagg/self-master")
    def secagg_submit_self_master(
        round_idx: int, client_id: int, body: dict, _: None = Depends(_auth),
    ) -> Dict[str, Any]:
        result = server.secagg_submit_self_master(round_idx, client_id, body.get("sm_hex", ""))
        # 先回 200，聚合放到后台，避免最后一个 client 的 HTTP 超时
        server.secagg_schedule_finalize(round_idx)
        return result

    @app.get("/api/round/{round_idx}/secagg/status")
    def secagg_status(round_idx: int, _: None = Depends(_auth)) -> Dict[str, Any]:
        return server.secagg_get_status(round_idx)

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedScale Central Aggregation Server")
    p.add_argument("--config", default="", help="run 配置 YAML（CLI 覆盖 yaml；CFG-1）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--minio-endpoint", default="http://127.0.0.1:9000")
    p.add_argument("--minio-access-key", default="fedscale")
    p.add_argument("--minio-secret-key", default="fedscale-minio-2026")
    p.add_argument("--minio-bucket", default=DEFAULT_BUCKET)
    p.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    p.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    p.add_argument("--ratio", type=float, default=0.2, help="informational; 不单独驱动调度，用 --coverage-h")
    p.add_argument("--coverage-h", type=int, default=DEFAULT_COVERAGE_H)
    p.add_argument("--slots-per-round", type=int, default=1, help="每轮选几个 slot（ALG-2，>1 如 40%%=H5×2）")
    p.add_argument(
        "--compressor",
        default="public_random",
        choices=["public_random", "dense"],
        help="block 选择：public_random=S3R12v3 公开 mask；dense=全量",
    )
    p.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="保留字段（yaml 兼容）；public_random 不使用",
    )
    p.add_argument("--always-on-threshold", type=int, default=4096,
                   help="numel <= this → always_on (LayerNorm/bias/gate scalars); 0=disable")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--memory-decay", type=float, default=DEFAULT_MEMORY_DECAY)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--model-path", default="", help="可选：CPU 加载 HF 模型作为 round-0")
    p.add_argument("--init-state", default="", help="优先：直接加载 state_dict .pt")
    p.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "round_logs" / "s3r12v3-fsdp"),
    )
    p.add_argument(
        "--transfer-dtype",
        default=DEFAULT_TRANSFER_DTYPE,
        help="通信落盘精度: auto(跟随模型)/fp16/fp32/bf16；int8 预留",
    )
    # IO-1
    p.add_argument(
        "--write-full-global-every-n-rounds",
        type=int,
        default=1,
        help="全量 global_state 写盘频率: 1=每轮(默认); N=每 N 轮; 0=只写 round-0 与最终轮",
    )
    # SYNC-1/2
    p.add_argument(
        "--min-clients-to-aggregate",
        type=int,
        default=0,
        help="最低参与人数即可聚合; 0=必须等齐 num_clients（旧行为）",
    )
    p.add_argument(
        "--client-upload-timeout-s",
        type=float,
        default=1800.0,
        help="本轮上传超时（秒）；超时后按 min-clients 策略决定部分聚合或失败",
    )
    p.add_argument(
        "--round-deadline-s",
        type=float,
        default=0.0,
        help="本轮聚合宽限上限（秒）；0=仅用 client-upload-timeout-s",
    )
    # SEC-0
    p.add_argument(
        "--auth-token",
        default="",
        help="控制面共享 token；非空时要求 Authorization: Bearer <token>，空=不鉴权",
    )
    # RES-1
    p.add_argument(
        "--resume-from-round",
        type=int,
        default=0,
        help="从该 round 恢复（需 MinIO 中已有该轮及之前的 delta/state）；0=不续训",
    )
    # OPS-2：MinIO 生命周期
    p.add_argument(
        "--minio-retention-recent-uploads",
        type=int,
        default=0,
        help="保留最近 K 轮 uploads，更早的删除；0=不清理（默认，避免踩实验）",
    )
    # OPS-3：Client 选择
    p.add_argument(
        "--selected-client-ids",
        default="",
        help="每轮只让这些 client 训练，逗号分隔如 0,1；空=全部参与（旧行为）",
    )
    # SEC-4：异常更新防护
    p.add_argument(
        "--delta-norm-clip",
        type=float,
        default=0.0,
        help="client delta L2 范数超过该值则裁剪到该值；0=不裁剪",
    )
    p.add_argument(
        "--delta-norm-reject",
        type=float,
        default=0.0,
        help="client delta L2 范数超过该值则本轮剔除该 client；0=不剔除",
    )
    # SEC-5：TLS
    p.add_argument("--tls-cert", default="", help="SEC-5: TLS 证书路径；启用 https 控制面")
    p.add_argument("--tls-key", default="", help="SEC-5: TLS 私钥路径")
    # SEC-1/2/3：上传隐私（gidx 化 + 加密 + 末 block 填充）
    p.add_argument(
        "--sec-upload-privacy",
        action="store_true",
        default=False,
        help="SEC-1/2/3: 启用上传隐私（payload 不含 key_name, gidx 加密, 末 block 填充）",
    )
    # B-6: SecAgg (Windowed Secure Aggregation v2)
    p.add_argument("--secagg-enabled", action="store_true", default=False,
                   help="B: 启用 Windowed SecAgg（pairwise + self mask, server 只见聚合和）")
    p.add_argument("--secagg-modulus-bits", type=int, default=16,
                   help="B: 模数位宽 8/16/24/32；16=int16(2B,推荐)；8=int8(需自适应scale)")
    p.add_argument("--secagg-scale", type=float, default=0.0,
                   help="B: 定点量化 scale；0=per-window 公开聚合；>0=固定全局 scale")
    p.add_argument("--secagg-stochastic-rounding", action="store_true", default=False,
                   help="B: 随机舍入（让低精度平均无偏）")
    p.add_argument("--secagg-q-min", type=int, default=0,
                   help="B: 最小成功参与者数；0=num_clients")
    p.add_argument("--secagg-hadamard", action="store_true", default=False,
                   help="B: 量化前 Hadamard 旋转（压低动态范围，提升精度）")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="FedScale Central Aggregation Server", add_help=False)
    args = parse_args()
    # CFG-1：yaml 覆盖 argparse 默认值，CLI 显式参数再覆盖 yaml
    cfg = load_run_config(args.config)
    apply_to_args(args, cfg, parser=parser)
    if args.coverage_h <= 0:
        args.coverage_h = max(1, int(round(1.0 / max(args.ratio, 1e-6))))
    # CFG-2：禁止「只改 RATIO 不改 coverage_h」的假开关——当 ratio 与 coverage_h 明显不一致时报错
    if args.ratio > 0 and args.coverage_h > 0:
        implied_ratio = args.slots_per_round / args.coverage_h
        if abs(implied_ratio - args.ratio) > 0.02 and not _arg_was_set_cli("coverage-h"):
            raise SystemExit(
                f"REFUSING TO START: --ratio={args.ratio} but coverage_h={args.coverage_h} "
                f"slots={args.slots_per_round} (implies ratio≈{implied_ratio:.3f}). "
                f"Please set --coverage-h explicitly (e.g. ratio 0.05 -> coverage_h=20)."
            )

    run_snapshot = snapshot_effective_config(args)
    # OPS-3：解析 selected_client_ids
    sel_ids: Optional[List[int]] = None
    if args.selected_client_ids:
        sel_ids = [int(x) for x in args.selected_client_ids.split(",") if x.strip()]
    minio = MinIOClient(
        endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        bucket=args.minio_bucket,
    )
    init_state = load_initial_state(args, minio)
    server = AggregationServer(
        minio=minio,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        seed=args.seed,
        coverage_h=args.coverage_h,
        results_dir=Path(args.results_dir),
        init_state=init_state,
        transfer_dtype_name=args.transfer_dtype,
        block_size=args.block_size,
        memory_decay=args.memory_decay,
        slots_per_round=args.slots_per_round,
        write_full_global_every_n_rounds=args.write_full_global_every_n_rounds,
        client_upload_timeout_s=args.client_upload_timeout_s,
        min_clients_to_aggregate=args.min_clients_to_aggregate,
        round_deadline_s=args.round_deadline_s,
        auth_token=args.auth_token,
        resume_from_round=args.resume_from_round,
        run_config_snapshot=run_snapshot,
        minio_retention_recent_uploads=args.minio_retention_recent_uploads,
        selected_client_ids=sel_ids,
        delta_norm_clip=args.delta_norm_clip,
        delta_norm_reject=args.delta_norm_reject,
        compressor=args.compressor,
        rho=args.rho,
        sec_upload_privacy=args.sec_upload_privacy,
        always_on_threshold=getattr(args, "always_on_threshold", 4096),
        secagg_enabled=getattr(args, "secagg_enabled", False),
        secagg_modulus_bits=getattr(args, "secagg_modulus_bits", 16),
        secagg_scale=getattr(args, "secagg_scale", 0.0),
        secagg_stochastic_rounding=getattr(args, "secagg_stochastic_rounding", False),
        secagg_q_min=getattr(args, "secagg_q_min", 0),
        secagg_hadamard=getattr(args, "secagg_hadamard", False),
    )
    # CFG-1：把生效的 run.yaml 写入 results 目录，保证可复现
    try:
        import yaml as _yaml

        (Path(args.results_dir) / "run.yaml").write_text(
            _yaml.safe_dump(run_snapshot, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        pass
    app = build_app(server)
    # SEC-5：TLS
    ssl_kwargs: Dict[str, Any] = {}
    if args.tls_cert and args.tls_key:
        ssl_kwargs["ssl_certfile"] = args.tls_cert
        ssl_kwargs["ssl_keyfile"] = args.tls_key
        logger.info("SEC-5: TLS enabled cert=%s", args.tls_cert)
    logger.info("Listening on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", **ssl_kwargs)


if __name__ == "__main__":
    main()
