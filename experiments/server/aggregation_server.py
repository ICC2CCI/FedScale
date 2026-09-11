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
    global_delta_key,
    global_state_key,
    plan_key,
    selected_from_jsonable,
    upload_blocks_key,
)
from shared.run_config import apply_to_args, load_run_config, snapshot_effective_config
from shared.state_dict_utils import cpu_state, floating_elem_count

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


class ClientTimingBody(BaseModel):
    """聚合完成后客户端补报完整 round 耗时。"""

    timings: Dict[str, float] = Field(default_factory=dict)


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
        )
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

        self.current_round = 1
        self.lock = threading.Lock()
        self.round_uploads: Dict[int, Dict[int, UploadCompleteBody]] = {}
        self.round_results: Dict[int, Dict[str, Any]] = {}
        self.round_plans: Dict[int, RoundPlan] = {}
        self.round_log: List[Dict[str, Any]] = []
        self.round_meta: Dict[int, Dict[str, Any]] = {}
        self._aggregating = False
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
            metas = self.round_uploads[round_idx]
            weights: List[float] = []
            deltas = []
            losses = []
            eval_losses: List[float] = []

            # SYNC-1：部分聚合时只下载已上传的 client
            participated = sorted(metas.keys())
            t_dl0 = time.monotonic()
            upload_sizes_b: List[int] = []
            for cid in participated:
                key = upload_blocks_key(round_idx, cid)
                try:
                    upload_sizes_b.append(self.minio.object_size(key))
                except Exception:
                    upload_sizes_b.append(0)
                payload = self.minio.get_torch(key, map_location="cpu")
                if isinstance(payload, dict) and "block_delta" in payload:
                    deltas.append(payload["block_delta"])
                    weights.append(float(payload.get("num_examples", metas[cid].num_examples)))
                    losses.append(float(payload.get("train_loss", metas[cid].train_loss)))
                    # payload 或 upload-complete body 都可带 eval
                    ev = payload.get("eval_loss", metas[cid].eval_loss)
                    if ev is not None:
                        eval_losses.append(float(ev))
                else:
                    deltas.append(payload)
                    weights.append(float(metas[cid].num_examples))
                    losses.append(float(metas[cid].train_loss))
                    if metas[cid].eval_loss is not None:
                        eval_losses.append(float(metas[cid].eval_loss))
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
            agg_delta = apply_block_delta(
                self.global_state,
                deltas,
                weights,
                selected,
                out_dtype=self.transfer_dtype,
            )
            t_apply = time.monotonic() - t_apply0

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
                ev = metas[cid].eval_loss
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
