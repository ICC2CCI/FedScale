"""Central Server：REST API + S3R12v3 block FedAvg 聚合。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.block_scheduler import BlockScheduler
from shared.block_selection import apply_block_delta
from shared.minio_client import MinIOClient
from shared.protocol import (
    DEFAULT_BUCKET,
    DEFAULT_COVERAGE_H,
    DEFAULT_NUM_CLIENTS,
    DEFAULT_NUM_ROUNDS,
    DEFAULT_SEED,
    RoundPlan,
    global_state_key,
    plan_key,
    selected_from_jsonable,
    upload_blocks_key,
)
from shared.state_dict_utils import cpu_state, floating_elem_count

logger = logging.getLogger("aggregation_server")


class UploadCompleteBody(BaseModel):
    num_examples: int = Field(ge=1)
    train_loss: float = 0.0


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
    ) -> None:
        self.minio = minio
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.seed = seed
        self.coverage_h = coverage_h
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.global_state = cpu_state(init_state)
        self.scheduler = BlockScheduler(self.global_state, seed=seed, coverage_h=coverage_h)
        self.current_round = 1
        self.lock = threading.Lock()
        self.round_uploads: Dict[int, Dict[int, UploadCompleteBody]] = {}
        self.round_results: Dict[int, Dict[str, Any]] = {}
        self.round_plans: Dict[int, RoundPlan] = {}
        self.round_log: List[Dict[str, Any]] = []
        self._aggregating = False

        # 写入 round-0 初始全局状态 + 第 1 轮 plan
        if not self.minio.exists(global_state_key(0)):
            logger.info("Uploading initial global_state/round-0/state.pt")
            self.minio.put_torch(global_state_key(0), self.global_state)
        self._ensure_plan(1)

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
            return self._ensure_plan(round_idx).to_dict()

    def upload_complete(self, round_idx: int, client_id: int, body: UploadCompleteBody) -> Dict[str, Any]:
        if client_id < 0 or client_id >= self.num_clients:
            raise HTTPException(400, f"client_id must be in [0, {self.num_clients})")
        if round_idx < 1 or round_idx > self.num_rounds:
            raise HTTPException(404, f"round {round_idx} out of range")

        key = upload_blocks_key(round_idx, client_id)
        if not self.minio.exists(key):
            raise HTTPException(400, f"missing MinIO object: {key}")

        should_aggregate = False
        with self.lock:
            if round_idx != self.current_round:
                raise HTTPException(409, f"expected round {self.current_round}, got {round_idx}")
            bucket = self.round_uploads.setdefault(round_idx, {})
            bucket[client_id] = body
            logger.info(
                "Upload complete round=%s client=%s (%s/%s) loss=%.4f n=%s",
                round_idx,
                client_id,
                len(bucket),
                self.num_clients,
                body.train_loss,
                body.num_examples,
            )
            if len(bucket) >= self.num_clients and round_idx not in self.round_results and not self._aggregating:
                self._aggregating = True
                should_aggregate = True

        if should_aggregate:
            threading.Thread(target=self._aggregate_round, args=(round_idx,), daemon=True).start()
        return {"accepted": True, "received": len(self.round_uploads.get(round_idx, {}))}

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

    def _aggregate_round(self, round_idx: int) -> None:
        try:
            plan = self._ensure_plan(round_idx)
            selected = selected_from_jsonable(plan.selected_by_key)
            metas = self.round_uploads[round_idx]
            weights: List[float] = []
            deltas = []
            losses = []
            for cid in range(self.num_clients):
                payload = self.minio.get_torch(upload_blocks_key(round_idx, cid), map_location="cpu")
                if isinstance(payload, dict) and "block_delta" in payload:
                    deltas.append(payload["block_delta"])
                    weights.append(float(payload.get("num_examples", metas[cid].num_examples)))
                    losses.append(float(payload.get("train_loss", metas[cid].train_loss)))
                else:
                    deltas.append(payload)
                    weights.append(float(metas[cid].num_examples))
                    losses.append(float(metas[cid].train_loss))

            apply_block_delta(self.global_state, deltas, weights, selected)
            self.minio.put_torch(global_state_key(round_idx), self.global_state)

            avg_train = sum(losses) / max(len(losses), 1)
            result = {
                "round": round_idx,
                "done": True,
                "eval_loss": None,
                "avg_train_loss": round(avg_train, 6),
                "upload_ratio": round(plan.upload_ratio, 6),
                "n_selected_blocks": plan.n_selected_blocks,
                "selected_elems": plan.selected_elems,
                "message": "aggregated",
            }
            entry = {
                "round": round_idx,
                "epoch": plan.epoch,
                "slot": plan.slot,
                "avg_train_loss": result["avg_train_loss"],
                "eval_loss": None,
                "pct_of_total": round(plan.upload_ratio * 100, 2),
                "n_selected_blocks": plan.n_selected_blocks,
                "selected_elems_M": round(plan.selected_elems / 1e6, 3),
            }
            with self.lock:
                self.round_results[round_idx] = result
                self.round_log.append(entry)
                self._write_round_log()
                if round_idx < self.num_rounds:
                    self.current_round = round_idx + 1
                    self._ensure_plan(self.current_round)
                else:
                    self.current_round = self.num_rounds + 1
                self._aggregating = False
            logger.info("Aggregated round %s avg_train=%.4f ratio=%.2f%%", round_idx, avg_train, plan.upload_ratio * 100)
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

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/round/current")
    def current_round() -> Dict[str, Any]:
        return server.status()

    @app.get("/api/round/{round_idx}/plan")
    def round_plan(round_idx: int) -> Dict[str, Any]:
        return server.get_plan(round_idx)

    @app.post("/api/round/{round_idx}/client/{client_id}/upload-complete")
    def upload_complete(round_idx: int, client_id: int, body: UploadCompleteBody) -> Dict[str, Any]:
        return server.upload_complete(round_idx, client_id, body)

    @app.get("/api/round/{round_idx}/result")
    def round_result(round_idx: int) -> Dict[str, Any]:
        return server.get_result(round_idx)

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedScale Central Aggregation Server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--minio-endpoint", default="http://127.0.0.1:9000")
    p.add_argument("--minio-access-key", default="fedscale")
    p.add_argument("--minio-secret-key", default="fedscale-minio-2026")
    p.add_argument("--minio-bucket", default=DEFAULT_BUCKET)
    p.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    p.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    p.add_argument("--ratio", type=float, default=0.2, help="informational; H derived if --coverage-h unset")
    p.add_argument("--coverage-h", type=int, default=DEFAULT_COVERAGE_H)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--model-path", default="", help="可选：CPU 加载 HF 模型作为 round-0")
    p.add_argument("--init-state", default="", help="优先：直接加载 state_dict .pt")
    p.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "round_logs" / "s3r12v3-fsdp"),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.coverage_h <= 0:
        args.coverage_h = max(1, int(round(1.0 / max(args.ratio, 1e-6))))

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
    )
    app = build_app(server)
    logger.info("Listening on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
