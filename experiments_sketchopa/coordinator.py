"""SketchOPA 实验 Central 协调器（轻量 REST + MinIO）。

不依赖现有 aggregation_server，独立运行。
通过 HTTP 协调两个 ICC client，通过 MinIO 传输模型/sketch（独立 key 前缀 sketchopa/）。

流程（每轮）：
  1. coordinator 发布 round 开始 + 全局模型
  2. 各 client 下载全局模型 → 本地训练 → Top-k/Sketch 编码 → 上传
  3. coordinator 聚合（dense/topk/sketchopa）→ 发布新全局模型
  4. 各 client 下载新模型 → eval → 上报

运行（Central 节点）：
    source /home/pcllgr/miniconda3/bin/activate fedscale-server
    python experiments_sketchopa/coordinator.py --port 8090 --method dense --num-rounds 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sketchopa.layout import build_layout, SketchLayout
from sketchopa.count_sketch import CountSketchEncoder
from sketchopa.recover import recover_topk, compute_k_recover, build_sparse_update, apply_update_to_state_dict
from sketchopa.quant import quantize_to_zq, zq_to_float
from sketchopa.shprg import SHPRG
from sketchopa.seedshare import TwoClientSeedExchange, SHAMIR_PRIME
from sketchopa.topk import flat_state_dict, unflat_to_state_dict

logger = logging.getLogger("sketchopa_coordinator")

MINIO_BUCKET = "fedscale-bucket"
KEY_PREFIX = "sketchopa"


# ---- Pydantic 模型（模块级，Pydantic v2 要求） ----
from pydantic import BaseModel as _BaseModel, Field as _Field
from typing import Any as _Any


class ClientMetaBody(_BaseModel):
    client_id: int
    num_params: int
    layer_meta: List[List[_Any]]  # [[name, start, end], ...]
    num_examples: int


class UploadNotifyBody(_BaseModel):
    client_id: int
    round_idx: int
    num_examples: int
    train_loss: float
    method_specific: Dict = _Field(default_factory=dict)


class EvalReportBody(_BaseModel):
    client_id: int
    round_idx: int
    eval_loss: float


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--method", choices=["dense", "topk", "sketchopa"], required=True)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--num-clients", type=int, default=2)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--minio-endpoint", type=str, default="http://192.168.235.42:9000")
    p.add_argument("--minio-access-key", type=str, default="fedscale")
    p.add_argument("--minio-secret-key", type=str, default="fedscale-minio-2026")

    # sketchopa 参数（需与 client 一致）
    p.add_argument("--topk-ratio", type=float, default=0.0167)
    p.add_argument("--recover-rho", type=float, default=1.0)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--budget-ratio", type=float, default=1.0)
    p.add_argument("--allocation", choices=["uniform_param", "topk_count_calibrated"], default="uniform_param")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--secure", action="store_true", default=True)
    p.add_argument("--no-secure", dest="secure", action="store_false")
    p.add_argument("--clip-norm", type=float, default=10.0)
    p.add_argument("--quant-scale", type=float, default=1e5)
    p.add_argument("--sketch-seed", type=int, default=20260623)
    p.add_argument("--calibration-path", type=str, default=None)

    # 模型元信息（从 client 上报获取）
    p.add_argument("--model-meta-timeout", type=int, default=120, help="等待 client 上报模型元信息秒数")
    return p.parse_args()


class CoordinatorState:
    def __init__(self, args):
        self.args = args
        self.round_idx = 0
        self.num_clients = args.num_clients
        self.client_meta: Dict[int, Dict] = {}  # client_id -> {num_params, layer_meta, num_examples}
        self.model_meta_received = 0
        self.uploads: Dict[int, Dict] = {}  # round -> {client_id -> upload}
        self.eval_results: Dict[int, Dict] = {}
        self.layout: Optional[SketchLayout] = None
        self.encoder: Optional[CountSketchEncoder] = None
        self.shprg: Optional[SHPRG] = None
        self.exchange: Optional[TwoClientSeedExchange] = None
        self.round_log: List[Dict] = []
        self.lock = threading.Lock()
        self.round_event = threading.Event()
        self.out_dir = args.out_dir or f"results_sketchopa/{args.method}_coord_{int(time.time())}"
        os.makedirs(self.out_dir, exist_ok=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    state = CoordinatorState(args)

    # MinIO client
    from minio import Minio
    minio = Minio(
        args.minio_endpoint.replace("http://", "").replace("https://", ""),
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=False,
    )
    if not minio.bucket_exists(MINIO_BUCKET):
        minio.make_bucket(MINIO_BUCKET)
    state.minio = minio

    # 启动 REST
    import uvicorn
    from fastapi import FastAPI, HTTPException

    app = FastAPI()

    # Pydantic 模型已移到模块级

    from fastapi import Body as FastBody

    @app.post("/api/meta")
    def report_meta(body: ClientMetaBody = FastBody(...)):
        should_init = False
        with state.lock:
            state.client_meta[body.client_id] = {
                "num_params": body.num_params,
                "layer_meta": [(r[0], r[1], r[2]) for r in body.layer_meta],
                "num_examples": body.num_examples,
            }
            state.model_meta_received += 1
            logger.info("Client %d meta received (%d/%d)", body.client_id, state.model_meta_received, state.num_clients)
            if state.model_meta_received >= state.num_clients and state.layout is None:
                should_init = True
        # layout 构建在后台线程执行，避免阻塞 HTTP 响应
        if should_init:
            t = threading.Thread(target=_init_layout, args=(state,), daemon=True)
            t.start()
        return {"ok": True}

    @app.get("/api/round/{round_idx}/client/{client_id}/start")
    def round_start(round_idx: int, client_id: int):
        return {
            "round_idx": round_idx,
            "method": state.args.method,
            "global_model_key": f"{KEY_PREFIX}/round_{round_idx}/global_state.pt",
        }

    @app.post("/api/round/{round_idx}/client/{client_id}/upload")
    def upload_notify(round_idx: int, client_id: int, body: UploadNotifyBody = FastBody(...)):
        should_aggregate = False
        with state.lock:
            if round_idx not in state.uploads:
                state.uploads[round_idx] = {}
            state.uploads[round_idx][client_id] = {
                "num_examples": body.num_examples,
                "train_loss": body.train_loss,
                "method_specific": body.method_specific,
            }
            n = len(state.uploads[round_idx])
            logger.info("Round %d: client %d uploaded (%d/%d)", round_idx, client_id, n, state.num_clients)
            # 只有第一个达到 num_clients 的才触发聚合（防止重复）
            if n >= state.num_clients and not state.uploads[round_idx].get("_aggregating", False):
                state.uploads[round_idx]["_aggregating"] = True
                should_aggregate = True
        # 聚合在锁外执行，避免阻塞其他 HTTP 请求
        if should_aggregate:
            _do_aggregate(state, round_idx)
        return {"ok": True}

    @app.get("/api/round/{round_idx}/status")
    def round_status(round_idx: int):
        with state.lock:
            # done = 该轮聚合已完成（round_idx+1 的全局模型已写入，或 state.round_idx 已超过 round_idx）
            done = state.round_idx > round_idx
            return {"round_idx": round_idx, "done": done, "global_model_key": f"{KEY_PREFIX}/round_{round_idx+1}/global_state.pt"}

    @app.post("/api/round/{round_idx}/client/{client_id}/eval")
    def eval_report(round_idx: int, client_id: int, body: EvalReportBody = FastBody(...)):
        with state.lock:
            if round_idx not in state.eval_results:
                state.eval_results[round_idx] = {}
            state.eval_results[round_idx][client_id] = body.eval_loss
            logger.info("Round %d: client %d eval_loss=%.4f", round_idx, client_id, body.eval_loss)
            # 更新 round_log 中该轮的 eval_losses
            for entry in state.round_log:
                if entry["round"] == round_idx:
                    entry["eval_losses"] = {str(k): round(v, 4) for k, v in state.eval_results[round_idx].items()}
                    break
            with open(os.path.join(state.out_dir, "round_log.json"), "w") as f:
                json.dump(state.round_log, f, indent=2)
        return {"ok": True}

    @app.get("/api/config")
    def get_config():
        return {k: v for k, v in vars(args).items() if not k.startswith("_")}

    logger.info("Coordinator on port %d, method=%s, out=%s", args.port, args.method, state.out_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def _init_layout(state: CoordinatorState):
    """初始化 Sketch layout（sketchopa 方法）。"""
    args = state.args
    if args.method != "sketchopa":
        return
    # 用第一个 client 的 layer_meta 构造 layout
    meta = list(state.client_meta.values())[0]
    layer_meta = meta["layer_meta"]
    d = meta["num_params"]
    k = int(args.topk_ratio * d)
    k = max(k, 1)

    # 直接构造 layout，不分配 630M dummy tensor
    from sketchopa.layout import LayerLayout, SketchLayout as _SL, _allocate_buckets, _load_calibration_weights
    layers = []
    for name, start, end in layer_meta:
        layers.append(LayerLayout(name=name, start=start, end=end,
                                   num_params=end - start, buckets_per_row=0))
    topk16_payload = k * 6
    total_budget = int(args.budget_ratio * topk16_payload / (args.rows * 2))
    total_budget = max(total_budget, 8 * len(layers))
    if args.allocation == "topk_count_calibrated":
        weights = _load_calibration_weights(layer_meta, args.calibration_path)
    else:
        weights = [float(end - start) for (_, start, end) in layer_meta]
    buckets = _allocate_buckets(layer_meta, total_budget, weights, 8)
    for i, layer in enumerate(layers):
        layer.buckets_per_row = buckets[i]
    state.layout = _SL(rows=args.rows, value_bytes=2, idx_bytes=4,
                        layers=layers, allocation_strategy=args.allocation,
                        budget_ratio=args.budget_ratio, min_buckets=8, bucket_factor=8)
    state.layout.layout_hash = state.layout.compute_hash()
    state.encoder = CountSketchEncoder(state.layout, sketch_seed=args.sketch_seed)
    if args.secure:
        state.shprg = SHPRG(length=state.layout.sketch_len, ctx_hash=state.layout.layout_hash,
                            modulus_q=SHAMIR_PRIME, seed_dim=4)
        state.exchange = TwoClientSeedExchange(modulus=SHAMIR_PRIME)
    logger.info("Layout: %s, sketch_len=%d, k=%d, d=%d",
                args.allocation, state.layout.sketch_len, k, d)
    with open(os.path.join(state.out_dir, "layout.json"), "w") as f:
        json.dump(state.layout.to_dict(), f, indent=2)


def _do_aggregate(state: CoordinatorState, round_idx: int):
    """聚合各 client 上传，写新全局模型到 MinIO。"""
    args = state.args
    uploads = state.uploads[round_idx]

    # 下载各 client 的上传数据
    minio = state.minio
    client_data = {}
    for cid, meta in uploads.items():
        key = f"{KEY_PREFIX}/round_{round_idx}/client_{cid}/upload.pt"
        import io
        try:
            resp = minio.get_object(MINIO_BUCKET, key)
            data = resp.read()
            buf = io.BytesIO(data)
            client_data[cid] = torch.load(buf, map_location="cpu", weights_only=False)
        finally:
            resp.close()
            resp.release_conn()

    # 聚合
    if args.method == "dense":
        agg_state = _aggregate_dense(client_data)
    elif args.method == "topk":
        agg_state = _aggregate_topk(client_data, state, round_idx)
    elif args.method == "sketchopa":
        agg_state = _aggregate_sketchopa(client_data, state, round_idx)
    else:
        raise ValueError(args.method)

    # 写新全局模型到 MinIO
    import io
    buf = io.BytesIO()
    torch.save(agg_state, buf)
    buf.seek(0)
    key = f"{KEY_PREFIX}/round_{round_idx+1}/global_state.pt"
    minio.put_object(MINIO_BUCKET, key, buf, len(buf.getvalue()))
    logger.info("Round %d aggregated, new global state -> %s", round_idx, key)

    # 记录日志
    eval_losses = state.eval_results.get(round_idx, {})
    entry = {
        "round": round_idx,
        "train_loss": round(sum(u["train_loss"] for u in uploads.values()) / len(uploads), 4),
        "eval_losses": {str(k): round(v, 4) for k, v in eval_losses.items()},
        "num_examples": sum(u["num_examples"] for u in uploads.values()),
    }
    state.round_log.append(entry)
    with open(os.path.join(state.out_dir, "round_log.json"), "w") as f:
        json.dump(state.round_log, f, indent=2)
    state.round_idx = round_idx + 1


def _aggregate_dense(client_data: Dict[int, Dict]) -> Dict[str, torch.Tensor]:
    total_ex = sum(d["num_examples"] for d in client_data.values())
    # 取第一个 client 的 state 结构
    first_state = client_data[list(client_data.keys())[0]]["state"]
    agg = {}
    for key in first_state:
        agg[key] = sum(d["state"][key] * (d["num_examples"] / total_ex) for d in client_data.values())
    return agg


def _aggregate_topk(client_data, state, round_idx):
    meta = list(state.client_meta.values())[0]
    layer_meta = meta["layer_meta"]
    d = meta["num_params"]
    total_ex = sum(d["num_examples"] for d in client_data.values())

    agg_flat = torch.zeros(d, dtype=torch.float32)
    for cid, data in client_data.items():
        w = data["num_examples"] / total_ex
        agg_flat[data["coords"]] += data["vals"] * w

    # 需要全局模型 before（从 MinIO 下载）
    global_before = _load_global_state(state.minio, round_idx)
    return apply_update_to_state_dict(agg_flat, layer_meta, global_before)


def _aggregate_sketchopa(client_data, state, round_idx):
    args = state.args
    layout = state.layout
    encoder = state.encoder
    meta = list(state.client_meta.values())[0]
    layer_meta = meta["layer_meta"]
    d = meta["num_params"]

    if args.secure and state.shprg is not None:
        cts = [data["ct_zq"] for data in client_data.values()]
        seeds = [data["seed_vec"] for data in client_data.values()]
        agg_share = state.exchange.auxiliary_aggregate(seeds[0], state.exchange.client_pack_share(seeds[1]))
        s_sum = state.exchange.server_reconstruct([agg_share])
        agg_ct = sum(cts) % SHAMIR_PRIME
        agg_zq = state.shprg.unmask_sketch_zq(agg_ct, s_sum)
        agg_sketch = zq_to_float(agg_zq, args.quant_scale, SHAMIR_PRIME)
    else:
        agg_zq = sum(data["ct_zq"] for data in client_data.values()) % SHAMIR_PRIME
        agg_sketch = zq_to_float(agg_zq, args.quant_scale, SHAMIR_PRIME)

    candidates = torch.cat([data["coords"] for data in client_data.values()])
    k = int(args.topk_ratio * d)
    k_rec = compute_k_recover(k, len(client_data), args.recover_rho, d)
    coords_r, vals_r, rec_stats = recover_topk(
        agg_sketch, encoder, layout, candidates, k_rec, clip_norm=args.clip_norm,
    )
    agg_flat = build_sparse_update(coords_r, vals_r, d)
    global_before = _load_global_state(state.minio, round_idx)
    return apply_update_to_state_dict(agg_flat, layer_meta, global_before)


def _load_global_state(minio, round_idx) -> Dict[str, torch.Tensor]:
    import io
    key = f"{KEY_PREFIX}/round_{round_idx}/global_state.pt"
    resp = minio.get_object(MINIO_BUCKET, key)
    try:
        data = resp.read()
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    finally:
        resp.close()
        resp.release_conn()


if __name__ == "__main__":
    main()
