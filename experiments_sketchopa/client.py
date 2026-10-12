"""SketchOPA 实验 Client（ICC 节点，FSDP）。

每个 ICC 节点跑一个 client，通过 HTTP 连接 Central coordinator，通过 MinIO 传输模型/sketch。

运行（ICC1）：
    source /home/pcl/miniconda3/bin/activate flwr-ft
    cd /home/pcl/liuchao/fedscale-icc-1
    accelerate launch --config_file accelerate_config.yaml \
        /home/pcl/liuchao/fedscale-icc-server/experiments_sketchopa/client.py \
        --client-id 0 --server-url http://192.168.235.42:8090 \
        --model-path model/Qwen/Qwen2.5-0.5B \
        --data-path data/splits/icc1_client0_train.json \
        --eval-path data/medical_flashcards_eval.json

注意：experiments_sketchopa 需在 ICC 节点可见（NFS 或拷贝）。
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


_HTTP = None


def _http():
    global _HTTP
    if _HTTP is None:
        _HTTP = _make_session()
    return _HTTP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (
    build_dataloader, build_eval_loader, load_model, load_tokenizer,
    train_local_steps, evaluate, setup_accelerate, get_full_state_fsdp,
)
from sketchopa.layout import build_layout, SketchLayout
from sketchopa.count_sketch import CountSketchEncoder
from sketchopa.topk import (
    global_topk_flat, fsdp_global_topk, ResidualBuffer,
    flat_state_dict, unflat_to_state_dict, compute_delta, count_topk_per_layer,
)
from sketchopa.quant import quantize_to_zq, zq_to_float
from sketchopa.shprg import SHPRG
from sketchopa.seedshare import TwoClientSeedExchange, SHAMIR_PRIME

logger = logging.getLogger("sketchopa_client")

MINIO_BUCKET = "fedscale-bucket"
KEY_PREFIX = "sketchopa"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--server-url", type=str, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--eval-path", type=str, required=True)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--local-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--minio-endpoint", type=str, default="http://192.168.235.42:9000")
    p.add_argument("--minio-access-key", type=str, default="fedscale")
    p.add_argument("--minio-secret-key", type=str, default="fedscale-minio-2026")
    p.add_argument("--seed", type=int, default=42)
    # sketchopa 参数从 coordinator config 获取，这里设默认值兜底
    p.add_argument("--topk-ratio", type=float, default=0.0167)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--residual-clip", type=float, default=0.0)
    p.add_argument("--residual-decay", type=float, default=1.0)
    p.add_argument("--quant-scale", type=float, default=1e5)
    p.add_argument("--sketch-seed", type=int, default=20260623)
    p.add_argument("--save-calibration", type=str, default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = setup_accelerate()
    is_main = accelerator.is_main_process

    # 从 coordinator 获取 config
    config = _get_config(args.server_url)
    method = config.get("method", "dense")
    logger.info("Client %d: method=%s", args.client_id, method)

    # MinIO
    from minio import Minio
    minio = Minio(
        args.minio_endpoint.replace("http://", "").replace("https://", ""),
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=False,
    )

    # 模型 + tokenizer
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path)
    model = model.to(accelerator.device)

    # 数据
    loader = build_dataloader(args.data_path, tokenizer, args.seq_len, args.batch_size)
    eval_loader = build_eval_loader(args.eval_path, tokenizer, args.seq_len, args.batch_size)

    # 上报模型元信息
    if is_main:
        ref_state = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
        flat_ref, layer_meta = flat_state_dict(ref_state)
        d = flat_ref.numel()
        _report_meta(args.server_url, args.client_id, d, layer_meta, len(loader.dataset))
        logger.info("Reported meta: d=%d, layers=%d, examples=%d", d, len(layer_meta), len(loader.dataset))

    accelerator.wait_for_everyone()

    # sketchopa: 初始化 layout/encoder（is_main）
    layout = None
    encoder = None
    shprg = None
    exchange = None
    residual = None
    if method == "sketchopa" and is_main:
        k = int(args.topk_ratio * d)
        k = max(k, 1)
        # 直接构造 layout，不分配 630M 的 dummy tensor
        from sketchopa.layout import LayerLayout, SketchLayout, _allocate_buckets, _load_calibration_weights
        layers = []
        for name, start, end in layer_meta:
            n = end - start
            layers.append(LayerLayout(name=name, start=start, end=end,
                                       num_params=n, buckets_per_row=0))
        topk16_payload = k * 6
        total_budget = int(1.0 * topk16_payload / (config.get("rows", 5) * 2))
        total_budget = max(total_budget, 8 * len(layers))
        allocation = config.get("allocation", "uniform_param")
        cal_path = config.get("calibration_path")
        if allocation == "topk_count_calibrated":
            weights = _load_calibration_weights(layer_meta, cal_path)
        else:
            weights = [float(end - start) for (_, start, end) in layer_meta]
        buckets = _allocate_buckets(layer_meta, total_budget, weights, 8)
        for i, layer in enumerate(layers):
            layer.buckets_per_row = buckets[i]
        layout = SketchLayout(rows=config.get("rows", 5), value_bytes=2, idx_bytes=4,
                               layers=layers, allocation_strategy=allocation,
                               budget_ratio=1.0, min_buckets=8, bucket_factor=8)
        layout.layout_hash = layout.compute_hash()
        encoder = CountSketchEncoder(layout, sketch_seed=args.sketch_seed)
        if config.get("secure", True):
            shprg = SHPRG(length=layout.sketch_len, ctx_hash=layout.layout_hash,
                          modulus_q=SHAMIR_PRIME, seed_dim=4)
            exchange = TwoClientSeedExchange(modulus=SHAMIR_PRIME)
        logger.info("Layout: sketch_len=%d, k=%d", layout.sketch_len, k)

    if config.get("residual", False) and is_main:
        residual = ResidualBuffer(layer_meta, clip_norm=args.residual_clip,
                                  decay=args.residual_decay)

    calibration_counts: Dict[str, int] = {name: 0 for name, _, _ in (layer_meta if is_main else [])}

    for round_idx in range(args.num_rounds):
        t0 = time.monotonic()
        logger.info("=== Client %d Round %d ===", args.client_id, round_idx)

        # 下载全局模型（round 0：client 0 负责上传初始模型）
        if is_main:
            if round_idx == 0 and args.client_id == 0:
                # 上传初始模型
                init_state = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
                _save_global_state(minio, 0, init_state)
                logger.info("Uploaded initial global model to round_0")
            global_state = _load_global_state(minio, round_idx)
            _load_state_to_model(model, global_state, accelerator)
        accelerator.wait_for_everyone()

        # 本地训练
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, s / max(1, int(0.1 * args.local_steps))))
        model.train()
        train_loss, _ = train_local_steps(
            model, loader, optimizer, scheduler,
            local_steps=args.local_steps, device=accelerator.device,
            grad_accum=args.grad_accum,
        )

        if is_main:
            after_state = get_full_state_fsdp(model)
            after_state = {k: v.detach().cpu().to(torch.float32) for k, v in after_state.items()}
            delta = compute_delta(global_state, after_state)
            flat_delta, _ = flat_state_dict(delta)

            upload_data = {"num_examples": len(loader.dataset), "train_loss": train_loss}

            # k 对 topk/sketchopa 都需要
            k = int(args.topk_ratio * d)
            k = max(k, 1)

            if method == "dense":
                upload_data["state"] = after_state

            elif method == "topk":
                if residual is not None:
                    coords, vals = residual.add_and_select(flat_delta, k)
                else:
                    coords, vals = global_topk_flat(flat_delta, k)
                upload_data["coords"] = coords
                upload_data["vals"] = vals
                # calibration 统计
                cnt = count_topk_per_layer(coords, layout) if layout else {}
                for name, c in cnt.items():
                    calibration_counts[name] = calibration_counts.get(name, 0) + c

            elif method == "sketchopa":
                if residual is not None:
                    coords, vals = residual.add_and_select(flat_delta, k)
                else:
                    coords, vals = global_topk_flat(flat_delta, k)
                cnt = count_topk_per_layer(coords, layout)
                for name, c in cnt.items():
                    calibration_counts[name] = calibration_counts.get(name, 0) + c

                sketch = encoder.encode(coords, vals)
                sk_zq = quantize_to_zq(sketch, args.quant_scale, SHAMIR_PRIME)

                if shprg is not None:
                    seed_vec = _sample_seed_vec(args.seed, round_idx, args.client_id)
                    ct_zq = shprg.mask_sketch_zq(sk_zq, seed_vec)
                    upload_data["seed_vec"] = seed_vec
                else:
                    ct_zq = sk_zq
                upload_data["ct_zq"] = ct_zq
                upload_data["coords"] = coords  # 实验记录用

            # 上传到 MinIO
            buf = io.BytesIO()
            torch.save(upload_data, buf)
            buf.seek(0)
            key = f"{KEY_PREFIX}/round_{round_idx}/client_{args.client_id}/upload.pt"
            minio.put_object(MINIO_BUCKET, key, buf, len(buf.getvalue()))

            # 通知 coordinator
            method_specific = {}
            if method == "sketchopa":
                method_specific = {"sketch_len": layout.sketch_len if layout else 0}
            _notify_upload(args.server_url, round_idx, args.client_id,
                           len(loader.dataset), train_loss, method_specific)

        accelerator.wait_for_everyone()

        # 等待聚合完成
        if is_main:
            _wait_round_done(args.server_url, round_idx, timeout=7200)
            # 下载新全局模型
            new_state = _load_global_state(minio, round_idx + 1)
            _load_state_to_model(model, new_state, accelerator)

        accelerator.wait_for_everyone()

        # eval
        eval_loss = evaluate(model, eval_loader, accelerator.device)
        if is_main:
            _report_eval(args.server_url, round_idx, args.client_id, eval_loss)
            elapsed = time.monotonic() - t0
            logger.info("C%d R%d eval_loss=%.4f train_loss=%.4f (%.1fs)",
                        args.client_id, round_idx, eval_loss, train_loss, elapsed)

    # 保存 calibration
    if is_main and args.save_calibration and (method in ("topk", "sketchopa")):
        from sketchopa.layout import save_calibration
        save_calibration(calibration_counts, args.save_calibration,
                         extra={"client_id": args.client_id, "method": method})
        logger.info("Saved calibration to %s", args.save_calibration)

    if is_main:
        logger.info("Client %d done.", args.client_id)


def _get_config(server_url: str) -> Dict:
    resp = _http().get(f"{server_url}/api/config", timeout=60)
    resp.raise_for_status()
    return resp.json()


def _report_meta(server_url: str, client_id: int, num_params: int,
                 layer_meta: List[Tuple[str, int, int]], num_examples: int):
    body = {
        "client_id": client_id,
        "num_params": num_params,
        "layer_meta": [[n, s, e] for n, s, e in layer_meta],
        "num_examples": num_examples,
    }
    _http().post(f"{server_url}/api/meta", json=body, timeout=120).raise_for_status()


def _safe_float(x):
    """防止 NaN/inf 导致 JSON 序列化失败。"""
    import math
    f = float(x)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _notify_upload(server_url, round_idx, client_id, num_examples, train_loss, method_specific):
    body = {
        "client_id": client_id,
        "round_idx": round_idx,
        "num_examples": num_examples,
        "train_loss": _safe_float(train_loss),
        "method_specific": method_specific,
    }
    _http().post(f"{server_url}/api/round/{round_idx}/client/{client_id}/upload", json=body, timeout=120).raise_for_status()


def _wait_round_done(server_url, round_idx, timeout=3600):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            resp = _http().get(f"{server_url}/api/round/{round_idx}/status", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("done"):
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"round {round_idx} not done in {timeout}s")


def _report_eval(server_url, round_idx, client_id, eval_loss):
    body = {"client_id": client_id, "round_idx": round_idx, "eval_loss": _safe_float(eval_loss)}
    _http().post(f"{server_url}/api/round/{round_idx}/client/{client_id}/eval", json=body, timeout=60).raise_for_status()


def _load_global_state(minio) -> Dict[str, torch.Tensor]:
    # round_idx 由调用方控制，这里简化
    pass


def _load_global_state(minio, round_idx) -> Dict[str, torch.Tensor]:
    key = f"{KEY_PREFIX}/round_{round_idx}/global_state.pt"
    resp = minio.get_object(MINIO_BUCKET, key)
    try:
        data = resp.read()
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    finally:
        resp.close()
        resp.release_conn()


def _save_global_state(minio, round_idx, state_dict: Dict[str, torch.Tensor]) -> None:
    key = f"{KEY_PREFIX}/round_{round_idx}/global_state.pt"
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    minio.put_object(MINIO_BUCKET, key, buf, len(buf.getvalue()))


def _load_state_to_model(model, state_dict: Dict[str, torch.Tensor], accelerator) -> None:
    target_dtype = next(model.parameters()).dtype
    sd = {k: v.to(target_dtype) for k, v in state_dict.items()}
    model.load_state_dict(sd, strict=True)


def _sample_seed_vec(base_seed: int, round_idx: int, cid: int) -> List[int]:
    import hashlib
    h = hashlib.sha256(f"seed|{base_seed}|r{round_idx}|c{cid}".encode()).digest()
    return [int.from_bytes(h[i*2:i*2+2], "little") % SHAMIR_PRIME for i in range(4)]


if __name__ == "__main__":
    main()
