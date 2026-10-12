"""SketchOPA 统一实验入口（单进程模拟 2-client 联邦）。

支持三种 method：
- dense：B0 基线，全量 FedAvg
- topk：B1 基线，明文 global Top-k 聚合
- sketchopa：E1-E3，Top-k → CountSketch → 安全聚合 → 恢复

单进程内串行跑两个 client 的训练，中心聚合。FSDP 用于多卡训练加速。
真实双集群部署可后续在此基础上加 REST 层。

运行（ICC 节点，flwr-ft 环境）：
    accelerate launch --config_file configs/accelerate_config.yaml \
        experiments_sketchopa/run_experiment.py \
        --method dense --num-rounds 20 \
        --model-path model/Qwen/Qwen2.5-0.5B \
        --data-paths data/splits/icc1_client0_train.json data/splits/client1_train.json \
        --eval-path data/medical_flashcards_eval.json

    # sketchopa 主实验（calibrated + residual）
    accelerate launch --config_file configs/accelerate_config.yaml \
        experiments_sketchopa/run_experiment.py \
        --method sketchopa --allocation topk_count_calibrated \
        --calibration-path experiments_sketchopa/calibrations/qwen05b_topk_count.json \
        --residual --num-rounds 20 \
        --model-path model/Qwen/Qwen2.5-0.5B \
        --data-paths data/splits/icc1_client0_train.json data/splits/client1_train.json \
        --eval-path data/medical_flashcards_eval.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (
    build_dataloader, build_eval_loader, load_model, load_tokenizer,
    train_local_steps, evaluate, setup_accelerate, get_full_state_fsdp,
)
from sketchopa.layout import build_layout, SketchLayout
from sketchopa.count_sketch import CountSketchEncoder
from sketchopa.topk import (
    global_topk_flat, ResidualBuffer, flat_state_dict, unflat_to_state_dict,
    compute_delta, count_topk_per_layer,
)
from sketchopa.recover import recover_topk, compute_k_recover, build_sparse_update, apply_update_to_state_dict
from sketchopa.quant import quantize, dequantize_fp16, dequantize_int16, quantize_to_zq, zq_to_float
from sketchopa.shprg import SHPRG
from sketchopa.seedshare import TwoClientSeedExchange, SHAMIR_PRIME

logger = logging.getLogger("sketchopa_exp")


def parse_args():
    p = argparse.ArgumentParser()
    # 基本
    p.add_argument("--method", choices=["dense", "topk", "sketchopa"], required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--data-paths", nargs=2, required=True, help="两个 client 的训练数据路径")
    p.add_argument("--eval-path", type=str, required=True)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--local-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # topk / sketchopa
    p.add_argument("--topk-ratio", type=float, default=0.0167, help="k/d 比例")
    p.add_argument("--recover-rho", type=float, default=1.0)

    # sketchopa 专用
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--budget-ratio", type=float, default=1.0)
    p.add_argument("--allocation", choices=["uniform_param", "topk_count_calibrated"], default="uniform_param")
    p.add_argument("--calibration-path", type=str, default=None)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--residual-clip", type=float, default=0.0)
    p.add_argument("--residual-decay", type=float, default=1.0)
    p.add_argument("--transport-dtype", choices=["float16", "int16"], default="float16")
    p.add_argument("--quant-scale", type=float, default=1e6)
    p.add_argument("--sketch-seed", type=int, default=20260623)
    p.add_argument("--secure", action="store_true", default=True, help="启用安全层（默认开）")
    p.add_argument("--no-secure", dest="secure", action="store_false")
    p.add_argument("--clip-norm", type=float, default=10.0, help="恢复后 clip 范数")
    p.add_argument("--save-calibration", type=str, default=None, help="保存各层 Top-k 计数到文件")

    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = setup_accelerate()
    is_main = accelerator.is_main_process

    if args.out_dir is None:
        args.out_dir = f"results_sketchopa/{args.method}_{int(time.time())}"
    os.makedirs(args.out_dir, exist_ok=True)
    if is_main:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path)
    model = model.to(accelerator.device)

    # 数据
    loaders = [
        build_dataloader(args.data_paths[i], tokenizer, args.seq_len, args.batch_size)
        for i in range(2)
    ]
    eval_loader = build_eval_loader(args.eval_path, tokenizer, args.seq_len, args.batch_size)

    # 获取模型元信息（展平）
    ref_state = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
    flat_ref, layer_meta = flat_state_dict(ref_state)
    d = flat_ref.numel()

    # Top-k 数量
    k = int(args.topk_ratio * d)
    k = max(k, 1)

    # Sketch layout（仅 sketchopa 方法需要）
    layout: Optional[SketchLayout] = None
    encoder: Optional[CountSketchEncoder] = None
    if args.method == "sketchopa":
        layout = build_layout(
            ref_state, k=k, rows=args.rows, value_bytes=2, budget_ratio=args.budget_ratio,
            allocation_strategy=args.allocation,
            calibration_path=args.calibration_path,
        )
        encoder = CountSketchEncoder(layout, sketch_seed=args.sketch_seed)
        if is_main:
            logger.info("Layout: %s, sketch_len=%d, payload=%d bytes, k=%d, d=%d",
                        args.allocation, layout.sketch_len, layout.payload_bytes, k, d)
            with open(os.path.join(args.out_dir, "layout.json"), "w") as f:
                json.dump(layout.to_dict(), f, indent=2)

    # 残差缓冲（每个 client 一个）
    residuals = [
        ResidualBuffer(layer_meta, clip_norm=args.residual_clip, decay=args.residual_decay)
        if args.residual else None
        for _ in range(2)
    ]

    # 安全层组件
    shprg: Optional[SHPRG] = None
    exchange: Optional[TwoClientSeedExchange] = None
    if args.method == "sketchopa" and args.secure:
        shprg = SHPRG(length=layout.sketch_len, ctx_hash=layout.layout_hash,
                      modulus_q=SHAMIR_PRIME, seed_dim=4)
        exchange = TwoClientSeedExchange(modulus=SHAMIR_PRIME)

    round_log: List[Dict] = []
    calibration_counts: Dict[str, int] = {name: 0 for name, _, _ in layer_meta}

    for round_idx in range(args.num_rounds):
        t0 = time.monotonic()
        logger.info("=== Round %d (method=%s) ===", round_idx, args.method)

        # 每轮保存训练前 state
        global_state_before = get_full_state_fsdp(model)
        global_state_before = {k: v.detach().cpu().to(torch.float32) for k, v in global_state_before.items()}

        client_uploads: List = []  # 每个 client 的上传内容

        for cid in range(2):
            # 加载全局模型
            _load_state_to_model(model, global_state_before, accelerator)

            # 本地训练
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda s: min(1.0, s / max(1, int(0.1 * args.local_steps))))
            model.train()
            train_loss, _ = train_local_steps(
                model, loaders[cid], optimizer, scheduler,
                local_steps=args.local_steps, device=accelerator.device,
                grad_accum=args.grad_accum,
            )

            # 训练后 state
            after_state = get_full_state_fsdp(model)
            after_state = {k: v.detach().cpu().to(torch.float32) for k, v in after_state.items()}

            # 计算 delta
            delta = compute_delta(global_state_before, after_state)
            flat_delta, _ = flat_state_dict(delta)

            num_examples = len(loaders[cid].dataset)

            if args.method == "dense":
                client_uploads.append({"state": after_state, "num_examples": num_examples,
                                       "train_loss": train_loss})

            elif args.method == "topk":
                # global Top-k（明文）
                if args.residual and residuals[cid] is not None:
                    coords, vals = residuals[cid].add_and_select(flat_delta, k)
                else:
                    coords, vals = global_topk_flat(flat_delta, k)
                # 统计 calibration
                if is_main:
                    cnt = count_topk_per_layer(coords, layout) if layout else {}
                    for name, c in cnt.items():
                        calibration_counts[name] = calibration_counts.get(name, 0) + c
                client_uploads.append({"coords": coords, "vals": vals, "num_examples": num_examples,
                                       "train_loss": train_loss})

            elif args.method == "sketchopa":
                # global Top-k（+ 残差）
                if args.residual and residuals[cid] is not None:
                    coords, vals = residuals[cid].add_and_select(flat_delta, k)
                else:
                    coords, vals = global_topk_flat(flat_delta, k)

                # 统计 calibration
                if is_main:
                    cnt = count_topk_per_layer(coords, layout)
                    for name, c in cnt.items():
                        calibration_counts[name] = calibration_counts.get(name, 0) + c

                # CountSketch 编码
                sketch = encoder.encode(coords, vals)

                # 加权（按 num_examples，简化：2-client 等权）
                # 量化到 Z_q（安全聚合 mask 域）
                sk_zq = quantize_to_zq(sketch, args.quant_scale, SHAMIR_PRIME)

                if args.secure and shprg is not None:
                    # 采样 seed_vec
                    seed_vec = _sample_seed_vec(args.seed, round_idx, cid)
                    ct_zq = shprg.mask_sketch_zq(sk_zq, seed_vec)
                else:
                    ct_zq = sk_zq
                    seed_vec = None

                client_uploads.append({
                    "ct_zq": ct_zq, "seed_vec": seed_vec,
                    "coords": coords,  # 用于 active_union 候选（安全层下 server 不应知道，这里实验记录用）
                    "num_examples": num_examples, "train_loss": train_loss,
                })

        # === Server 聚合 ===
        t_agg0 = time.monotonic()
        agg_state, agg_stats = _server_aggregate(
            args, method=args.method, client_uploads=client_uploads,
            global_state_before=global_state_before, layer_meta=layer_meta,
            layout=layout, encoder=encoder, k=k, d=d,
            shprg=shprg, exchange=exchange,
        )
        t_agg = time.monotonic() - t_agg0

        # 加载聚合后全局模型
        _load_state_to_model(model, agg_state, accelerator)

        # eval
        eval_loss = evaluate(model, eval_loader, accelerator.device)
        elapsed = time.monotonic() - t0

        if is_main:
            entry = {
                "round": round_idx,
                "train_loss": round(sum(u["train_loss"] for u in client_uploads) / len(client_uploads), 4),
                "eval_loss": round(eval_loss, 4),
                "elapsed_s": round(elapsed, 2),
                "agg_s": round(t_agg, 3),
                "num_examples": sum(u["num_examples"] for u in client_uploads),
                **agg_stats,
            }
            round_log.append(entry)
            logger.info("R%d eval_loss=%.4f train_loss=%.4f agg=%.1fs total=%.1fs",
                        round_idx, eval_loss, entry["train_loss"], t_agg, elapsed)
            with open(os.path.join(args.out_dir, "round_log.json"), "w") as f:
                json.dump(round_log, f, indent=2)

    # 保存 calibration
    if is_main and args.save_calibration:
        from sketchopa.layout import save_calibration
        save_calibration(calibration_counts, args.save_calibration,
                         extra={"k": k, "rounds": args.num_rounds, "method": args.method})
        logger.info("Saved calibration to %s", args.save_calibration)

    if is_main:
        logger.info("Done. Results in %s", args.out_dir)


def _server_aggregate(
    args, method: str, client_uploads: List,
    global_state_before: Dict[str, torch.Tensor],
    layer_meta: List[Tuple[str, int, int]],
    layout: Optional[SketchLayout], encoder: Optional[CountSketchEncoder],
    k: int, d: int,
    shprg: Optional[SHPRG], exchange: Optional[TwoClientSeedExchange],
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Server 端聚合，返回 (新全局 state_dict, 统计)。"""
    stats: Dict = {}

    if method == "dense":
        # FedAvg by num_examples
        total_ex = sum(u["num_examples"] for u in client_uploads)
        agg = {}
        for key in client_uploads[0]["state"]:
            agg[key] = sum(u["state"][key] * (u["num_examples"] / total_ex) for u in client_uploads)
        stats["payload_bytes"] = sum(u["state"][next(iter(u["state"]))].numel() for u in client_uploads) * 4
        return agg, stats

    if method == "topk":
        # 明文 Top-k：各 client 的 (coords, vals) 直接求和到稠密向量
        total_ex = sum(u["num_examples"] for u in client_uploads)
        agg_flat = torch.zeros(d, dtype=torch.float32)
        for u in client_uploads:
            w = u["num_examples"] / total_ex
            agg_flat[u["coords"]] += u["vals"] * w
        agg = apply_update_to_state_dict(agg_flat, layer_meta, global_state_before)
        stats["payload_bytes"] = sum(u["coords"].numel() * 6 for u in client_uploads)
        return agg, stats

    if method == "sketchopa":
        # 安全聚合：unmask → 求和 → 恢复
        if args.secure and shprg is not None:
            # 收集 masked ct 和 seed，求 s_sum
            cts = [u["ct_zq"] for u in client_uploads]
            seeds = [u["seed_vec"] for u in client_uploads]
            # 2-client seed 交换
            agg_share = exchange.auxiliary_aggregate(seeds[0], exchange.client_pack_share(seeds[1]))
            s_sum = exchange.server_reconstruct([agg_share])
            # 聚合 ct
            agg_ct = sum(cts) % SHAMIR_PRIME
            # unmask
            agg_zq = shprg.unmask_sketch_zq(agg_ct, s_sum)
            agg_sketch = zq_to_float(agg_zq, args.quant_scale, SHAMIR_PRIME)
        else:
            # 明文 sketch 聚合（也在 Z_q 域，需 centered 还原）
            agg_zq = sum(u["ct_zq"] for u in client_uploads) % SHAMIR_PRIME
            agg_sketch = zq_to_float(agg_zq, args.quant_scale, SHAMIR_PRIME)

        # active_union 候选坐标
        candidates = torch.cat([u["coords"] for u in client_uploads])
        num_active = len(client_uploads)
        k_rec = compute_k_recover(k, num_active, args.recover_rho, d)

        coords_r, vals_r, rec_stats = recover_topk(
            agg_sketch, encoder, layout, candidates, k_rec, clip_norm=args.clip_norm,
        )
        stats.update(rec_stats)
        stats["k_recover"] = k_rec
        stats["sketch_len"] = layout.sketch_len
        stats["payload_bytes"] = layout.payload_bytes * len(client_uploads)

        # 构建稠密更新并加到全局模型
        agg_flat = build_sparse_update(coords_r, vals_r, d)
        agg = apply_update_to_state_dict(agg_flat, layer_meta, global_state_before)
        return agg, stats

    raise ValueError(f"unknown method: {method}")


def _sample_seed_vec(base_seed: int, round_idx: int, cid: int) -> List[int]:
    """确定性采样 seed_vec。"""
    import hashlib
    h = hashlib.sha256(f"seed|{base_seed}|r{round_idx}|c{cid}".encode()).digest()
    return [int.from_bytes(h[i*2:i*2+2], "little") % SHAMIR_PRIME for i in range(4)]


def _load_state_to_model(model, state_dict: Dict[str, torch.Tensor], accelerator) -> None:
    """加载全局 state 到模型。"""
    target_dtype = next(model.parameters()).dtype
    sd = {k: v.to(target_dtype) for k, v in state_dict.items()}
    model.load_state_dict(sd, strict=True)


if __name__ == "__main__":
    main()
