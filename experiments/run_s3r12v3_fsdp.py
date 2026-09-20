"""S3R12v3 FSDP 客户端（ICC1/ICC2）。

用法（在仓库根目录，8 卡）::

    accelerate launch --config_file configs/accelerate_config.yaml \\
      experiments/run_s3r12v3_fsdp.py \\
      --client-id 0 \\
      --server-url http://192.168.235.42:8080 \\
      --minio-endpoint http://192.168.235.42:9000 \\
      --model-path model/Qwen/Qwen2.5-0.5B \\
      --data-path data/splits/icc1_client0_train.json

通信：
- 上传：仅本轮选中 blocks（~ratio），精度由 --transfer-dtype 控制
- 下载：优先 cache / global_delta；冷启动默认可跳过 round-0（本地 model-path）
- 在线 eval：本地训练后算 eval_loss 并上报服务端
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# SEC-5：TLS 证书验证开关（自签名证书时设为 False）
_REQUESTS_VERIFY: bool = True


def _set_requests_verify(verify: bool) -> None:
    global _REQUESTS_VERIFY
    _REQUESTS_VERIFY = bool(verify)
    if not _REQUESTS_VERIFY:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from shared.block_selection import (  # noqa: E402
    add_block_delta,
    build_group_blocks,
    encode_block_delta,
    flatten_group_blocks,
    merge_quant_residual_memory,
    resolve_transfer_dtype,
    selected_from_flat,
    update_block_memory,
    update_block_memory_from_states,
)
from shared.block_crypto import (  # noqa: E402
    derive_round_key,
    encode_sec_block_payload,
    pad_slice,
)
from shared.minio_client import MinIOClient  # noqa: E402
from shared.protocol import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BUCKET,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LOCAL_STEPS,
    DEFAULT_LR,
    DEFAULT_MEMORY_DECAY,
    DEFAULT_QUANT_RESIDUAL_DECAY,
    DEFAULT_SEQ_LEN,
    DEFAULT_TRANSFER_DTYPE,
    RoundPlan,
    agg_block_done_key,
    agg_block_key,
    epoch_seed_from_plan,
    global_delta_key,
    global_state_key,
    selected_from_jsonable,
    upload_block_key,
    upload_blocks_key,
)
from shared.run_config import apply_to_args, load_run_config  # noqa: E402
from shared.state_dict_utils import (  # noqa: E402
    add_state,
    broadcast_object,
    get_full_state_fsdp,
    load_full_state_fsdp,
    sub_state,
    zero_state_like,
)

logger = logging.getLogger("s3r12v3_fsdp_client")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_chat_texts(rows: List[Dict[str, Any]], tokenizer) -> List[str]:
    texts = []
    for r in rows:
        messages = [
            {"role": "system", "content": r.get("instruction", "")},
            {"role": "user", "content": r.get("input", "")},
            {"role": "assistant", "content": r.get("output", "")},
        ]
        texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )
    return texts


def patch_qwen2_attn_fp32_qk() -> None:
    """V100 无 bf16：fp16 QK matmul 会 inf。Softmax 虽已 fp32，但 AMP 仍把 matmul 打回 fp16。

    只把 QK（contract dim = head_dim）升到 fp32，权重仍 fp16，避免 7B 全 fp32 OOM。
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    if getattr(Qwen2Attention.forward, "_fedscale_fp32_qk", False):
        return
    orig_fwd = Qwen2Attention.forward

    def forward(self, *args, **kwargs):
        orig_matmul = torch.matmul
        head_dim = int(getattr(self, "head_dim", 0) or 0)

        def matmul_qk_fp32(a, b):
            if (
                head_dim
                and torch.is_tensor(a)
                and torch.is_tensor(b)
                and a.dim() == 4
                and b.dim() == 4
                and a.shape[-1] == head_dim
            ):
                with torch.autocast(device_type="cuda", enabled=False):
                    return orig_matmul(a.float(), b.float())
            return orig_matmul(a, b)

        torch.matmul = matmul_qk_fp32
        try:
            return orig_fwd(self, *args, **kwargs)
        finally:
            torch.matmul = orig_matmul

    forward._fedscale_fp32_qk = True
    Qwen2Attention.forward = forward


def causal_lm_loss_fp32(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted CE in fp32. fp16 logits × Qwen vocab(~152k) 在 7B+ 上会 overflow 成 nan。"""
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def model_step_loss(outputs, batch) -> torch.Tensor:
    labels = batch.get("labels") if isinstance(batch, dict) else None
    if labels is not None and getattr(outputs, "logits", None) is not None:
        return causal_lm_loss_fp32(outputs.logits, labels)
    return outputs.loss


def load_causal_lm(model_path: str):
    # 7B 全 fp32 在 8×V100 上 ~29GiB/卡 OOM（embed/lm_head 未切分）。
    # 权重保持 fp16 AMP；QK matmul 单独 fp32（并关掉 autocast）。
    patch_qwen2_attn_fp32_qk()
    param_dtype = torch.float16
    kwargs = dict(
        torch_dtype=param_dtype,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="eager",
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config._fedscale_attn_impl = "eager"
    model.config._fedscale_attn_qk = "fp32"
    model.config._fedscale_param_dtype = str(param_dtype).replace("torch.", "")
    return model


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], tokenizer, seq_len: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def wait_json(url: str, timeout_s: float = 3600.0, interval_s: float = 2.0, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=30, headers=headers, verify=_REQUESTS_VERIFY)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(interval_s)
    raise RuntimeError(f"timeout waiting for {url}: {last_err}")


def _broadcast_flag(accelerator, flag: int) -> int:
    """短 collective：广播 int 状态码。0=继续, 1=成功, 2=失败。"""
    device = accelerator.device
    t = torch.tensor([int(flag)], device=device, dtype=torch.int64)
    if accelerator.num_processes > 1:
        import torch.distributed as dist

        dist.broadcast(t, src=0)
    return int(t.item())


def run_rank0_io_with_heartbeat(
    accelerator,
    work_fn,
    *,
    interval_s: float = 2.0,
    timeout_s: float = 3600.0,
    label: str = "io",
):
    """在 rank0 上跑可能很长的网络 I/O，其它 rank 用短心跳同步，避免长 NCCL 等待。

    work_fn() 仅在 main process 调用，返回任意结果；失败应抛异常。
    返回值会 ``broadcast_object`` 到所有 rank——不要把完整 state_dict 放进返回值，
    否则每个进程都会反序列化一份全量副本（3B/7B 会把主机内存打满）。
    """
    import threading

    box: Dict[str, Any] = {"done": False, "ok": False, "value": None, "error": None}

    if accelerator.is_main_process:

        def _target() -> None:
            try:
                box["value"] = work_fn()
                box["ok"] = True
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc
                box["ok"] = False
            finally:
                box["done"] = True

        th = threading.Thread(target=_target, name=f"rank0-{label}", daemon=True)
        th.start()
    else:
        th = None

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        flag = 0
        if accelerator.is_main_process:
            if box["done"]:
                flag = 1 if box["ok"] else 2
        flag = _broadcast_flag(accelerator, flag)
        if flag == 1:
            err_or_val = broadcast_object(box.get("value") if accelerator.is_main_process else None, src=0)
            return err_or_val
        if flag == 2:
            err = broadcast_object(box.get("error") if accelerator.is_main_process else None, src=0)
            if isinstance(err, Exception):
                raise RuntimeError(f"{label} failed: {err}") from err
            raise RuntimeError(f"{label} failed: {err}")
        time.sleep(interval_s)

    raise TimeoutError(f"{label} timed out after {timeout_s}s")


def wait_aggregate_with_heartbeat(
    accelerator,
    *,
    server: str,
    round_idx: int,
    poll_interval: float,
    timeout_s: float = 3600.0,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """全 ranks 一起短轮询等待聚合结果，避免 rank0 单线程卡 NCCL。"""
    deadline = time.time() + timeout_s
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        flag = 0
        if accelerator.is_main_process:
            try:
                resp = requests.get(f"{server}/api/round/{round_idx}/result", timeout=30, headers=headers, verify=_REQUESTS_VERIFY)
                resp.raise_for_status()
                last = resp.json()
                if last.get("done"):
                    flag = 1
                elif last.get("message") == "aggregation_failed":
                    flag = 2
            except Exception as exc:  # noqa: BLE001
                logger.warning("poll result failed: %s", exc)
                flag = 0
        flag = _broadcast_flag(accelerator, flag)
        if flag == 1:
            return broadcast_object(last if accelerator.is_main_process else None, src=0)
        if flag == 2:
            detail = broadcast_object(last if accelerator.is_main_process else None, src=0)
            raise RuntimeError(f"aggregation failed: {detail}")
        time.sleep(poll_interval)
    raise TimeoutError(f"wait aggregate round {round_idx} timed out after {timeout_s}s")


def train_local_steps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator,
    local_steps: int,
    grad_accum: int,
    detailed_metrics: bool = False,
) -> tuple[float, dict]:
    """Run local training and return (avg_loss, step_metrics).

    detailed_metrics=False (default): fast path for normal runs — no torch.profiler,
    no per-step nvidia-smi, no cuda.synchronize timing fences.

    detailed_metrics=True: dashboard instrumentation (forward/backward/NCCL/GPU util).
    Only enable for dedicated profiling runs; it can inflate train_local_s ~8x.
    """
    model.train()
    step = 0
    loss_sum = 0.0
    loss_count = 0
    total_tokens_processed = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    step_records: list[dict] = []
    train_start = time.monotonic()
    gpu_mem_base = 0.0
    if torch.cuda.is_available():
        gpu_mem_base = torch.cuda.memory_allocated() / (1024 * 1024)

    _gpu_util_samples: list[float] = []
    _cpu_util_samples: list[float] = []
    _gpu_mem_samples: list[float] = []

    def _sample_gpu_util() -> float | None:
        try:
            import subprocess as _sp
            _r = _sp.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            lines = _r.stdout.strip().splitlines()
            if lines:
                vals = lines[0].split(",")
                return round(float(vals[0].strip()), 2)
        except Exception:
            pass
        return None

    def _sample_gpu_mem() -> float | None:
        if torch.cuda.is_available():
            try:
                return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            except Exception:
                pass
        return None

    def _sample_cpu_util() -> float | None:
        try:
            import psutil as _ps
            return round(_ps.cpu_percent(interval=None), 2)
        except Exception:
            return None

    from collections import defaultdict as _dd
    _nccl_stats: dict = _dd(lambda: {"count": 0, "total_us": 0, "bytes": 0})
    _fsdp_param_count = 0
    _est_ar_bytes = 0
    _est_ag_bytes = 0
    _est_rs_bytes = 0
    _prof = None

    if detailed_metrics:
        _sample_cpu_util()
        try:
            _fsdp_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception:
            pass
        _world_size = getattr(accelerator, "num_processes", 1) or 1
        _dtype_bytes = 2  # fp16
        _est_ag_bytes = _fsdp_param_count * _dtype_bytes
        _est_rs_bytes = _fsdp_param_count * _dtype_bytes // _world_size

        def _make_profiler():
            try:
                from torch.profiler import profile, ProfilerActivity
                return profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_stack=False,
                    with_modules=False,
                )
            except Exception:
                return None

        _prof = _make_profiler()
    else:
        def _make_profiler():
            return None

    while step < local_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        if "input_ids" in batch:
            total_tokens_processed += int(batch["input_ids"].numel())
        step_start = time.monotonic()
        if _prof is not None:
            _prof.start()
        with accelerator.accumulate(model):
            if detailed_metrics and torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_start = time.monotonic()
            outputs = model(**batch)
            loss = model_step_loss(outputs, batch)
            if detailed_metrics and torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_ms = (time.monotonic() - fwd_start) * 1000

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    step += 1
                    if accelerator.is_main_process and step % 10 == 0:
                        logger.warning("local step %s/%s loss=non-finite (skipped)", step, local_steps)
                continue

            bwd_start = time.monotonic()
            accelerator.backward(loss)
            if detailed_metrics and torch.cuda.is_available():
                torch.cuda.synchronize()
            bwd_ms = (time.monotonic() - bwd_start) * 1000

            loss_sum += float(loss.detach().item())
            loss_count += 1
            if accelerator.sync_gradients:
                if _prof is not None:
                    _prof.stop()
                    try:
                        events = _prof.key_averages()
                        for evt in events:
                            key = evt.key
                            key_l = key.lower()
                            if (
                                "nccl" in key_l
                                or "all_reduce" in key_l
                                or "all_gather" in key_l
                                or "reduce_scatter" in key_l
                            ):
                                cat = (
                                    "all_reduce" if "all_reduce" in key_l else
                                    "all_gather" if "all_gather" in key_l else
                                    "reduce_scatter" if "reduce_scatter" in key_l else
                                    "other_nccl"
                                )
                                _nccl_stats[cat]["count"] += evt.count
                                _nccl_stats[cat]["total_us"] += (
                                    evt.self_device_time_total
                                    if hasattr(evt, "self_device_time_total") else 0
                                )
                        _prof = _make_profiler()
                    except Exception:
                        _prof = _make_profiler()
                opt_start = time.monotonic()
                try:
                    accelerator.clip_grad_norm_(1.0)
                except TypeError:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if detailed_metrics and torch.cuda.is_available():
                    torch.cuda.synchronize()
                opt_ms = (time.monotonic() - opt_start) * 1000
                step += 1
                total_ms = (time.monotonic() - step_start) * 1000
                if detailed_metrics:
                    ar_ms = (
                        round(
                            _nccl_stats["all_reduce"]["total_us"] / 1000.0
                            / max(_nccl_stats["all_reduce"]["count"], 1),
                            2,
                        )
                        if _nccl_stats["all_reduce"]["count"] > 0 else 0.0
                    )
                    ag_ms = (
                        round(
                            _nccl_stats["all_gather"]["total_us"] / 1000.0
                            / max(_nccl_stats["all_gather"]["count"], 1),
                            2,
                        )
                        if _nccl_stats["all_gather"]["count"] > 0 else 0.0
                    )
                    rs_ms = (
                        round(
                            _nccl_stats["reduce_scatter"]["total_us"] / 1000.0
                            / max(_nccl_stats["reduce_scatter"]["count"], 1),
                            2,
                        )
                        if _nccl_stats["reduce_scatter"]["count"] > 0 else 0.0
                    )
                    _gpu_u = _sample_gpu_util()
                    _cpu_u = _sample_cpu_util()
                    _gpu_m = _sample_gpu_mem()
                    if _gpu_u is not None:
                        _gpu_util_samples.append(_gpu_u)
                    if _cpu_u is not None:
                        _cpu_util_samples.append(_cpu_u)
                    if _gpu_m is not None:
                        _gpu_mem_samples.append(_gpu_m)
                    step_records.append({
                        "step": step,
                        "forward_ms": round(fwd_ms, 2),
                        "backward_ms": round(bwd_ms, 2),
                        "optimizer_ms": round(opt_ms, 2),
                        "comm_ms": round(max(0.0, total_ms - fwd_ms - bwd_ms - opt_ms), 2),
                        "total_ms": round(total_ms, 2),
                        "loss": round(loss_sum / max(loss_count, 1), 6),
                        "all_reduce_ms": ar_ms,
                        "all_gather_ms": ag_ms,
                        "reduce_scatter_ms": rs_ms,
                        "all_reduce_bytes": _est_ar_bytes,
                        "all_gather_bytes": _est_ag_bytes,
                        "reduce_scatter_bytes": _est_rs_bytes,
                        "gpu_util_pct": _gpu_u,
                        "gpu_mem_mb": _gpu_m,
                        "cpu_util_pct": _cpu_u,
                    })
                if accelerator.is_main_process and (step == 1 or step % 10 == 0):
                    logger.info(
                        "local step %s/%s loss=%.4f",
                        step,
                        local_steps,
                        loss_sum / max(loss_count, 1),
                    )
            else:
                if _prof is not None:
                    _prof.stop()

    train_time_s = time.monotonic() - train_start
    gpu_mem_peak_mb = 0.0
    if torch.cuda.is_available():
        gpu_mem_peak_mb = max(
            gpu_mem_base,
            torch.cuda.max_memory_allocated() / (1024 * 1024),
        )

    gpu_util_pct = None
    if _gpu_util_samples:
        gpu_util_pct = round(sum(_gpu_util_samples) / len(_gpu_util_samples), 2)
    elif detailed_metrics and torch.cuda.is_available():
        try:
            gpu_util_pct = round(torch.cuda.utilization(), 2)
        except Exception:
            pass
        if gpu_util_pct is None:
            try:
                import subprocess as _sp
                _r = _sp.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                gpu_util_pct = round(float(_r.stdout.strip().splitlines()[0]), 2)
            except Exception:
                gpu_util_pct = None

    cpu_mem_peak_mb = 0.0
    cpu_util_pct = 0.0
    try:
        import resource as _resource
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        cpu_mem_peak_mb = round(rss / 1024.0, 2) if rss > 0 else 0.0
    except Exception:
        pass
    if _cpu_util_samples:
        cpu_util_pct = round(sum(_cpu_util_samples) / len(_cpu_util_samples), 2)
    elif detailed_metrics:
        try:
            import psutil
            cpu_util_pct = round(psutil.cpu_percent(interval=0.1), 2)
        except Exception:
            pass

    gpu_mem_avg_mb = None
    if _gpu_mem_samples:
        gpu_mem_avg_mb = round(sum(_gpu_mem_samples) / len(_gpu_mem_samples), 2)

    avg_loss = loss_sum / max(loss_count, 1)
    n = len(step_records)
    total_ar_ms = sum(s.get("all_reduce_ms", 0) for s in step_records)
    total_ag_ms = sum(s.get("all_gather_ms", 0) for s in step_records)
    total_rs_ms = sum(s.get("reduce_scatter_ms", 0) for s in step_records)
    total_ar_bytes = sum(s.get("all_reduce_bytes", 0) for s in step_records)
    total_ag_bytes = sum(s.get("all_gather_bytes", 0) for s in step_records)
    total_rs_bytes = sum(s.get("reduce_scatter_bytes", 0) for s in step_records)
    throughput_tokens_per_s = (
        round(total_tokens_processed / train_time_s, 2) if train_time_s > 0 else 0.0
    )
    net_rx = None
    net_tx = None
    net_total = None
    if detailed_metrics:
        try:
            import psutil as _ps
            _net = _ps.net_io_counters()
            net_rx = _net.bytes_recv
            net_tx = _net.bytes_sent
            net_total = net_rx + net_tx
        except Exception:
            pass
    summary = {
        "steps": step_records,
        "total_train_time_s": round(train_time_s, 2),
        "avg_step_time_ms": round(sum(s["total_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_forward_ms": round(sum(s["forward_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_backward_ms": round(sum(s["backward_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_comm_ms": round(sum(s["comm_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_optimizer_ms": round(sum(s["optimizer_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_all_reduce_ms": round(total_ar_ms / n, 2) if n else 0.0,
        "avg_all_gather_ms": round(total_ag_ms / n, 2) if n else 0.0,
        "avg_reduce_scatter_ms": round(total_rs_ms / n, 2) if n else 0.0,
        "total_all_reduce_bytes": total_ar_bytes,
        "total_all_gather_bytes": total_ag_bytes,
        "total_reduce_scatter_bytes": total_rs_bytes,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "total_tokens": total_tokens_processed,
        "num_steps": n if n else step,
        "detailed_metrics": detailed_metrics,
    }
    resources = {
        "gpu_memory_peak_mb": round(gpu_mem_peak_mb, 2),
        "gpu_utilization_avg_pct": gpu_util_pct,
        "cpu_utilization_avg_pct": cpu_util_pct,
        "cpu_memory_peak_mb": cpu_mem_peak_mb,
        "gpu_memory_avg_mb": gpu_mem_avg_mb,
        "gpu_util_samples": _gpu_util_samples,
        "cpu_util_samples": _cpu_util_samples,
        "gpu_mem_samples": _gpu_mem_samples,
        "network_rx_bytes": net_rx,
        "network_tx_bytes": net_tx,
        "network_total_bytes": net_total,
        "total_nccl_bytes": total_ar_bytes + total_ag_bytes + total_rs_bytes,
        "nccl_collective_calls": sum(
            1 for s in step_records
            if s.get("all_reduce_ms", 0) > 0
            or s.get("all_gather_ms", 0) > 0
            or s.get("reduce_scatter_ms", 0) > 0
        ),
        "avg_nccl_comm_ms": round((total_ar_ms + total_ag_ms + total_rs_ms) / n, 2) if n else 0.0,
    }
    return avg_loss, {"training": summary, "resources": resources}


@torch.no_grad()
def eval_local_batches(
    model: torch.nn.Module,
    dataloader: DataLoader,
    accelerator,
    max_batches: int = 0,
) -> float:
    """在线 eval：FSDP 下各 rank 共同前向，返回全局平均 loss。"""
    import torch.distributed as dist

    model.eval()
    total = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    count = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    for i, batch in enumerate(dataloader):
        if max_batches > 0 and i >= max_batches:
            break
        outputs = model(**batch)
        bs = int(batch["input_ids"].size(0))
        step_loss = model_step_loss(outputs, batch)
        total += step_loss.detach().double() * bs
        count += float(bs)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    denom = float(count.item()) if float(count.item()) > 0 else 1.0
    return float(total.item()) / denom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3R12v3 FSDP federated client")
    p.add_argument("--config", default="", help="run 配置 YAML（CLI 覆盖 yaml；CFG-1）")
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--server-url", required=True)
    p.add_argument("--minio-endpoint", required=True)
    p.add_argument("--minio-access-key", default="fedscale")
    p.add_argument("--minio-secret-key", default="fedscale-minio-2026")
    p.add_argument("--minio-bucket", default=DEFAULT_BUCKET)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--num-examples", type=int, default=0, help="覆盖上报样本数；默认用数据集长度")
    p.add_argument("--local-steps", type=int, default=DEFAULT_LOCAL_STEPS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--memory-decay", type=float, default=DEFAULT_MEMORY_DECAY)
    p.add_argument(
        "--quant-residual-decay",
        type=float,
        default=DEFAULT_QUANT_RESIDUAL_DECAY,
        help="INT16 量化残差衰减；1.0=完整保留（不要和 block memory_decay 混用）",
    )
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
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
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument(
        "--transfer-dtype",
        default=DEFAULT_TRANSFER_DTYPE,
        help="通信落盘精度: auto(跟随模型)/fp16/fp32/bf16；int8 预留",
    )
    p.add_argument(
        "--skip-round0-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="首轮用本地 model-path 作为 round-0，跳过全量下载（默认开）",
    )
    p.add_argument(
        "--online-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每轮本地训练后在线 eval 并上报（默认开）",
    )
    p.add_argument(
        "--eval-path",
        default="data/medical_flashcards_eval.json",
        help="在线 eval 数据集路径",
    )
    p.add_argument(
        "--eval-max-batches",
        type=int,
        default=0,
        help="在线 eval 最多 batch 数；0=全集",
    )
    p.add_argument(
        "--eval-every-n-rounds",
        type=int,
        default=1,
        help="eval 频率：1=每轮(默认)；5=每5轮一次(省~10s/轮)；round 1 始终 eval",
    )
    # SEC-0：控制面 token
    p.add_argument(
        "--auth-token",
        default="",
        help="控制面共享 token；非空时所有 /api/** 请求带 Authorization: Bearer <token>",
    )
    # SEC-1/2/3：上传隐私（gidx 化 + 加密 + 末 block 填充）
    p.add_argument(
        "--sec-upload-privacy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="SEC-1/2/3: 启用上传隐私（payload 不含 key_name, gidx 加密, 末 block 填充）",
    )
    # SEC-5：TLS（自签名证书时跳过验证）
    p.add_argument(
        "--tls-no-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="SEC-5: 跳过 TLS 证书验证（自签名证书时用）；生产环境应改用 CA 签名证书",
    )
    # B: SecAgg (Windowed Secure Aggregation v2)
    p.add_argument("--secagg-enabled", action=argparse.BooleanOptionalAction, default=False,
                   help="B: 启用 Windowed SecAgg（pairwise+self mask）")
    p.add_argument("--secagg-modulus-bits", type=int, default=16,
                   help="B: 模数位宽 8/16/24/32")
    p.add_argument("--secagg-scale", type=float, default=0.0,
                   help="B: 定点量化 scale；0=per-window 公开聚合；>0=固定全局 scale")
    p.add_argument("--secagg-stochastic-rounding", action=argparse.BooleanOptionalAction, default=False,
                   help="B: 随机舍入")
    p.add_argument("--secagg-q-min", type=int, default=0,
                   help="B: 最小成功参与者数；0=num_clients")
    p.add_argument("--secagg-hadamard", action=argparse.BooleanOptionalAction, default=False,
                   help="B: 量化前 Hadamard 旋转（压低动态范围）")
    # RES-2：client memory 持久化目录
    p.add_argument(
        "--client-state-dir",
        default="",
        help="客户端持久化目录（memory + 本地全局版本）；空=只在内存（旧行为）",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="从 --client-state-dir 恢复 memory / local_global（RES-2）",
    )
    p.add_argument(
        "--detailed-train-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="训练细粒度观测（torch.profiler / 每步 nvidia-smi / cuda.synchronize）；"
             "默认关。仅专项 profiling 时开启，否则 train_local_s 会膨胀约 8x",
    )
    return p.parse_args()


def _auth_headers(token: str) -> Dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    # CFG-1：yaml 覆盖 argparse 默认值，CLI 显式参数再覆盖 yaml
    cfg = load_run_config(args.config)
    apply_to_args(
        args, cfg,
        skip=("client_id", "server_url", "minio_endpoint", "minio_access_key",
              "minio_secret_key", "minio_bucket", "model_path", "data_path",
              "num_examples", "config", "client_state_dir", "resume", "poll_interval"),
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # SEC-5：TLS 证书验证（自签名证书时跳过）
    _set_requests_verify(not args.tls_no_verify)

    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    rank = accelerator.process_index
    is_main = accelerator.is_main_process

    # SEC-0：控制面 token（非空时所有 /api/** 带 Bearer）
    auth_headers = _auth_headers(args.auth_token)

    torch.manual_seed(args.seed + args.client_id * 17 + rank)

    if is_main:
        logger.info(
            "Client %s starting on %s ranks, model=%s data=%s",
            args.client_id,
            accelerator.num_processes,
            args.model_path,
            args.data_path,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="right", legacy=False, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm(args.model_path)
    model.config.use_cache = False
    if is_main:
        logger.info(
            "model attn=%s attn_qk=%s param_dtype=%s mixed_precision=%s",
            getattr(model.config, "_fedscale_attn_impl", None),
            getattr(model.config, "_fedscale_attn_qk", None),
            getattr(model.config, "_fedscale_param_dtype", None),
            accelerator.mixed_precision,
        )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    rows = load_json(Path(args.data_path))
    texts = to_chat_texts(rows, tokenizer)
    dataset = ChatDataset(texts, tokenizer, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * args.local_steps)),
        num_training_steps=args.local_steps,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    eval_loader = None
    if args.online_eval:
        eval_file = Path(args.eval_path)
        if not eval_file.is_absolute():
            eval_file = REPO_ROOT / eval_file
        if eval_file.exists():
            eval_rows = load_json(eval_file)
            eval_texts = to_chat_texts(eval_rows, tokenizer)
            eval_ds = ChatDataset(eval_texts, tokenizer, args.seq_len)
            eval_loader = DataLoader(
                eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
            )
            eval_loader = accelerator.prepare(eval_loader)
            if is_main:
                logger.info(
                    "Online eval enabled path=%s examples=%s max_batches=%s",
                    eval_file,
                    len(eval_ds),
                    args.eval_max_batches,
                )
        elif is_main:
            logger.warning("Online eval requested but missing %s; disabled", eval_file)

    minio = None
    memory = None
    quant_residual_mem: Dict[str, Any] = {}
    # 本地缓存的全局 state：version=k 表示等于 MinIO global_state/round-k
    local_global: Optional[Dict[str, Any]] = None
    local_version: Optional[int] = None
    transfer_dtype = None  # 首轮拿到 global_state 后按 --transfer-dtype 解析
    # RES-2：客户端状态持久化目录
    client_state_dir: Optional[Path] = None
    if is_main and args.client_state_dir:
        client_state_dir = Path(args.client_state_dir)
        client_state_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        minio = MinIOClient(
            endpoint=args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            bucket=args.minio_bucket,
        )
        logger.info(
            "transfer-dtype arg=%s skip_round0_download=%s online_eval=%s",
            args.transfer_dtype,
            args.skip_round0_download,
            args.online_eval and eval_loader is not None,
        )
        # RES-2：恢复 memory + local_global
        if args.resume and client_state_dir is not None and (client_state_dir / "memory.pt").exists():
            try:
                mem = torch.load(client_state_dir / "memory.pt", map_location="cpu", weights_only=False)
                memory = mem
                logger.info("RES-2: restored client memory from %s", client_state_dir / "memory.pt")
            except Exception as exc:  # noqa: BLE001
                logger.warning("RES-2: failed to restore memory: %s", exc)
        if args.resume and client_state_dir is not None and (client_state_dir / "quant_residual.pt").exists():
            try:
                quant_residual_mem = torch.load(
                    client_state_dir / "quant_residual.pt", map_location="cpu", weights_only=False,
                ) or {}
                logger.info("RES-2: restored quant residual from %s", client_state_dir / "quant_residual.pt")
            except Exception as exc:  # noqa: BLE001
                logger.warning("RES-2: failed to restore quant residual: %s", exc)

    # 首轮：用本地基地模型作 round-0，避免再下 942MiB
    # 注意：FSDP FULL_STATE_DICT 是集合通信，必须所有 rank 同步调用，不能丢进 rank0 线程
    if args.skip_round0_download:
        snap = get_full_state_fsdp(model)
        accelerator.wait_for_everyone()
        if is_main and isinstance(snap, dict) and snap:
            local_global = snap
            local_version = 0
            logger.info(
                "Seeded local_global from local model as version=0 (skip round-0 MinIO download)"
            )


    server = args.server_url.rstrip("/")
    num_examples = args.num_examples or len(dataset)

    while True:
        t_round0 = time.monotonic()
        # 拉 current/plan 也可能慢：用 heartbeat，避免非 rank0 卡在长 barrier
        status = run_rank0_io_with_heartbeat(
            accelerator,
            lambda: wait_json(f"{server}/api/round/current", interval_s=args.poll_interval, headers=auth_headers),
            interval_s=args.poll_interval,
            timeout_s=3600.0,
            label="wait-current",
        )
        if status.get("status") == "finished":
            if is_main:
                logger.info("Server finished all rounds")
            break
        round_idx = int(status["round"])
        if is_main:
            logger.info("=== Round %s status=%s ===", round_idx, status.get("status"))

        plan_raw = run_rank0_io_with_heartbeat(
            accelerator,
            lambda: wait_json(f"{server}/api/round/{round_idx}/plan", interval_s=args.poll_interval, headers=auth_headers),
            interval_s=args.poll_interval,
            timeout_s=3600.0,
            label=f"wait-plan-{round_idx}",
        )
        plan = RoundPlan.from_dict(plan_raw)
        selected = selected_from_jsonable(plan.selected_by_key)
        # 流式 pipeline：plan 带 block_list 时用 per-block 上传+下载
        use_pipeline = bool(plan.block_list) and str(
            getattr(args, "compressor", "public_random")
        ) != "dense"

        # OPS-3：Client 选择。未选中则跳过训练，但仍对齐 delta 并等待聚合结果
        selected_client_ids = plan_raw.get("selected_client_ids")
        am_selected = (
            selected_client_ids is None
            or int(args.client_id) in set(int(c) for c in selected_client_ids)
        )
        if not am_selected and is_main:
            logger.info(
                "OPS-3: client %s NOT selected for round %s (selected=%s); skip training, stay aligned",
                args.client_id,
                round_idx,
                selected_client_ids,
            )

        # 同步到 round-(N-1) 全局状态：优先增量 delta，失败/断档回退全量
        t_download = 0.0
        download_bytes = 0.0
        download_mode = "unknown"

        def _sync_global_base():
            """返回 {state, bytes, mode, version}，version == round_idx-1。"""
            nonlocal t_download, download_bytes, download_mode, local_global, local_version
            assert minio is not None
            target = round_idx - 1
            t0 = time.monotonic()
            nbytes = 0.0
            mode = "full"

            if local_global is not None and local_version == target:
                mode = "cache" if target > 0 else "local_base"
                logger.info("Reusing cached global_state version=%s mode=%s", target, mode)
            elif (
                local_global is not None
                and local_version is not None
                and local_version == target - 1
                and target >= 1
                and minio.exists(global_delta_key(target))
            ):
                dkey = global_delta_key(target)
                logger.info("Downloading incremental %s (local_version=%s -> %s)", dkey, local_version, target)
                payload, nbytes = minio.get_torch_with_size(dkey, map_location="cpu")
                if not isinstance(payload, dict) or "block_delta" not in payload:
                    raise RuntimeError(f"invalid global delta payload: {dkey}")
                add_block_delta(local_global, payload["block_delta"])
                local_version = target
                mode = "delta"
                logger.info(
                    "Applied delta %s size=%.2f MiB (now version=%s)",
                    dkey,
                    nbytes / (1024 * 1024),
                    local_version,
                )
            else:
                # 冷启动、断档或 delta 缺失：拉全量
                fkey = global_state_key(target)
                logger.info(
                    "Downloading full %s (local_version=%s)",
                    fkey,
                    local_version,
                )
                local_global, nbytes = minio.get_torch_with_size(fkey, map_location="cpu")
                local_version = target
                mode = "full"
                logger.info(
                    "Downloaded full %s size=%.2f MiB",
                    fkey,
                    nbytes / (1024 * 1024),
                )

            t_download = time.monotonic() - t0
            download_bytes = float(nbytes)
            download_mode = mode
            # 只广播 meta。完整 state 留在 rank0 的 local_global，避免 8 份 pickle 副本。
            return {
                "bytes": float(nbytes),
                "mode": mode,
                "version": int(local_version) if local_version is not None else -1,
            }

        dl = run_rank0_io_with_heartbeat(
            accelerator,
            _sync_global_base if is_main else (lambda: None),
            interval_s=args.poll_interval,
            timeout_s=3600.0,
            label=f"sync-global-{round_idx}",
        )
        download_bytes = float(dl.get("bytes", 0.0)) if isinstance(dl, dict) else 0.0
        download_mode = str(dl.get("mode", "unknown")) if isinstance(dl, dict) else "unknown"
        global_state = local_global if is_main else None
        if is_main and memory is None:
            memory = zero_state_like(global_state)
        if is_main and transfer_dtype is None and global_state is not None:
            transfer_dtype = resolve_transfer_dtype(args.transfer_dtype, ref_state=global_state)
            logger.info(
                "Resolved transfer_dtype=%s (arg=%s, model_ref=%s)",
                transfer_dtype,
                args.transfer_dtype,
                next((str(v.dtype) for v in global_state.values() if hasattr(v, "is_floating_point") and v.is_floating_point()), "?"),
            )

        t_load0 = time.monotonic()
        # local_base：GPU 分片已是 round-0，不必再灌一遍。
        # 其它模式：完整 CPU state 只在 rank0，按 FSDP unit scatter，禁止 8 份整模。
        if download_mode == "local_base":
            t_broadcast_load = 0.0
            if is_main:
                logger.info("Skip FSDP reload: local_base already matches round-0 shards")
            accelerator.wait_for_everyone()
        else:
            load_full_state_fsdp(model, global_state if is_main else None)
            accelerator.wait_for_everyone()
            t_broadcast_load = time.monotonic() - t_load0

        # A-5: 客户端独立验证 plan（mask_hash + layout_hash）
        # 防止 server 在看到本轮更新后篡改 mask（spec 第 3 节 Independence from current private updates）
        if is_main and plan.mask_hash and plan.mask_policy_id:
            from shared.block_selection import recompute_selected_blocks, build_group_blocks, build_permutations, build_selected_blocks
            from shared.canonical_encoding import compute_mask_hash, compute_layout_hash, MASK_POLICY_ID
            try:
                # 独立重算 selected blocks
                always_on_keys = set(plan.always_on_keys) if plan.always_on_keys else None
                recomputed = recompute_selected_blocks(
                    reference_state=global_state,
                    round_idx=round_idx,
                    coverage_h=plan.coverage_h,
                    seed=plan.seed,
                    block_size=args.block_size,
                    always_on_keys=always_on_keys,
                    always_on_threshold=0 if always_on_keys else 4096,
                )
                recomputed_hash = compute_mask_hash(recomputed)
                if recomputed_hash != plan.mask_hash:
                    logger.error(
                        "MASK VERIFICATION FAILED: recomputed mask_hash=%s != plan mask_hash=%s; "
                        "refusing to participate in round %s",
                        recomputed_hash[:16], plan.mask_hash[:16], round_idx,
                    )
                    raise RuntimeError(f"mask_hash mismatch: server plan may be tampered")
                # 验证 layout_hash（只验证一次，因为 layout 不变）
                if plan.layout_hash:
                    group_blocks_l, always_on_l = build_group_blocks(
                        global_state, block_size=args.block_size,
                        always_on_threshold=4096,
                    )
                    layout_hash_l = compute_layout_hash(
                        group_blocks_l, always_on_keys=always_on_l,
                        coverage_h=plan.coverage_h, block_size=args.block_size,
                    )
                    if layout_hash_l != plan.layout_hash:
                        logger.error(
                            "LAYOUT VERIFICATION FAILED: recomputed=%s != plan=%s",
                            layout_hash_l[:16], plan.layout_hash[:16],
                        )
                        raise RuntimeError("layout_hash mismatch: model layout differs from server")
                logger.info(
                    "Plan verified: mask_hash=%s layout_hash=%s (round %s)",
                    plan.mask_hash[:16], plan.layout_hash[:16], round_idx,
                )
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning("Plan verification skipped due to error: %s", e)

        if not am_selected:
            # OPS-3：未选中——不训练/不上传，只等聚合后对齐 delta
            if is_main:
                logger.info("OPS-3: skipping train/upload for round %s", round_idx)
            t_wait0 = time.monotonic()
            result = wait_aggregate_with_heartbeat(
                accelerator,
                server=server,
                round_idx=round_idx,
                poll_interval=args.poll_interval,
                timeout_s=3600.0,
                headers=auth_headers,
            )
            t_wait = time.monotonic() - t_wait0
            # 对齐 delta（与下方 _apply_post_aggregate_delta 同逻辑）
            def _skip_apply_delta():
                nonlocal local_global, local_version
                assert minio is not None
                if local_global is None or local_version != round_idx - 1:
                    return
                dkey = global_delta_key(round_idx)
                if not minio.exists(dkey):
                    return
                payload, _ = minio.get_torch_with_size(dkey, map_location="cpu")
                add_block_delta(local_global, payload["block_delta"])
                local_version = round_idx
            run_rank0_io_with_heartbeat(
                accelerator,
                _skip_apply_delta if is_main else (lambda: None),
                interval_s=args.poll_interval,
                timeout_s=3600.0,
                label=f"skip-apply-delta-{round_idx}",
            )
            accelerator.wait_for_everyone()
            continue

        t_train0 = time.monotonic()
        train_loss, train_step_metrics = train_local_steps(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            local_steps=args.local_steps,
            grad_accum=args.grad_accum,
            detailed_metrics=bool(getattr(args, "detailed_train_metrics", False)),
        )
        accelerator.wait_for_everyone()
        t_train = time.monotonic() - t_train0

        # 在线 eval（本地训练后），全 ranks 参与 FSDP 前向
        # 优化：eval_every_n_rounds 控制频率，默认每轮，可设 5 表示每 5 轮 eval 一次（省 ~10s/轮）
        do_eval_this_round = (
            eval_loader is not None
            and (args.eval_every_n_rounds <= 0 or round_idx % args.eval_every_n_rounds == 0 or round_idx == 1)
        )
        eval_loss_val: Optional[float] = None
        t_eval = 0.0
        if do_eval_this_round:
            t_eval0 = time.monotonic()
            eval_loss_val = eval_local_batches(
                model, eval_loader, accelerator, max_batches=args.eval_max_batches
            )
            t_eval = time.monotonic() - t_eval0
            if is_main:
                logger.info("Online eval_loss=%.6f in %.1fs", eval_loss_val, t_eval)
        accelerator.wait_for_everyone()

        # FSDP state export timing
        t_state_export0 = time.monotonic()
        full_state = get_full_state_fsdp(model)
        t_state_export = time.monotonic() - t_state_export0
        if is_main:
            logger.info("FSDP state export: %.3fs", t_state_export)
        mem_box: List[Any] = [memory]
        quant_mem_box: List[Any] = [quant_residual_mem]

        def _pipeline_upload_and_apply():
            """流式 per-block pipeline：逐 block 上传，聚合好的立即下载 apply。"""
            nonlocal local_global, local_version
            assert minio is not None and global_state is not None and mem_box[0] is not None and full_state is not None
            t_enc0 = time.monotonic()
            delta = sub_state(full_state, global_state)
            to_send = add_state(delta, mem_box[0])
            assert transfer_dtype is not None
            compressor = str(getattr(args, "compressor", "public_random") or "public_random")
            local_selected = selected
            if compressor == "dense":
                groups, _ = build_group_blocks(to_send, block_size=args.block_size)
                flat = flatten_group_blocks(groups)
                local_selected = selected_from_flat(flat, range(len(flat)))
            block_delta = encode_block_delta(to_send, local_selected, dtype=transfer_dtype)
            mem_box[0] = update_block_memory(to_send, local_selected, args.memory_decay)
            t_encode = time.monotonic() - t_enc0

            # 逐 block 上传并通知 server
            logger.info("Pipeline: uploading %s blocks (ratio=%.2f%%)", plan.n_selected_blocks, plan.upload_ratio * 100)
            t_up0 = time.monotonic()
            total_upload_bytes = 0
            applied_blocks = set()
            # SEC-2：派生 per-round 密钥（与 server 一致）
            round_key = (
                derive_round_key(epoch_seed_from_plan(plan.seed, plan.epoch), round_idx)
                if args.sec_upload_privacy else b""
            )
            for binfo in plan.block_list:
                gidx, key_name, start, end = binfo[0], binfo[1], binfo[2], binfo[3]
                # 取出该 block 的 delta slice
                single_block_delta = {}
                if key_name in block_delta:
                    for s, e, slice_data in block_delta[key_name]:
                        if s == start and e == end:
                            single_block_delta[key_name] = [(s, e, slice_data)]
                            break
                if args.sec_upload_privacy:
                    # SEC-1/2/3：gidx 化 + 加密 + 末 block 填充
                    slice_data = single_block_delta.get(key_name, [(0, 0, torch.zeros(1, dtype=transfer_dtype) if transfer_dtype else torch.zeros(1))])[0][2]
                    is_int8_item = (slice_data.dtype == torch.int8) if hasattr(slice_data, "dtype") else False
                    scale = None
                    if is_int8_item:
                        # int8 时 slice 在元组第 3 位，scale 在第 4 位
                        for it in block_delta.get(key_name, []):
                            if it[0] == start and it[1] == end and len(it) > 3:
                                scale = float(it[3]); break
                    real_len = int(end - start)
                    block_payload = encode_sec_block_payload(
                        gidx, slice_data, round_key,
                        real_len=real_len, is_int8=is_int8_item, scale=scale,
                        block_size=args.block_size,
                        num_examples=int(num_examples),
                        train_loss=float(train_loss),
                        eval_loss=float(eval_loss_val) if eval_loss_val is not None else None,
                    )
                else:
                    block_payload = {
                        "block_delta": single_block_delta,
                        "num_examples": int(num_examples),
                        "train_loss": float(train_loss),
                        "eval_loss": float(eval_loss_val) if eval_loss_val is not None else None,
                        "client_id": int(args.client_id),
                        "round": int(round_idx),
                        "block_idx": int(gidx),
                    }
                bkey = upload_block_key(round_idx, args.client_id, gidx)
                total_upload_bytes += minio.put_torch(bkey, block_payload)
                # 通知 server 该 block 已上传
                resp = requests.post(
                    f"{server}/api/round/{round_idx}/client/{args.client_id}/block/{gidx}/uploaded",
                    json={
                        "num_examples": int(num_examples),
                        "train_loss": float(train_loss),
                        "eval_loss": float(eval_loss_val) if eval_loss_val is not None else None,
                        "block_energies": [],
                    },
                    timeout=30,
                    headers=auth_headers,
                    verify=_REQUESTS_VERIFY,
                )
                resp.raise_for_status()
                rj = resp.json()
                if rj.get("aggregated"):
                    # server 已聚合该 block，立即下载 apply（与后续 block 上传并行）
                    agg_payload = minio.get_torch(agg_block_key(round_idx, gidx), map_location="cpu")
                    add_block_delta(local_global, agg_payload["block_delta"])
                    applied_blocks.add(int(gidx))
                    logger.info("Pipeline: applied block %s immediately", gidx)
            t_upload = time.monotonic() - t_up0

            # 所有 block 上传完，等剩余未聚合的 block 完成
            t_wait0 = time.monotonic()
            # 轮询 block-status，下载尚未 apply 的 block
            deadline = time.time() + 3600.0
            while time.time() < deadline:
                resp = requests.get(
                    f"{server}/api/round/{round_idx}/block-status",
                    timeout=30, headers=auth_headers, verify=_REQUESTS_VERIFY,
                )
                resp.raise_for_status()
                status = resp.json()
                if status.get("done"):
                    break
                time.sleep(args.poll_interval)
            t_wait = time.monotonic() - t_wait0

            # 下载所有尚未 apply 的聚合 block
            t_post0 = time.monotonic()
            post_bytes = 0
            for binfo in plan.block_list:
                gidx = int(binfo[0])
                if gidx in applied_blocks:
                    continue
                agg_key = agg_block_key(round_idx, gidx)
                if minio.exists(agg_key):
                    payload, nbytes = minio.get_torch_with_size(agg_key, map_location="cpu")
                    add_block_delta(local_global, payload["block_delta"])
                    applied_blocks.add(gidx)
                    post_bytes += nbytes
            local_version = round_idx
            t_post = time.monotonic() - t_post0
            logger.info("Pipeline: all blocks done, applied remaining in %.2fs", t_post)

            mode_code = {"cache": 0.0, "delta": 1.0, "full": 2.0, "local_base": 3.0}.get(download_mode, 2.0)
            timings = {
                "download_mode": mode_code,
                "download_global_s": round(t_download, 3),
                "download_global_MiB": round(download_bytes / (1024 * 1024), 3),
                "broadcast_load_s": round(t_broadcast_load, 3),
                "train_local_s": round(t_train, 3),
                "eval_local_s": round(t_eval, 3),
                "encode_delta_s": round(t_encode, 3),
                "upload_minio_s": round(t_upload, 3),
                "upload_blocks_MiB": round(total_upload_bytes / (1024 * 1024), 3),
                "pipeline_wait_agg_s": round(t_wait, 3),
                "pipeline_post_apply_s": round(t_post, 3),
                # 聚合后下行（pipeline 路径在此记账；勿依赖外层 post_delta 覆盖）
                "post_delta_s": round(t_post, 3),
                "post_delta_MiB": round(post_bytes / (1024 * 1024), 3),
                # Evaluation-compatible WAN timing aliases
                "wan_download_s": round(t_download, 3),
                "wan_upload_s": round(t_upload, 3),
                "model_delta_bytes": int(total_upload_bytes),
            }
            return timings

        def _batch_upload_and_notify():
            """批量模式（回退）：打包成一个 blocks.pt 上传。"""
            assert minio is not None and global_state is not None and mem_box[0] is not None and full_state is not None
            t_enc0 = time.monotonic()
            delta = sub_state(full_state, global_state)
            to_send = add_state(delta, mem_box[0])
            assert transfer_dtype is not None
            compressor = str(getattr(args, "compressor", "public_random") or "public_random")
            local_selected = selected
            if compressor == "dense":
                groups, _ = build_group_blocks(to_send, block_size=args.block_size)
                flat = flatten_group_blocks(groups)
                local_selected = selected_from_flat(flat, range(len(flat)))
            block_delta = encode_block_delta(to_send, local_selected, dtype=transfer_dtype)
            mem_box[0] = update_block_memory(to_send, local_selected, args.memory_decay)
            t_encode = time.monotonic() - t_enc0
            if args.sec_upload_privacy:
                # SEC-1/2/3：批量模式也用 SEC 编码（每个 block 一个 SEC payload，打包成 list）
                round_key = derive_round_key(epoch_seed_from_plan(plan.seed, plan.epoch), round_idx)
                sec_blocks = []
                for binfo in plan.block_list:
                    gidx, key_name, start, end = binfo[0], binfo[1], binfo[2], binfo[3]
                    slice_data = None
                    scale = None
                    if key_name in block_delta:
                        for s, e, sd in block_delta[key_name]:
                            if s == start and e == end:
                                slice_data = sd; break
                    if slice_data is None:
                        slice_data = torch.zeros(end - start, dtype=transfer_dtype or torch.float16)
                    is_int8_item = (slice_data.dtype == torch.int8) if hasattr(slice_data, "dtype") else False
                    if is_int8_item:
                        for it in block_delta.get(key_name, []):
                            if it[0] == start and it[1] == end and len(it) > 3:
                                scale = float(it[3]); break
                    sec_blocks.append(encode_sec_block_payload(
                        gidx, slice_data, round_key,
                        real_len=int(end - start), is_int8=is_int8_item, scale=scale,
                        block_size=args.block_size,
                        num_examples=int(num_examples),
                        train_loss=float(train_loss),
                        eval_loss=float(eval_loss_val) if eval_loss_val is not None else None,
                    ))
                payload = {
                    "v": 1,  # SEC-1/2/3 批量格式标记
                    "sec_blocks": sec_blocks,
                    "num_examples": int(num_examples),
                    "train_loss": float(train_loss),
                    "eval_loss": float(eval_loss_val) if eval_loss_val is not None else None,
                    "client_id": int(args.client_id),
                    "round": int(round_idx),
                }
            else:
                payload = {
                    "block_delta": block_delta,
                    "num_examples": int(num_examples),
                    "train_loss": float(train_loss),
                    "eval_loss": float(eval_loss_val) if eval_loss_val is not None else None,
                    "client_id": int(args.client_id),
                    "round": int(round_idx),
                }
            up_key = upload_blocks_key(round_idx, args.client_id)
            logger.info(
                "Uploading %s blocks=%s ratio=%.2f%% train_loss=%.4f eval_loss=%s",
                up_key, plan.n_selected_blocks, plan.upload_ratio * 100, train_loss,
                f"{eval_loss_val:.4f}" if eval_loss_val is not None else "skip",
            )
            t_up0 = time.monotonic()
            upload_bytes = minio.put_torch(up_key, payload)
            t_upload = time.monotonic() - t_up0
            mode_code = {"cache": 0.0, "delta": 1.0, "full": 2.0, "local_base": 3.0}.get(download_mode, 2.0)
            timings = {
                "download_mode": mode_code,
                "download_global_s": round(t_download, 3),
                "download_global_MiB": round(download_bytes / (1024 * 1024), 3),
                "broadcast_load_s": round(t_broadcast_load, 3),
                "train_local_s": round(t_train, 3),
                "eval_local_s": round(t_eval, 3),
                "encode_delta_s": round(t_encode, 3),
                "upload_minio_s": round(t_upload, 3),
                "upload_blocks_MiB": round(upload_bytes / (1024 * 1024), 3),
                # Evaluation-compatible WAN timing aliases
                "wan_download_s": round(t_download, 3),
                "wan_upload_s": round(t_upload, 3),
                "model_delta_bytes": int(upload_bytes),
            }
            t_notify0 = time.monotonic()
            body = {"num_examples": int(num_examples), "train_loss": float(train_loss), "timings": timings}
            if eval_loss_val is not None:
                body["eval_loss"] = float(eval_loss_val)
            resp = requests.post(
                f"{server}/api/round/{round_idx}/client/{args.client_id}/upload-complete",
                json=body, timeout=60, headers=auth_headers, verify=_REQUESTS_VERIFY,
            )
            resp.raise_for_status()
            timings["notify_server_s"] = round(time.monotonic() - t_notify0, 3)
            return timings

        def _secagg_upload_and_apply():
            """B-7b: SecAgg pipeline — 公开聚合 scale → 量化 → DH → mask → 上传 z_k → 提交 self_master → apply。"""
            nonlocal local_global, local_version
            assert minio is not None and global_state is not None and mem_box[0] is not None and full_state is not None
            from shared.secagg_client import (
                SecAggClient,
                extract_window_to_send_fp32,
                prepare_window_quant_cache,
                window_amax_payload,
            )
            from shared.fixed_point import pack_zq, compute_q_max
            from shared.protocol import SecAggPlan, build_window_descriptors, upload_secagg_blob_key
            from shared.secagg_blob import pack_masked_windows_blob

            t_extract0 = time.monotonic()
            assert transfer_dtype is not None
            windows = build_window_descriptors(plan.block_list)
            # 量化前全程 FP32：x = delta + block_memory + quant_residual
            delta_slices = extract_window_to_send_fp32(
                full_state, global_state, mem_box[0], quant_mem_box[0] or {}, windows,
            )
            t_extract = time.monotonic() - t_extract0

            _slice_max = max((float(v.abs().max().item()) for v in delta_slices.values()), default=0.0)
            _fs_max = max(float(v.abs().max().item()) for v in full_state.values() if hasattr(v, "abs"))
            _gs_max = max(float(v.abs().max().item()) for v in global_state.values() if hasattr(v, "abs"))
            _fs_dtype = str(next(iter(full_state.values())).dtype) if full_state else "?"
            _gs_dtype = str(next(iter(global_state.values())).dtype) if global_state else "?"
            logger.info(
                "SecAgg to_send range: max|x|=%.4f (fp32) max|full_state|=%.4f(%s) max|global_state|=%.4f(%s)",
                _slice_max, _fs_max, _fs_dtype, _gs_max, _gs_dtype,
            )

            modulus_bits = int(getattr(args, "secagg_modulus_bits", 16))
            q = 1 << modulus_bits
            q_max = compute_q_max(modulus_bits, n_clients=2)
            fixed_scale = float(getattr(args, "secagg_scale", 0.0))
            use_per_window = fixed_scale <= 0.0
            fallback_scale = fixed_scale if fixed_scale > 0.0 else (2.0 ** -20)
            stochastic = bool(getattr(args, "secagg_stochastic_rounding", False))
            hadamard = bool(getattr(args, "secagg_hadamard", False))
            quant_decay = float(getattr(args, "quant_residual_decay", DEFAULT_QUANT_RESIDUAL_DECAY))

            secagg_plan = SecAggPlan(
                q_min=int(getattr(args, "secagg_q_min", 0)) or 2,
                quantization_scale=fallback_scale,
                modulus_bits=modulus_bits,
                modulus_q=q,
                q_max=q_max,
                stochastic_rounding=stochastic,
                hadamard_enabled=hadamard,
                hadamard_seed=round_idx,
            )

            secagg_client = SecAggClient(args.client_id, secagg_plan, windows)

            # Phase 1: 提交公钥。Hadamard 只报一个 global_amax，避免 per-window L∞ 泄露。
            t_amax0 = time.monotonic()
            quant_cache: Dict[int, Any] = {}
            announce_body = {"pk_hex": secagg_client.get_public_key_hex()}
            if hadamard:
                g_amax, quant_cache = prepare_window_quant_cache(delta_slices, windows, secagg_plan)
                announce_body["global_amax"] = g_amax
                logger.info("SecAgg: Hadamard global_amax=%e (no per-window L∞)", g_amax)
            else:
                announce_body["window_amax"] = window_amax_payload(
                    delta_slices, windows=windows, plan=secagg_plan,
                )
            t_amax = time.monotonic() - t_amax0
            t_dh0 = time.monotonic()
            resp = requests.post(
                f"{server}/api/round/{round_idx}/client/{args.client_id}/secagg/key-announce",
                json=announce_body,
                timeout=30, headers=auth_headers, verify=_REQUESTS_VERIFY,
            )
            resp.raise_for_status()

            # 等待所有 peer 公钥；window_scales 由 server 收齐 amax 后下发
            peer_keys = {}
            deadline_dh = time.time() + 300.0
            while time.time() < deadline_dh:
                resp = requests.get(
                    f"{server}/api/round/{round_idx}/secagg/peer-keys",
                    timeout=30, headers=auth_headers, verify=_REQUESTS_VERIFY,
                )
                resp.raise_for_status()
                pk_status = resp.json()
                if pk_status.get("status") == "ready":
                    peer_keys = pk_status["public_keys"]
                    try:
                        secagg_client.plan.apply_session_from_server(pk_status)
                    except ValueError as exc:
                        raise RuntimeError(str(exc)) from exc
                    window_scales = pk_status.get("window_scales") or {}
                    if window_scales:
                        secagg_client.plan.window_scales = {
                            str(k): float(v) for k, v in window_scales.items()
                        }
                    global_scale = pk_status.get("global_scale")
                    if global_scale is not None:
                        secagg_client.plan.quantization_scale = float(global_scale)
                    break
                time.sleep(args.poll_interval)
            else:
                raise RuntimeError("SecAgg: timeout waiting for peer keys")

            secagg_client.setup_dh(peer_keys)
            t_dh = time.monotonic() - t_dh0
            logger.info(
                "SecAgg: DH setup complete in %.2fs (hadamard=%s global_scale=%s per_window_scale=%s session=%s)",
                t_dh, hadamard,
                f"{secagg_client.plan.quantization_scale:e}" if hadamard else "n/a",
                use_per_window and not hadamard,
                (secagg_client.plan.secagg_session_id or "")[:16],
            )

            # Phase 3: 量化 + mask → 一个 blob PUT → 一次 HTTP 通知
            logger.info("SecAgg: encoding %s masked windows into one blob", len(windows))
            t_comp0 = time.monotonic()
            blob_parts = []
            quant_residual = {}
            clip_fracs = []
            zero_fracs = []
            scale_vals = []
            rel_l2_vals = []
            cosine_vals = []
            sqnr_vals = []

            for window in windows:
                wid = window.window_id
                delta_slice = delta_slices.get(wid)
                if delta_slice is None:
                    delta_slice = torch.zeros(window.vector_length, dtype=torch.float32)
                cached = quant_cache.get(wid)
                extra = {}
                if cached is not None:
                    extra["quant_input"] = cached[0]
                    extra["signs"] = cached[1]
                z_k, residual, stats = secagg_client.mask_window(
                    window, delta_slice, return_feedback=True, **extra,
                )
                z_len = int(z_k.numel())
                quant_residual.setdefault(window.key_name, []).append(
                    (window.start, window.end, residual.detach().to(dtype=torch.float32).cpu())
                )
                clip_fracs.append(float(stats["clip_frac"]))
                zero_fracs.append(float(stats["zero_frac"]))
                scale_vals.append(float(stats["scale"]))
                rel_l2_vals.append(float(stats.get("rel_l2", 0.0)))
                cosine_vals.append(float(stats.get("cosine", 1.0)))
                sqnr = float(stats.get("sqnr_db", 0.0))
                if sqnr == sqnr and abs(sqnr) != float("inf"):
                    sqnr_vals.append(sqnr)
                z_bytes = pack_zq(z_k, modulus_bits)
                blob_parts.append((wid, z_len, z_bytes))
                del z_k

            blob = pack_masked_windows_blob(blob_parts)
            t_compute = time.monotonic() - t_comp0

            t_put0 = time.monotonic()
            total_upload_bytes = len(blob)
            z_key = upload_secagg_blob_key(round_idx, args.client_id)
            minio.put_bytes(z_key, blob)
            del blob
            t_put = time.monotonic() - t_put0

            t_notify0 = time.monotonic()
            _tl = float(train_loss)
            if not (_tl == _tl and abs(_tl) != float("inf")):
                logger.warning("train_loss non-finite; omitting numeric train_loss")
                _tl = 0.0  # schema 仍要 float；0 只表示无效，日志已警告
            _el = None
            if eval_loss_val is not None:
                _el = float(eval_loss_val)
                if not (_el == _el and abs(_el) != float("inf")):
                    logger.warning("eval_loss non-finite; omitting from notify")
                    _el = None
            body = {
                "z_key": z_key,
                "window_ids": [p[0] for p in blob_parts],
                "num_examples": int(num_examples),
                "train_loss": _tl,
            }
            if _el is not None:
                body["eval_loss"] = _el
            resp = requests.post(
                f"{server}/api/round/{round_idx}/client/{args.client_id}/secagg/masked-blob",
                json=body, timeout=120, headers=auth_headers, verify=_REQUESTS_VERIFY,
            )
            resp.raise_for_status()
            t_notify = time.monotonic() - t_notify0

            mem_box[0] = update_block_memory_from_states(
                full_state, global_state, mem_box[0], selected, args.memory_decay,
            )
            quant_mem_box[0] = merge_quant_residual_memory(
                quant_mem_box[0] or {}, selected, quant_residual, quant_decay,
            )
            t_encode = t_extract + t_amax + t_compute
            t_upload = t_put + t_notify
            n_clip_windows = sum(1 for c in clip_fracs if c > 0.0)
            mean_clip = (sum(clip_fracs) / len(clip_fracs)) if clip_fracs else 0.0
            max_clip = max(clip_fracs) if clip_fracs else 0.0
            mean_zero = (sum(zero_fracs) / len(zero_fracs)) if zero_fracs else 0.0
            mean_rel_l2 = (sum(rel_l2_vals) / len(rel_l2_vals)) if rel_l2_vals else 0.0
            mean_cos = (sum(cosine_vals) / len(cosine_vals)) if cosine_vals else 1.0
            mean_sqnr = (sum(sqnr_vals) / len(sqnr_vals)) if sqnr_vals else 0.0
            logger.info(
                "SecAgg: blob put+notify %.2fs compute %.2fs (%.1f MiB raw) clip_windows=%s/%s mean_clip=%.4g max_clip=%.4g mean_zero=%.4g "
                "rel_l2=%.4g cosine=%.6f sqnr=%.2fdB scale_min=%e scale_max=%e hadamard=%s quant_decay=%.3f",
                t_upload, t_compute, total_upload_bytes / (1024 * 1024),
                n_clip_windows, len(windows), mean_clip, max_clip, mean_zero,
                mean_rel_l2, mean_cos, mean_sqnr,
                min(scale_vals) if scale_vals else 0.0,
                max(scale_vals) if scale_vals else 0.0,
                hadamard, quant_decay,
            )
            if max_clip > 0.0:
                logger.warning(
                    "SecAgg: clip_frac>0 (max=%.4g) — per-window scale 仍偏小，大更新被截断",
                    max_clip,
                )

            # Phase 4: 提交 self_master
            t_sm0 = time.monotonic()
            resp = requests.post(
                f"{server}/api/round/{round_idx}/client/{args.client_id}/secagg/self-master",
                json={"sm_hex": secagg_client.get_self_master_hex()},
                timeout=180, headers=auth_headers, verify=_REQUESTS_VERIFY,
            )
            resp.raise_for_status()
            t_sm = time.monotonic() - t_sm0
            logger.info("SecAgg: self_master submitted in %.2fs", t_sm)

            # 等待 server 聚合完成
            t_wait0 = time.monotonic()
            deadline = time.time() + 3600.0
            while time.time() < deadline:
                resp = requests.get(
                    f"{server}/api/round/{round_idx}/block-status",
                    timeout=30, headers=auth_headers, verify=_REQUESTS_VERIFY,
                )
                resp.raise_for_status()
                status = resp.json()
                if status.get("done"):
                    break
                time.sleep(args.poll_interval)
            t_wait = time.monotonic() - t_wait0

            # 下载聚合 delta 并 apply（一份 global_delta，不再按 window GET）
            t_post0 = time.monotonic()
            post_bytes = 0
            applied_blocks = set()
            try:
                payload, nbytes = minio.get_torch_with_size(
                    global_delta_key(round_idx), map_location="cpu",
                )
                add_block_delta(local_global, payload["block_delta"])
                post_bytes = nbytes
                applied_blocks.add(-1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SecAgg: global_delta missing (%s), fallback per-block", exc)
                for binfo in plan.block_list:
                    gidx = int(binfo[0])
                    agg_key = agg_block_key(round_idx, gidx)
                    if minio.exists(agg_key):
                        payload, nbytes = minio.get_torch_with_size(agg_key, map_location="cpu")
                        add_block_delta(local_global, payload["block_delta"])
                        applied_blocks.add(gidx)
                        post_bytes += nbytes
            local_version = round_idx
            t_post = time.monotonic() - t_post0
            logger.info("SecAgg: applied aggregated delta in %.2fs (%s keys)", t_post, len(applied_blocks))

            mode_code = {"cache": 0.0, "delta": 1.0, "full": 2.0, "local_base": 3.0}.get(download_mode, 2.0)
            timings = {
                "download_mode": mode_code,
                "download_global_s": round(t_download, 3),
                "download_global_MiB": round(download_bytes / (1024 * 1024), 3),
                "broadcast_load_s": round(t_broadcast_load, 3),
                "train_local_s": round(t_train, 3),
                "eval_local_s": round(t_eval, 3),
                "encode_delta_s": round(t_encode, 3),
                "secagg_extract_s": round(t_extract, 3),
                "secagg_amax_s": round(t_amax, 3),
                "secagg_compute_s": round(t_compute, 3),
                "secagg_dh_s": round(t_dh, 3),
                "secagg_put_s": round(t_put, 3),
                "secagg_notify_s": round(t_notify, 3),
                "upload_minio_s": round(t_upload, 3),
                "upload_blocks_MiB": round(total_upload_bytes / (1024 * 1024), 3),
                "secagg_self_master_s": round(t_sm, 3),
                "pipeline_wait_agg_s": round(t_wait, 3),
                "pipeline_post_apply_s": round(t_post, 3),
                # SecAgg 在上传路径内已 apply 聚合 block；必须在此记账，
                # 否则外层因 local_version 已推进而 post_delta_MiB=0，图上 ICC recv 全 0。
                "post_delta_s": round(t_post, 3),
                "post_delta_MiB": round(post_bytes / (1024 * 1024), 3),
            }
            return timings

        # 执行上传（SecAgg / pipeline / batch）
        if getattr(args, "secagg_enabled", False) and is_main:
            upload_fn = _secagg_upload_and_apply
        elif use_pipeline:
            upload_fn = _pipeline_upload_and_apply
        else:
            upload_fn = _batch_upload_and_notify
        pre_wait_timings = run_rank0_io_with_heartbeat(
            accelerator,
            upload_fn if is_main else (lambda: {}),
            interval_s=args.poll_interval,
            timeout_s=3600.0,
            label=f"upload-{round_idx}",
        )
        memory = mem_box[0]
        quant_residual_mem = quant_mem_box[0]

        # RES-2：每轮把 memory + local_version 持久化到本地
        if is_main and client_state_dir is not None and memory is not None:
            try:
                torch.save(memory, client_state_dir / "memory.pt")
                torch.save(quant_residual_mem or {}, client_state_dir / "quant_residual.pt")
                (client_state_dir / "local_version.txt").write_text(
                    str(local_version if local_version is not None else -1)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("RES-2: failed to persist memory: %s", exc)

        if use_pipeline:
            # pipeline 模式：block 已在上传过程中逐个 apply，只需等 server 标记 round 完成
            t_wait0 = time.monotonic()
            result = wait_aggregate_with_heartbeat(
                accelerator, server=server, round_idx=round_idx,
                poll_interval=args.poll_interval, timeout_s=3600.0, headers=auth_headers,
            )
            t_wait = time.monotonic() - t_wait0
            post = {"applied": 1.0, "bytes": 0.0, "seconds": 0.0}
        else:
            t_wait0 = time.monotonic()
            result = wait_aggregate_with_heartbeat(
                accelerator, server=server, round_idx=round_idx,
                poll_interval=args.poll_interval, timeout_s=3600.0, headers=auth_headers,
            )
            t_wait = time.monotonic() - t_wait0

            def _apply_post_aggregate_delta():
                nonlocal local_global, local_version
                assert minio is not None
                if local_global is None or local_version != round_idx - 1:
                    return {"applied": 0.0, "bytes": 0.0, "seconds": 0.0}
                dkey = global_delta_key(round_idx)
                if not minio.exists(dkey):
                    logger.warning("Missing %s after aggregate; next round may full-download", dkey)
                    return {"applied": 0.0, "bytes": 0.0, "seconds": 0.0}
                t_d0 = time.monotonic()
                payload, nbytes = minio.get_torch_with_size(dkey, map_location="cpu")
                add_block_delta(local_global, payload["block_delta"])
                local_version = round_idx
                dt = time.monotonic() - t_d0
                logger.info("Post-aggregate applied %s size=%.2f MiB in %.2fs (now version=%s)",
                            dkey, nbytes / (1024 * 1024), dt, local_version)
                return {"applied": 1.0, "bytes": float(nbytes), "seconds": float(dt)}

            post = run_rank0_io_with_heartbeat(
                accelerator,
                _apply_post_aggregate_delta if is_main else (lambda: {"applied": 0.0, "bytes": 0.0, "seconds": 0.0}),
                interval_s=args.poll_interval, timeout_s=3600.0,
                label=f"apply-delta-{round_idx}",
            )

        if is_main:
            full_timings = {
                **(pre_wait_timings or {}),
                "wait_aggregate_s": round(t_wait, 3),
                "round_total_s": round(time.monotonic() - t_round0, 3),
            }
            # SecAgg/pipeline 已在 upload 路径内 apply 并写入 post_delta_*；
            # 外层因 local_version 已是 round_idx 会得到 bytes=0，不能覆盖。
            if "post_delta_MiB" not in full_timings:
                full_timings["post_delta_s"] = round(float((post or {}).get("seconds", 0.0)), 3)
                full_timings["post_delta_MiB"] = round(
                    float((post or {}).get("bytes", 0.0)) / (1024 * 1024), 3
                )

            # Write metrics_detailed.json for evaluation package compatibility
            try:
                from datetime import datetime as _dt
                metrics_detailed = {
                    "timestamp": _dt.now().isoformat(),
                    "training": train_step_metrics.get("training", {}),
                    "resources": train_step_metrics.get("resources", {}),
                    "federated": {
                        "t_total_round_s": full_timings.get("round_total_s"),
                        "t_model_delta_export_s": full_timings.get("encode_delta_s"),
                        "t_full_update_compression_s": full_timings.get("encode_delta_s"),
                        "t_state_export_s": round(t_state_export, 3) if 't_state_export' in dir() else None,
                        "full_state_export_s": round(t_state_export, 3) if 't_state_export' in dir() else None,
                        "wan_download_s": full_timings.get("wan_download_s"),
                        "wan_upload_s": full_timings.get("wan_upload_s"),
                        "model_delta_bytes": full_timings.get("model_delta_bytes"),
                        "training_only_s": full_timings.get("train_local_s"),
                        "evaluation_s": full_timings.get("eval_local_s", 0.0),
                    },
                }
                metrics_dir = Path(args.client_state_dir or ".") / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                md_path = metrics_dir / f"metrics_detailed_round_{round_idx}.json"
                md_path.write_text(
                    json.dumps(metrics_detailed, indent=2), encoding="utf-8"
                )
                # Also overwrite the latest version
                (metrics_dir / "metrics_detailed.json").write_text(
                    json.dumps(metrics_detailed, indent=2), encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to write metrics_detailed.json: %s", exc)

            logger.info(
                "Round %s done mode=%s timing_s=%s server_timing=%s",
                round_idx,
                download_mode,
                full_timings,
                result.get("timing_s"),
            )
            try:
                treq = requests.post(
                    f"{server}/api/round/{round_idx}/client/{args.client_id}/timing",
                    json={"timings": {k: float(v) for k, v in full_timings.items() if isinstance(v, (int, float))}},
                    timeout=30,
                    headers=auth_headers,
                    verify=_REQUESTS_VERIFY,
                )
                treq.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to report client timing: %s", exc)

        accelerator.wait_for_everyone()
    if is_main:
        logger.info("Client %s done", args.client_id)


if __name__ == "__main__":
    main()
