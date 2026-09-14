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
    resolve_transfer_dtype,
    update_block_memory,
)
from shared.block_crypto import (  # noqa: E402
    derive_round_key,
    encode_sec_block_payload,
    pad_slice,
)
from shared.block_vote import (  # noqa: E402
    block_energies,
    flatten_group_blocks,
    select_topk_indices,
    selected_from_flat,
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
    get_sharded_block_delta,
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
) -> tuple[float, dict]:
    """Run local training and return (avg_loss, step_metrics).

    step_metrics contains per-step timing and resource data that the
    evaluation package can aggregate (category 1 + 2).
    """
    model.train()
    step = 0
    micro = 0
    loss_sum = 0.0
    loss_count = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    step_records: list[dict] = []
    train_start = time.monotonic()
    gpu_mem_base = 0.0
    if torch.cuda.is_available():
        gpu_mem_base = torch.cuda.memory_allocated() / (1024 * 1024)

    while step < local_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        step_start = time.monotonic()
        with accelerator.accumulate(model):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_start = time.monotonic()
            outputs = model(**batch)
            loss = outputs.loss
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_ms = (time.monotonic() - fwd_start) * 1000

            bwd_start = time.monotonic()
            accelerator.backward(loss)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            bwd_ms = (time.monotonic() - bwd_start) * 1000

            loss_sum += float(loss.detach().item())
            loss_count += 1
            if accelerator.sync_gradients:
                opt_start = time.monotonic()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                opt_ms = (time.monotonic() - opt_start) * 1000
                step += 1
                total_ms = (time.monotonic() - step_start) * 1000
                step_records.append({
                    "step": step,
                    "forward_ms": round(fwd_ms, 2),
                    "backward_ms": round(bwd_ms, 2),
                    "optimizer_ms": round(opt_ms, 2),
                    "comm_ms": round(max(0.0, total_ms - fwd_ms - bwd_ms - opt_ms), 2),
                    "total_ms": round(total_ms, 2),
                    "loss": round(loss_sum / max(loss_count, 1), 6),
                })
                if accelerator.is_main_process and step % 10 == 0:
                    logger.info(
                        "local step %s/%s loss=%.4f",
                        step,
                        local_steps,
                        loss_sum / max(loss_count, 1),
                    )
            else:
                micro += 1

    train_time_s = time.monotonic() - train_start
    gpu_mem_peak_mb = 0.0
    if torch.cuda.is_available():
        gpu_mem_peak_mb = max(
            gpu_mem_base,
            torch.cuda.max_memory_allocated() / (1024 * 1024),
        )

    # Sample GPU utilization at the end of training (instantaneous snapshot)
    gpu_util_pct = None
    if torch.cuda.is_available():
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

    # CPU memory (RSS) via resource module
    cpu_mem_peak_mb = 0.0
    cpu_util_pct = 0.0
    try:
        import resource as _resource
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB, macOS: bytes
        cpu_mem_peak_mb = round(rss / 1024.0, 2) if rss > 0 else 0.0
    except Exception:
        pass
    try:
        import psutil
        cpu_util_pct = round(psutil.cpu_percent(interval=0.1), 2)
    except Exception:
        pass

    avg_loss = loss_sum / max(loss_count, 1)
    n = len(step_records)
    summary = {
        "steps": step_records,
        "total_train_time_s": round(train_time_s, 2),
        "avg_step_time_ms": round(sum(s["total_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_forward_ms": round(sum(s["forward_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_backward_ms": round(sum(s["backward_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_comm_ms": round(sum(s["comm_ms"] for s in step_records) / n, 2) if n else 0.0,
        "avg_optimizer_ms": round(sum(s["optimizer_ms"] for s in step_records) / n, 2) if n else 0.0,
        "throughput_tokens_per_s": 0.0,
        "total_tokens": 0,
        "num_steps": n,
    }
    resources = {
        "gpu_memory_peak_mb": round(gpu_mem_peak_mb, 2),
        "gpu_utilization_avg_pct": gpu_util_pct,
        "cpu_utilization_avg_pct": cpu_util_pct,
        "cpu_memory_peak_mb": cpu_mem_peak_mb,
        "network_rx_bytes": None,
        "network_tx_bytes": None,
        "network_total_bytes": None,
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
        total += outputs.loss.detach().double() * bs
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
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument(
        "--compressor",
        default="public_random",
        choices=["public_random", "block_vote_lag", "block_topk", "dense"],
    )
    p.add_argument("--rho", type=float, default=0.0, help="vote/topk 比例；0 表示跟随 yaml/coverage")
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.config.use_cache = False
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
        use_pipeline = bool(plan.block_list) and str(getattr(args, "compressor", "public_random")) not in {
            "block_topk",
            "dense",
        }

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
            return {
                "state": local_global,
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
        global_state = dl["state"] if isinstance(dl, dict) else dl
        download_bytes = float(dl.get("bytes", 0.0)) if isinstance(dl, dict) else 0.0
        download_mode = str(dl.get("mode", "unknown")) if isinstance(dl, dict) else "unknown"
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
        # 不要 unwrap，保留 FSDP 包装
        load_full_state_fsdp(model, global_state)
        accelerator.wait_for_everyone()
        t_broadcast_load = time.monotonic() - t_load0

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

        # SCALE-3：逐 FSDP unit 提取 block delta，不 gather 完整 state
        # 这是 collective op，所有 rank 同步执行
        # public_random compressor 不需要 energies；其他 compressor 回退到 full state
        use_scale3 = str(getattr(args, "compressor", "public_random") or "public_random") in ("public_random", "dense")
        block_delta: Dict[str, Any] = {}
        full_state: Optional[Dict[str, torch.Tensor]] = None

        if use_scale3 and selected:
            t_enc0 = time.monotonic()
            # 所有 rank 同步：逐 FSDP unit unshard 提取 block delta
            block_delta, memory = get_sharded_block_delta(
                model,
                local_global=local_global or {},
                memory=memory or {},
                selected_by_key=selected,
                transfer_dtype=transfer_dtype,
                memory_decay=args.memory_decay,
                is_main=is_main,
            )
            accelerator.wait_for_everyone()
            t_encode = time.monotonic() - t_enc0
            if is_main:
                logger.info("SCALE-3: extracted block_delta from %s FSDP units in %.2fs (no full gather)",
                            len([m for m in model.modules() if 'FullyShardedDataParallel' in type(m).__name__]),
                            t_encode)
        else:
            # 回退：非 SCALE-3 路径（block_topk 等需要 energies 的 compressor）
            full_state = get_full_state_fsdp(model)
        mem_box: List[Any] = [memory]

        def _pipeline_upload_and_apply():
            """流式 per-block pipeline：逐 block 上传，聚合好的立即下载 apply。"""
            nonlocal local_global, local_version
            assert minio is not None and global_state is not None and mem_box[0] is not None

            if not use_scale3:
                # 回退路径：从 full_state 计算 block_delta
                assert full_state is not None
                t_enc0 = time.monotonic()
                delta = sub_state(full_state, global_state)
                to_send = add_state(delta, mem_box[0])
                assert transfer_dtype is not None
                groups = build_group_blocks(to_send, block_size=args.block_size)
                flat_blocks = flatten_group_blocks(groups)
                energies = block_energies(to_send, flat_blocks)
                compressor = str(getattr(args, "compressor", "public_random") or "public_random")
                rho = float(getattr(args, "rho", 0.0) or 0.0)
                if rho <= 0:
                    rho = 1.0 / max(int(plan.coverage_h or 1), 1)
                local_selected = selected
                if compressor == "block_topk":
                    local_selected = selected_from_flat(flat_blocks, select_topk_indices(energies, rho))
                elif compressor == "dense":
                    local_selected = selected_from_flat(flat_blocks, range(len(flat_blocks)))
                block_delta_local = encode_block_delta(to_send, local_selected, dtype=transfer_dtype)
                mem_box[0] = update_block_memory(to_send, local_selected, args.memory_decay)
                t_encode = time.monotonic() - t_enc0
                nonlocal_block_delta = block_delta_local
            else:
                # SCALE-3 路径：block_delta 已在主循环提取
                nonlocal_block_delta = block_delta
                t_encode = 0.0
                energies = []  # SCALE-3 不计算 energies（public_random 不需要）

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
                if key_name in nonlocal_block_delta:
                    for s, e, slice_data in nonlocal_block_delta[key_name]:
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
                        for it in nonlocal_block_delta.get(key_name, []):
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
                        "block_energies": energies if int(gidx) == 0 else [],
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
                # Evaluation-compatible WAN timing aliases
                "wan_download_s": round(t_download, 3),
                "wan_upload_s": round(t_upload, 3),
                "model_delta_bytes": int(total_upload_bytes),
            }
            return timings

        def _batch_upload_and_notify():
            """批量模式（回退）：打包成一个 blocks.pt 上传。"""
            assert minio is not None and global_state is not None and mem_box[0] is not None

            if use_scale3:
                # SCALE-3 路径：block_delta 已在主循环提取
                block_delta_batch = block_delta
                t_encode = 0.0
                energies = []  # SCALE-3 不计算 energies
            else:
                # 回退路径：从 full_state 计算 block_delta
                assert full_state is not None
                t_enc0 = time.monotonic()
                delta = sub_state(full_state, global_state)
                to_send = add_state(delta, mem_box[0])
                assert transfer_dtype is not None
                groups = build_group_blocks(to_send, block_size=args.block_size)
                flat = flatten_group_blocks(groups)
                energies = block_energies(to_send, flat)
                compressor = str(getattr(args, "compressor", "public_random") or "public_random")
                rho = float(getattr(args, "rho", 0.0) or 0.0)
                if rho <= 0:
                    rho = 1.0 / max(int(plan.coverage_h or 1), 1)
                local_selected = selected
                if compressor == "block_topk":
                    local_selected = selected_from_flat(flat, select_topk_indices(energies, rho))
                elif compressor == "dense":
                    local_selected = selected_from_flat(flat, range(len(flat)))
                block_delta_batch = encode_block_delta(to_send, local_selected, dtype=transfer_dtype)
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
                    if key_name in block_delta_batch:
                        for s, e, sd in block_delta_batch[key_name]:
                            if s == start and e == end:
                                slice_data = sd; break
                    if slice_data is None:
                        slice_data = torch.zeros(end - start, dtype=transfer_dtype or torch.float16)
                    is_int8_item = (slice_data.dtype == torch.int8) if hasattr(slice_data, "dtype") else False
                    if is_int8_item:
                        for it in block_delta_batch.get(key_name, []):
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
                    "block_delta": block_delta_batch,
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
            body = {"num_examples": int(num_examples), "train_loss": float(train_loss), "timings": timings, "block_energies": energies}
            if eval_loss_val is not None:
                body["eval_loss"] = float(eval_loss_val)
            resp = requests.post(
                f"{server}/api/round/{round_idx}/client/{args.client_id}/upload-complete",
                json=body, timeout=60, headers=auth_headers, verify=_REQUESTS_VERIFY,
            )
            resp.raise_for_status()
            timings["notify_server_s"] = round(time.monotonic() - t_notify0, 3)
            return timings

        # 执行上传（pipeline 或 batch）
        upload_fn = _pipeline_upload_and_apply if use_pipeline else _batch_upload_and_notify
        pre_wait_timings = run_rank0_io_with_heartbeat(
            accelerator,
            upload_fn if is_main else (lambda: {}),
            interval_s=args.poll_interval,
            timeout_s=3600.0,
            label=f"upload-{round_idx}",
        )
        memory = mem_box[0]

        # RES-2：每轮把 memory + local_version 持久化到本地
        if is_main and client_state_dir is not None and memory is not None:
            try:
                torch.save(memory, client_state_dir / "memory.pt")
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
                "post_delta_s": round(float((post or {}).get("seconds", 0.0)), 3),
                "post_delta_MiB": round(float((post or {}).get("bytes", 0.0)) / (1024 * 1024), 3),
                "round_total_s": round(time.monotonic() - t_round0, 3),
            }

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
