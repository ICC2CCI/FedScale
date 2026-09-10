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
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from shared.block_selection import (  # noqa: E402
    add_block_delta,
    encode_block_delta,
    resolve_transfer_dtype,
    update_block_memory,
)
from shared.minio_client import MinIOClient  # noqa: E402
from shared.protocol import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_BUCKET,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LOCAL_STEPS,
    DEFAULT_LR,
    DEFAULT_MEMORY_DECAY,
    DEFAULT_SEQ_LEN,
    DEFAULT_TRANSFER_DTYPE,
    RoundPlan,
    global_delta_key,
    global_state_key,
    selected_from_jsonable,
    upload_blocks_key,
)
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


def wait_json(url: str, timeout_s: float = 3600.0, interval_s: float = 2.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=30)
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
) -> Dict[str, Any]:
    """全 ranks 一起短轮询等待聚合结果，避免 rank0 单线程卡 NCCL。"""
    deadline = time.time() + timeout_s
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        flag = 0
        if accelerator.is_main_process:
            try:
                resp = requests.get(f"{server}/api/round/{round_idx}/result", timeout=30)
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
) -> float:
    model.train()
    step = 0
    micro = 0
    loss_sum = 0.0
    loss_count = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    while step < local_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            loss_sum += float(loss.detach().item())
            loss_count += 1
            if accelerator.sync_gradients:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if accelerator.is_main_process and step % 10 == 0:
                    logger.info(
                        "local step %s/%s loss=%.4f",
                        step,
                        local_steps,
                        loss_sum / max(loss_count, 1),
                    )
            else:
                micro += 1
    return loss_sum / max(loss_count, 1)


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
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    rank = accelerator.process_index
    is_main = accelerator.is_main_process

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
            lambda: wait_json(f"{server}/api/round/current", interval_s=args.poll_interval),
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
            lambda: wait_json(f"{server}/api/round/{round_idx}/plan", interval_s=args.poll_interval),
            interval_s=args.poll_interval,
            timeout_s=3600.0,
            label=f"wait-plan-{round_idx}",
        )
        plan = RoundPlan.from_dict(plan_raw)
        selected = selected_from_jsonable(plan.selected_by_key)

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

        t_train0 = time.monotonic()
        train_loss = train_local_steps(
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
        eval_loss_val: Optional[float] = None
        t_eval = 0.0
        if eval_loader is not None:
            t_eval0 = time.monotonic()
            eval_loss_val = eval_local_batches(
                model, eval_loader, accelerator, max_batches=args.eval_max_batches
            )
            t_eval = time.monotonic() - t_eval0
            if is_main:
                logger.info("Online eval_loss=%.6f in %.1fs", eval_loss_val, t_eval)
        accelerator.wait_for_everyone()

        full_state = get_full_state_fsdp(model)
        mem_box: List[Any] = [memory]

        def _upload_and_notify():
            assert minio is not None and global_state is not None and mem_box[0] is not None and full_state is not None
            t_enc0 = time.monotonic()
            delta = sub_state(full_state, global_state)
            to_send = add_state(delta, mem_box[0])
            assert transfer_dtype is not None
            block_delta = encode_block_delta(to_send, selected, dtype=transfer_dtype)
            mem_box[0] = update_block_memory(to_send, selected, args.memory_decay)
            t_encode = time.monotonic() - t_enc0
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
                up_key,
                plan.n_selected_blocks,
                plan.upload_ratio * 100,
                train_loss,
                f"{eval_loss_val:.4f}" if eval_loss_val is not None else "null",
            )
            t_up0 = time.monotonic()
            upload_bytes = minio.put_torch(up_key, payload)
            t_upload = time.monotonic() - t_up0
            logger.info(
                "Uploaded %s size=%.2f MiB in %.2fs",
                up_key,
                upload_bytes / (1024 * 1024),
                t_upload,
            )

            mode_code = {
                "cache": 0.0,
                "delta": 1.0,
                "full": 2.0,
                "local_base": 3.0,
            }.get(download_mode, 2.0)
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
            }
            t_notify0 = time.monotonic()
            body = {
                "num_examples": int(num_examples),
                "train_loss": float(train_loss),
                "timings": timings,
            }
            if eval_loss_val is not None:
                body["eval_loss"] = float(eval_loss_val)
            resp = requests.post(
                f"{server}/api/round/{round_idx}/client/{args.client_id}/upload-complete",
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            timings["notify_server_s"] = round(time.monotonic() - t_notify0, 3)
            return timings

        pre_wait_timings = run_rank0_io_with_heartbeat(
            accelerator,
            _upload_and_notify if is_main else (lambda: {}),
            interval_s=args.poll_interval,
            timeout_s=3600.0,
            label=f"upload-{round_idx}",
        )
        memory = mem_box[0]

        t_wait0 = time.monotonic()
        result = wait_aggregate_with_heartbeat(
            accelerator,
            server=server,
            round_idx=round_idx,
            poll_interval=args.poll_interval,
            timeout_s=3600.0,
        )
        t_wait = time.monotonic() - t_wait0

        # 聚合完成后立刻打上本轮 global_delta，下一轮即可 cache/少下载
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
            logger.info(
                "Post-aggregate applied %s size=%.2f MiB in %.2fs (now version=%s)",
                dkey,
                nbytes / (1024 * 1024),
                dt,
                local_version,
            )
            return {"applied": 1.0, "bytes": float(nbytes), "seconds": float(dt)}

        post = run_rank0_io_with_heartbeat(
            accelerator,
            _apply_post_aggregate_delta if is_main else (lambda: {"applied": 0.0, "bytes": 0.0, "seconds": 0.0}),
            interval_s=args.poll_interval,
            timeout_s=3600.0,
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
                )
                treq.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to report client timing: %s", exc)

        accelerator.wait_for_everyone()
    if is_main:
        logger.info("Client %s done", args.client_id)


if __name__ == "__main__":
    main()
