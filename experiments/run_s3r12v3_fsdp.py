"""S3R12v3 FSDP 客户端（ICC1/ICC2）。

用法（在仓库根目录，8 卡）::

    accelerate launch --config_file configs/accelerate_config.yaml \\
      experiments/run_s3r12v3_fsdp.py \\
      --client-id 0 \\
      --server-url http://192.168.235.42:8080 \\
      --minio-endpoint http://192.168.235.42:9000 \\
      --model-path model/Qwen/Qwen2.5-0.5B \\
      --data-path data/splits/icc1_client0_train.json
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

from shared.block_selection import encode_block_delta, update_block_memory  # noqa: E402
from shared.minio_client import MinIOClient  # noqa: E402
from shared.protocol import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_BUCKET,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LOCAL_STEPS,
    DEFAULT_LR,
    DEFAULT_MEMORY_DECAY,
    DEFAULT_SEQ_LEN,
    RoundPlan,
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

    minio = None
    memory = None
    if is_main:
        minio = MinIOClient(
            endpoint=args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            bucket=args.minio_bucket,
        )

    server = args.server_url.rstrip("/")
    num_examples = args.num_examples or len(dataset)

    while True:
        status = wait_json(f"{server}/api/round/current", interval_s=args.poll_interval)
        if status.get("status") == "finished":
            if is_main:
                logger.info("Server finished all rounds")
            break
        round_idx = int(status["round"])
        if is_main:
            logger.info("=== Round %s status=%s ===", round_idx, status.get("status"))

        plan_raw = wait_json(f"{server}/api/round/{round_idx}/plan", interval_s=args.poll_interval)
        plan = RoundPlan.from_dict(plan_raw)
        selected = selected_from_jsonable(plan.selected_by_key)

        # 下载 round-(N-1) 全局状态（仅 rank0），再广播给所有 rank 后加载进 FSDP
        global_state = None
        if is_main:
            assert minio is not None
            prev_key = global_state_key(round_idx - 1)
            logger.info("Downloading %s", prev_key)
            global_state = minio.get_torch(prev_key, map_location="cpu")
            if memory is None:
                memory = zero_state_like(global_state)

        accelerator.wait_for_everyone()
        if accelerator.num_processes > 1:
            global_state = broadcast_object(global_state if is_main else None, src=0)
        # 不要 unwrap，保留 FSDP 包装
        load_full_state_fsdp(model, global_state)
        accelerator.wait_for_everyone()

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

        full_state = get_full_state_fsdp(model)
        if is_main:
            assert minio is not None and global_state is not None and memory is not None and full_state is not None
            delta = sub_state(full_state, global_state)
            to_send = add_state(delta, memory)
            block_delta = encode_block_delta(to_send, selected)
            memory = update_block_memory(to_send, selected, args.memory_decay)

            payload = {
                "block_delta": block_delta,
                "num_examples": int(num_examples),
                "train_loss": float(train_loss),
                "client_id": int(args.client_id),
                "round": int(round_idx),
            }
            up_key = upload_blocks_key(round_idx, args.client_id)
            logger.info(
                "Uploading %s blocks=%s ratio=%.2f%% train_loss=%.4f",
                up_key,
                plan.n_selected_blocks,
                plan.upload_ratio * 100,
                train_loss,
            )
            minio.put_torch(up_key, payload)
            resp = requests.post(
                f"{server}/api/round/{round_idx}/client/{args.client_id}/upload-complete",
                json={"num_examples": int(num_examples), "train_loss": float(train_loss)},
                timeout=60,
            )
            resp.raise_for_status()

            # 等待聚合完成
            while True:
                result = wait_json(f"{server}/api/round/{round_idx}/result", interval_s=args.poll_interval)
                if result.get("done"):
                    logger.info("Round %s aggregated: %s", round_idx, result)
                    break
                if result.get("message") == "aggregation_failed":
                    raise RuntimeError(f"aggregation failed: {result}")
                time.sleep(args.poll_interval)

        accelerator.wait_for_everyone()
        # 轻微同步，避免主进程仍在上传时其他 rank 抢跑下一轮
        time.sleep(0.5 if not is_main else 0.0)

    if is_main:
        logger.info("Client %s done", args.client_id)


if __name__ == "__main__":
    main()
