#!/usr/bin/env python3
"""离线评估 MinIO 中各轮 global_state 的 eval_loss（在有 GPU 的 ICC 上跑）。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from shared.minio_client import MinIOClient  # noqa: E402
from shared.protocol import DEFAULT_BUCKET, DEFAULT_SEQ_LEN, global_state_key  # noqa: E402
from shared.state_dict_utils import load_full_state_fsdp  # noqa: E402

logger = logging.getLogger("eval_minio_rounds")


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


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        # mean over tokens roughly via loss * batch
        bs = batch["input_ids"].size(0)
        total += float(out.loss.item()) * bs
        n += bs
    return total / max(n, 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--minio-endpoint", required=True)
    p.add_argument("--minio-access-key", default="fedscale")
    p.add_argument("--minio-secret-key", default="fedscale-minio-2026")
    p.add_argument("--minio-bucket", default=DEFAULT_BUCKET)
    p.add_argument("--model-path", required=True)
    p.add_argument("--eval-path", required=True)
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--rounds-from", type=int, default=0)
    p.add_argument("--rounds-to", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--max-batches", type=int, default=0, help="0=full eval set")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="right", legacy=False, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_json(Path(args.eval_path))
    texts = to_chat_texts(rows, tokenizer)
    ds = ChatDataset(texts, tokenizer, args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    if args.max_batches > 0:
        from itertools import islice

        class _Limited:
            def __iter__(self):
                return islice(loader, args.max_batches)

            def __len__(self):
                return args.max_batches

        eval_loader = _Limited()
    else:
        eval_loader = loader

    logger.info("Loading base model on %s", device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False

    minio = MinIOClient(
        endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        bucket=args.minio_bucket,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            results = []
    done = {int(r["round"]) for r in results if "round" in r and r.get("eval_loss") is not None}

    for rnd in range(args.rounds_from, args.rounds_to + 1):
        if rnd in done:
            logger.info("Skip round %s (already have eval)", rnd)
            continue
        key = global_state_key(rnd)
        logger.info("Downloading %s", key)
        t0 = time.monotonic()
        state, nbytes = minio.get_torch_with_size(key, map_location="cpu")
        logger.info("Got %.1f MiB in %.1fs", nbytes / 1024 / 1024, time.monotonic() - t0)

        # 非 FSDP 包装：直接 load_state_dict
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("missing keys: %s...", list(missing)[:5])
        if unexpected:
            logger.warning("unexpected keys: %s...", list(unexpected)[:5])
        del state

        t1 = time.monotonic()
        loss = eval_loss(model, eval_loader, device)
        dt = time.monotonic() - t1
        entry = {
            "round": rnd,
            "eval_loss": round(float(loss), 6),
            "eval_seconds": round(dt, 3),
            "state_MiB": round(nbytes / 1024 / 1024, 3),
            "n_eval_examples": len(ds) if args.max_batches <= 0 else args.max_batches * args.batch_size,
        }
        results = [r for r in results if int(r.get("round", -1)) != rnd]
        results.append(entry)
        results.sort(key=lambda x: int(x["round"]))
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Round %s eval_loss=%.6f (%.1fs)", rnd, loss, dt)
        torch.cuda.empty_cache()

    logger.info("Done -> %s", out_path)


if __name__ == "__main__":
    main()
