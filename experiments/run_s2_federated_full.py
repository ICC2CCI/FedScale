"""S2: Lightweight federated full fine-tuning simulation (full upload, 2 clients + 1 server).

Simulates Flower FedAvg on a single GPU by sequentially training 2 clients per round
and averaging their full FP16 state dicts. No Flower runtime needed; reuses the
FedAvg logic from flowertune_llm.aggregation and state helpers from model_state.

Usage:
    python scripts/run_s2_federated_full.py
"""
import copy
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

BASE_MODEL = "/data/models/Qwen/Qwen2.5-0.5B"
DATA_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/flower-llm/llamafactory-local/assets/datasets")
OUTPUT_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s2-federated-full")
LOG_PATH = Path("/data/home/qiaoyanchen/liuchao/fedscale/logs/s2-federated-full.log")
NUM_ROUNDS = 20
LOCAL_STEPS_PER_ROUND = 30
BATCH = 8
GRAD_ACCUM = 2
LR = 1e-5
SEQ_LEN = 512
SEED = 20260831
NUM_CLIENTS = 2

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
random.seed(SEED); torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def to_chat_dataset(rows):
    texts = []
    for r in rows:
        messages = [
            {"role": "system", "content": r["instruction"]},
            {"role": "user", "content": r["input"]},
            {"role": "assistant", "content": r["output"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return Dataset.from_dict({"text": texts})


def iid_split(rows, n):
    random.seed(SEED); random.shuffle(rows)
    return [rows[i::n] for i in range(n)]


def get_full_state(model):
    return {k: v.detach().to(device="cpu") for k, v in model.state_dict().items()}


def load_full_state(model, state):
    model.load_state_dict({k: v.to(device=model.device) for k, v in state.items()}, strict=True)


def fedavg(states, weights):
    keys = states[0].keys()
    total = float(sum(weights))
    result = {}
    with torch.no_grad():
        for k in keys:
            src = states[0][k]
            if src.is_floating_point():
                acc = torch.zeros_like(src, dtype=torch.float32)
                for s, w in zip(states, weights):
                    acc.add_(s[k].to(dtype=torch.float32), alpha=float(w) / total)
                result[k] = acc.to(dtype=src.dtype)
            else:
                result[k] = src.clone()
    return result


print("Loading tokenizer + base model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, padding_side="right", legacy=False, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
response_template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

print("Loading + splitting data...", flush=True)
train_rows = load_json(DATA_DIR / "medical_flashcards_train.json")
eval_rows = load_json(DATA_DIR / "medical_flashcards_eval.json")
client_train = iid_split(train_rows, NUM_CLIENTS)
print(f"  total train: {len(train_rows)}, per client: {[len(c) for c in client_train]}, eval: {len(eval_rows)}", flush=True)

eval_ds = to_chat_dataset(eval_rows)

print("Loading base model (FP16)...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=False, attn_implementation="eager").to("cuda")
global_state = get_full_state(base_model)

round_log = []
for rnd in range(1, NUM_ROUNDS + 1):
    client_states = []
    client_losses = []
    for cid in range(NUM_CLIENTS):
        print(f"\n=== Round {rnd}/{NUM_ROUNDS} Client {cid} ===", flush=True)
        load_full_state(base_model, global_state)
        base_model.train()
        train_ds = to_chat_dataset(client_train[cid])
        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / f"r{rnd}-c{cid}"),
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            max_steps=LOCAL_STEPS_PER_ROUND,
            logging_steps=5,
            save_strategy="no",
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            report_to="none",
            seed=SEED + rnd * 100 + cid,
            dataloader_num_workers=2,
        )
        trainer = SFTTrainer(
            model=base_model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=SEQ_LEN,
            packing=False,
        )
        train_res = trainer.train()
        last_loss = train_res.training_loss
        client_losses.append(last_loss)
        client_states.append(get_full_state(base_model))
        del trainer
        torch.cuda.empty_cache()
        print(f"  client {cid} train_loss={last_loss:.4f}", flush=True)

    weights = [len(c) for c in client_train]
    global_state = fedavg(client_states, weights)
    del client_states
    torch.cuda.empty_cache()

    print(f"\n--- Round {rnd} eval ---", flush=True)
    load_full_state(base_model, global_state)
    base_model.eval()
    eval_args = TrainingArguments(output_dir=str(OUTPUT_DIR / f"r{rnd}-eval"), per_device_eval_batch_size=BATCH, bf16=True, report_to="none", seed=SEED)
    eval_trainer = SFTTrainer(model=base_model, args=eval_args, eval_dataset=eval_ds, data_collator=data_collator, tokenizer=tokenizer, dataset_text_field="text", max_seq_length=SEQ_LEN, packing=False)
    eval_res = eval_trainer.evaluate()
    eval_loss = eval_res["eval_loss"]
    del eval_trainer
    torch.cuda.empty_cache()

    avg_train = sum(client_losses) / len(client_losses)
    entry = {"round": rnd, "avg_train_loss": round(avg_train, 4), "eval_loss": round(eval_loss, 4)}
    round_log.append(entry)
    print(f"  Round {rnd}: avg_train={avg_train:.4f} eval_loss={eval_loss:.4f}", flush=True)
    with open(LOG_PATH, "w") as f:
        json.dump(round_log, f, indent=2)

torch.save(global_state, OUTPUT_DIR / "final_global.pt")
with open(OUTPUT_DIR / "round_log.json", "w") as f:
    json.dump(round_log, f, indent=2)
print(f"\n=== S2 DONE. Log: {LOG_PATH} ===", flush=True)
print("round_log:", json.dumps(round_log, indent=2), flush=True)
