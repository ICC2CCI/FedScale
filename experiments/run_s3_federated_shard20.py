"""S3: Federated sharded 20% block upload (fedscale-int8) simulation.

Each round: server picks 20% canonical blocks via rotating public mask.
Clients train full model locally but only upload the selected block deltas
(INT8 quantized). Server averages those deltas and applies to global state.
Unselected blocks' local updates are discarded.

Usage:
    python scripts/run_s3_federated_shard20.py
"""
import copy
import json
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

sys.path.insert(0, "/data/home/qiaoyanchen/liuchao/fedscale/flower-llm/flowertune-llm")
from flowertune_llm.fedscale_state import (
    build_canonical_layout,
    public_block_mask,
    encode_fedscale_int8_delta,
    apply_fedscale_int8_delta,
    is_fedscale_int8_delta,
    encoded_block_ids,
)

BASE_MODEL = "/data/models/Qwen/Qwen2.5-0.5B"
DATA_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/flower-llm/llamafactory-local/assets/datasets")
OUTPUT_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3-federated-shard20")
LOG_PATH = Path("/data/home/qiaoyanchen/liuchao/fedscale/logs/s3-federated-shard20.log")
NUM_ROUNDS = 20
LOCAL_STEPS_PER_ROUND = 30
BATCH = 8
GRAD_ACCUM = 2
LR = 1e-5
SEQ_LEN = 512
SEED = 20260831
NUM_CLIENTS = 2
BLOCK_SIZE = 1048576
MASK_RATIO = 0.2

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

print("Loading base model (bf16)...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=False, attn_implementation="eager").to("cuda")
global_state = get_full_state(base_model)

print("Building canonical layout...", flush=True)
layout = build_canonical_layout(global_state, BLOCK_SIZE)
print(f"  total elements: {layout.total_elements}, blocks: {len(layout.blocks)}, block_size: {BLOCK_SIZE}", flush=True)
print(f"  20% mask selects {int(len(layout.blocks)*MASK_RATIO)} blocks/round, full coverage in ~{int(1/MASK_RATIO)} rounds", flush=True)

round_log = []
for rnd in range(1, NUM_ROUNDS + 1):
    block_ids = public_block_mask(layout, rnd, MASK_RATIO)
    print(f"\n=== Round {rnd}/{NUM_ROUNDS} | blocks: {len(block_ids)}/{len(layout.blocks)} ({len(block_ids)/len(layout.blocks)*100:.0f}%) ===", flush=True)

    client_encoded = []
    client_losses = []
    client_num_examples = []
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid}...", flush=True)
        load_full_state(base_model, global_state)
        base_model.train()
        train_ds = to_chat_dataset(client_train[cid])
        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / f"r{rnd}-c{cid}"),
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            max_steps=LOCAL_STEPS_PER_ROUND,
            logging_steps=10,
            save_strategy="no",
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
        client_num_examples.append(len(client_train[cid]))

        trained_state = get_full_state(base_model)
        encoded = encode_fedscale_int8_delta(trained_state, global_state, layout, block_ids)
        client_encoded.append(encoded)
        del trainer, trained_state
        torch.cuda.empty_cache()
        print(f"    train_loss={last_loss:.4f}, uploaded {len(block_ids)} blocks", flush=True)

    print(f"  Aggregating {len(client_encoded)} client deltas...", flush=True)
    client_weight = 1.0 / len(client_encoded)
    for enc in client_encoded:
        apply_fedscale_int8_delta(global_state, enc, layout, weight=client_weight, clone=False)
    del client_encoded
    torch.cuda.empty_cache()

    print(f"  Evaluating global...", flush=True)
    load_full_state(base_model, global_state)
    base_model.eval()
    eval_args = TrainingArguments(output_dir=str(OUTPUT_DIR / f"r{rnd}-eval"), per_device_eval_batch_size=BATCH, bf16=True, report_to="none", seed=SEED)
    eval_trainer = SFTTrainer(model=base_model, args=eval_args, eval_dataset=eval_ds, data_collator=data_collator, tokenizer=tokenizer, dataset_text_field="text", max_seq_length=SEQ_LEN, packing=False)
    eval_res = eval_trainer.evaluate()
    eval_loss = eval_res["eval_loss"]
    del eval_trainer
    torch.cuda.empty_cache()

    avg_train = sum(client_losses) / len(client_losses)
    entry = {"round": rnd, "avg_train_loss": round(avg_train, 4), "eval_loss": round(eval_loss, 4), "blocks_uploaded": len(block_ids), "block_ids": list(block_ids)}
    round_log.append(entry)
    print(f"  Round {rnd}: avg_train={avg_train:.4f} eval_loss={eval_loss:.4f} blocks={len(block_ids)}/{len(layout.blocks)}", flush=True)
    with open(LOG_PATH, "w") as f:
        json.dump(round_log, f, indent=2)

torch.save(global_state, OUTPUT_DIR / "final_global.pt")
with open(OUTPUT_DIR / "round_log.json", "w") as f:
    json.dump(round_log, f, indent=2)
print(f"\n=== S3 DONE. Log: {LOG_PATH} ===", flush=True)
print(json.dumps(round_log, indent=2), flush=True)
