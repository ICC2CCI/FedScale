"""S3R5: Sparsified SGD with Memory (error feedback) adapted to FedAvg.

Paper: "Sparsified SGD with Memory" (Stich et al., NeurIPS 2018)
Key: client maintains memory (unsent update), uploads (delta + memory) for
selected blocks, server accumulates averaged delta (NOT parameter average).

Flow per round:
  client: delta = trained - global
          to_send = delta + memory
          upload selected blocks of to_send (INT8 quantized)
          memory = to_send - dequant(upload)   # quantization error stays in memory
  server: global += avg(client uploads)

Usage:
    python scripts/run_s3r5_sgd_memory.py
"""
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
)

BASE_MODEL = "/data/models/Qwen/Qwen2.5-0.5B"
DATA_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/flower-llm/llamafactory-local/assets/datasets")
OUTPUT_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r6-sgd-memory-decay")
LOG_PATH = Path("/data/home/qiaoyanchen/liuchao/fedscale/logs/s3r6-sgd-memory-decay.log")
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
MEMORY_DECAY = 0.9

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


def sub_state(a, b):
    out = {}
    for k in a:
        av = a[k]; bv = b.get(k)
        if bv is None:
            out[k] = av.clone() if hasattr(av, "clone") else av
        elif av.is_floating_point():
            out[k] = (av.to(dtype=torch.float32) - bv.to(dtype=torch.float32)).to(dtype=av.dtype)
        else:
            out[k] = av.clone()
    return out


def add_state(a, b):
    out = {}
    for k in a:
        av = a[k]; bv = b.get(k)
        if bv is None:
            out[k] = av.clone() if hasattr(av, "clone") else av
        elif av.is_floating_point():
            out[k] = (av.to(dtype=torch.float32) + bv.to(dtype=torch.float32)).to(dtype=av.dtype)
        else:
            out[k] = av.clone()
    return out


def zero_state_like(state):
    return {k: torch.zeros_like(v) if v.is_floating_point() else v.clone() for k, v in state.items()}


def apply_sparse_delta_inplace(global_state, encoded_list, layout):
    """Server: global += avg(client uploads). Each encoded is INT8 delta for selected blocks."""
    weight = 1.0 / len(encoded_list)
    for enc in encoded_list:
        apply_fedscale_int8_delta(global_state, enc, layout, weight=weight, clone=False)


def dequant_encoded_to_full(encoded, layout, state_template):
    """Reconstruct dequantized delta (full state, selected blocks only) from encoded INT8."""
    result = zero_state_like(state_template)
    from flowertune_llm.fedscale_state import encoded_block_ids
    ids = encoded_block_ids(encoded, layout)
    offsets = encoded["__fedscale_block_offsets__"].to(dtype=torch.int64)
    scales = encoded["__fedscale_block_scales__"].to(dtype=torch.float32)
    values = encoded["__fedscale_block_values__"].to(dtype=torch.float32)
    for idx, bid in enumerate(ids):
        blk = layout.blocks[bid]
        delta_slice = values[int(offsets[idx]):int(offsets[idx+1])] * scales[idx]
        cursor = 0
        for tr in layout.tensors:
            ov_s = max(blk.start, tr.start)
            ov_e = min(blk.end, tr.end)
            if ov_s >= ov_e:
                continue
            size = ov_e - ov_s
            lo = ov_s - tr.start
            result[tr.name].reshape(-1)[lo:lo+size] = delta_slice[cursor:cursor+size].to(dtype=result[tr.name].dtype)
            cursor += size
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
print(f"  per client: {[len(c) for c in client_train]}, eval: {len(eval_rows)}", flush=True)
eval_ds = to_chat_dataset(eval_rows)

print("Loading base model (bf16)...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=False, attn_implementation="eager").to("cuda")
global_state = get_full_state(base_model)

print("Building canonical layout...", flush=True)
layout = build_canonical_layout(global_state, BLOCK_SIZE)
print(f"  blocks: {len(layout.blocks)}, 20% = {int(len(layout.blocks)*MASK_RATIO)}", flush=True)

client_memory = [zero_state_like(global_state) for _ in range(NUM_CLIENTS)]
print(f"  initialized {NUM_CLIENTS} client memory (zeros, error feedback)", flush=True)

round_log = []
for rnd in range(1, NUM_ROUNDS + 1):
    block_ids = public_block_mask(layout, rnd, MASK_RATIO)
    print(f"\n=== Round {rnd}/{NUM_ROUNDS} | blocks: {len(block_ids)} ===", flush=True)

    client_encoded = []
    client_losses = []
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid} (SGD with memory)...", flush=True)
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
            model=base_model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=data_collator, tokenizer=tokenizer,
            dataset_text_field="text", max_seq_length=SEQ_LEN, packing=False,
        )
        train_res = trainer.train()
        last_loss = train_res.training_loss
        client_losses.append(last_loss)

        trained_state = get_full_state(base_model)
        delta = sub_state(trained_state, global_state)
        to_send = add_state(delta, client_memory[cid])

        encoded = encode_fedscale_int8_delta(to_send, zero_state_like(global_state), layout, block_ids)
        client_encoded.append(encoded)

        dequant = dequant_encoded_to_full(encoded, layout, global_state)
        new_memory = sub_state(to_send, dequant)
        for k in new_memory:
            if new_memory[k].is_floating_point():
                new_memory[k] = (new_memory[k].to(dtype=torch.float32) * MEMORY_DECAY).to(dtype=new_memory[k].dtype)
        client_memory[cid] = new_memory

        mem_norm = sum(float(v.float().norm()) for v in client_memory[cid].values() if v.is_floating_point())
        del trainer, trained_state, delta, to_send, dequant
        torch.cuda.empty_cache()
        print(f"    train_loss={last_loss:.4f}, memory_norm={mem_norm:.2f}", flush=True)

    print(f"  Server: global += avg(client uploads)...", flush=True)
    apply_sparse_delta_inplace(global_state, client_encoded, layout)
    del client_encoded
    torch.cuda.empty_cache()

    print(f"  Evaluating...", flush=True)
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
print(f"\n=== S3R5 DONE ===", flush=True)
print(json.dumps(round_log, indent=2), flush=True)
