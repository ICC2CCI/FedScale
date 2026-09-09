"""S3R10: Layer-based sharding, continuous 20% layers per shard, random shard selection + memory decay 0.9.

24 layers split into 5 continuous shards (0-4, 5-9, 10-14, 15-19, 20-23).
Each round: randomly select 1 shard (20% layers), upload its delta + memory.

Usage:
    python scripts/run_s3r10_layer_continuous_random.py
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

BASE_MODEL = "/data/models/Qwen/Qwen2.5-0.5B"
DATA_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/flower-llm/llamafactory-local/assets/datasets")
OUTPUT_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r11-layer-random20")
LOG_PATH = Path("/data/home/qiaoyanchen/liuchao/fedscale/logs/s3r11-layer-random20.log")
NUM_ROUNDS = 20
LOCAL_STEPS_PER_ROUND = 30
BATCH = 8
GRAD_ACCUM = 2
LR = 1e-5
SEQ_LEN = 512
SEED = 20260831
NUM_CLIENTS = 2
MEMORY_DECAY = 0.9
NUM_SHARDS = 5
MASK_RATIO_LAYERS = 0.2

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


def build_layer_shards(state, num_shards):
    """Split model layers into num_shards continuous groups. Return list of key-sets."""
    layer_keys = sorted([k for k in state if ".layers." in k and k.endswith(".weight")])
    layer_nums = sorted(set(int(k.split(".layers.")[1].split(".")[0]) for k in layer_keys))
    num_layers = len(layer_nums)
    shards = [[] for _ in range(num_shards)]
    for ln in layer_nums:
        shard_idx = min(int(ln / num_layers * num_shards), num_shards - 1)
        for k in state:
            if f".layers.{ln}." in k:
                shards[shard_idx].append(k)
    non_layer_keys = [k for k in state if k not in set(sum(shards, []))]
    for k in non_layer_keys:
        if state[k].is_floating_point():
            shards[0].append(k)
    return [set(s) for s in shards]


def encode_shard_delta(to_send, selected_keys):
    """Return only selected_keys from to_send (already delta+memory)."""
    return {k: to_send[k].clone() for k in selected_keys if k in to_send and to_send[k].is_floating_point()}


def apply_shard_delta(global_state, shard_deltas, weights):
    """global[selected_keys] += avg(shard_deltas) for selected keys only."""
    total_w = sum(weights)
    for k in global_state:
        if not global_state[k].is_floating_point():
            continue
        has_key = any(k in sd for sd in shard_deltas)
        if not has_key:
            continue
        acc = torch.zeros_like(global_state[k], dtype=torch.float32)
        for sd, w in zip(shard_deltas, weights):
            if k in sd:
                acc.add_(sd[k].to(dtype=torch.float32), alpha=w/total_w)
        global_state[k] = (global_state[k].to(dtype=torch.float32) + acc).to(dtype=global_state[k].dtype)


def dequant_shard(shard_delta, selected_keys):
    """No quantization in this version, dequant = identity."""
    return {k: v.clone() for k, v in shard_delta.items() if k in selected_keys}


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
base_state = {k: v.clone() for k, v in global_state.items()}

print("Building layer shards...", flush=True)
all_layer_nums = sorted(set(int(k.split(".layers.")[1].split(".")[0]) for k in global_state if ".layers." in k))
num_layers = len(all_layer_nums)
num_select = max(1, int(num_layers * MASK_RATIO_LAYERS))
non_layer_keys = [k for k in global_state if ".layers." not in k and global_state[k].is_floating_point()]
print(f"  {num_layers} layers, select {num_select} random layers/round ({MASK_RATIO_LAYERS*100:.0f}%)", flush=True)
shards = build_layer_shards(global_state, NUM_SHARDS)
for i, s in enumerate(shards):
    layer_nums = sorted(set(int(k.split(".layers.")[1].split(".")[0]) for k in s if ".layers." in k))
    print(f"  shard {i}: {len(s)} keys, layers {layer_nums[0] if layer_nums else '?'}-{layer_nums[-1] if layer_nums else '?'}", flush=True)

client_memory = [zero_state_like(global_state) for _ in range(NUM_CLIENTS)]

round_log = []
for rnd in range(1, NUM_ROUNDS + 1):
    selected_layers = set(random.sample(all_layer_nums, num_select))
    selected_keys = set(non_layer_keys)
    for k in global_state:
        if ".layers." in k:
            ln = int(k.split(".layers.")[1].split(".")[0])
            if ln in selected_layers:
                selected_keys.add(k)
    print(f"\n=== Round {rnd}/{NUM_ROUNDS} | {len(selected_layers)} random layers: {sorted(selected_layers)} | {len(selected_keys)} keys ===", flush=True)

    client_deltas = []
    client_losses = []
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

        shard_delta = encode_shard_delta(to_send, selected_keys)
        client_deltas.append(shard_delta)

        dequant = dequant_shard(shard_delta, selected_keys)
        new_memory = to_send
        for k in new_memory:
            if k in dequant:
                new_memory[k] = sub_state({k: new_memory[k]}, {k: dequant[k]})[k]
            if new_memory[k].is_floating_point():
                new_memory[k] = (new_memory[k].to(dtype=torch.float32) * MEMORY_DECAY).to(dtype=new_memory[k].dtype)
        client_memory[cid] = new_memory

        del trainer, trained_state, delta, to_send, shard_delta, dequant
        torch.cuda.empty_cache()
        print(f"    train_loss={last_loss:.4f}", flush=True)

    print(f"  Server: global += avg(shard deltas)...", flush=True)
    apply_shard_delta(global_state, client_deltas, [len(c) for c in client_train])
    del client_deltas
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
    entry = {"round": rnd, "avg_train_loss": round(avg_train, 4), "eval_loss": round(eval_loss, 4), "selected_layers": sorted(selected_layers)}
    round_log.append(entry)
    print(f"  Round {rnd}: avg_train={avg_train:.4f} eval_loss={eval_loss:.4f} layers={sorted(selected_layers)}", flush=True)
    with open(LOG_PATH, "w") as f:
        json.dump(round_log, f, indent=2)

torch.save(global_state, OUTPUT_DIR / "final_global.pt")
with open(OUTPUT_DIR / "round_log.json", "w") as f:
    json.dump(round_log, f, indent=2)
print(f"\n=== S3R10 DONE ===", flush=True)
print(json.dumps(round_log, indent=2), flush=True)
