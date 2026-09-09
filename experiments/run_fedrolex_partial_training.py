"""FedRolex-style Partial Training: only train selected layers, no memory needed.

Each round: server randomly selects 20% layers, clients only train those layers
(other layers frozen via requires_grad=False), upload those layers' updates.
No memory mechanism needed — untrained layers have no updates to discard.

Usage:
    python scripts/run_fedrolex_partial_training.py
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

BASE_MODEL = "/data/models/Qwen/Qwen2.5-0.5B"
DATA_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/flower-llm/llamafactory-local/assets/datasets")
OUTPUT_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/fedrolex-partial-training")
LOG_PATH = Path("/data/home/qiaoyanchen/liuchao/fedscale/logs/fedrolex-partial-training.log")
NUM_ROUNDS = 20
LOCAL_STEPS_PER_ROUND = 30
BATCH = 8
GRAD_ACCUM = 2
LR = 1e-5
SEQ_LEN = 512
SEED = 20260831
NUM_CLIENTS = 2
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


def freeze_layers(model, selected_layers):
    """Freeze all layers except selected_layers. Also freeze embed/lm_head."""
    for name, param in model.named_parameters():
        param.requires_grad = False
    for ln in selected_layers:
        for name, param in model.named_parameters():
            if f".layers.{ln}." in name:
                param.requires_grad = True


def avg_selected_states(global_state, client_states, selected_keys, weights):
    """global[selected_keys] = avg(client_states[selected_keys])."""
    total_w = sum(weights)
    for k in global_state:
        if k not in selected_keys:
            continue
        if not global_state[k].is_floating_point():
            continue
        acc = torch.zeros_like(global_state[k], dtype=torch.float32)
        for cs, w in zip(client_states, weights):
            if k in cs:
                acc.add_(cs[k].to(dtype=torch.float32), alpha=w/total_w)
        global_state[k] = acc.to(dtype=global_state[k].dtype)


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

all_layer_nums = sorted(set(int(k.split(".layers.")[1].split(".")[0]) for k in global_state if ".layers." in k))
num_layers = len(all_layer_nums)
num_select = max(1, int(num_layers * MASK_RATIO_LAYERS))
print(f"  {num_layers} layers, select {num_select} random layers/round ({MASK_RATIO_LAYERS*100:.0f}%)", flush=True)

round_log = []
for rnd in range(1, NUM_ROUNDS + 1):
    selected_layers = set(random.sample(all_layer_nums, num_select))
    selected_keys = set()
    for k in global_state:
        if ".layers." in k:
            ln = int(k.split(".layers.")[1].split(".")[0])
            if ln in selected_layers:
                selected_keys.add(k)
    print(f"\n=== Round {rnd}/{NUM_ROUNDS} | partial train layers {sorted(selected_layers)} | {len(selected_keys)} keys ===", flush=True)

    client_states = []
    client_losses = []
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid} (partial training)...", flush=True)
        load_full_state(base_model, global_state)
        freeze_layers(base_model, selected_layers)
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
            gradient_checkpointing=False,
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
        client_states.append({k: trained_state[k].clone() for k in selected_keys})

        del trainer, trained_state
        torch.cuda.empty_cache()
        n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"    train_loss={last_loss:.4f}, trainable params: {n_trainable/1e6:.1f}M / ~494M", flush=True)

    print(f"  Server: avg selected layers...", flush=True)
    avg_selected_states(global_state, client_states, selected_keys, [len(c) for c in client_train])
    del client_states
    torch.cuda.empty_cache()

    print(f"  Evaluating...", flush=True)
    load_full_state(base_model, global_state)
    for p in base_model.parameters():
        p.requires_grad = True
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
print(f"\n=== FedRolex Partial Training DONE ===", flush=True)
print(json.dumps(round_log, indent=2), flush=True)
