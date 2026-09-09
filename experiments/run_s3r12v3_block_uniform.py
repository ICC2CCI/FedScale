"""S3R12v3: Block-level hierarchical permutation + rotation + memory (uniform upload).

Improvement over S3R12v2: split each key into 1MB blocks, shuffle and rotate at block
level within each group. This makes per-round upload size approximately constant
(~20% of total), eliminating the 7%-37% variance of S3R12v2.

Each group (24 layers + 1 non_layer) independently Fisher-Yates shuffles its blocks.
Each round selects blocks where position % H == slot.
H=5 rounds = 1 Mask Epoch -> 100% block coverage, no starvation.
memory + decay 0.9 retained.

Usage:
    python scripts/run_s3r12v3_block_uniform.py
"""
import hashlib
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
OUTPUT_DIR = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v3-block-uniform")
LOG_PATH = Path("/data/home/qiaoyanchen/liuchao/fedscale/logs/s3r12v3-block-uniform.log")
NUM_ROUNDS = 20
LOCAL_STEPS_PER_ROUND = 30
BATCH = 8
GRAD_ACCUM = 2
LR = 1e-5
SEQ_LEN = 512
SEED = 20260831
NUM_CLIENTS = 2
MEMORY_DECAY = 0.9
COVERAGE_H = 5
BLOCK_SIZE = 524288

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


def hkdf_group_seed(epoch_seed: bytes, group_id: str) -> bytes:
    info = b"FedScale-GroupMask-v1|" + group_id.encode("utf-8")
    return hashlib.sha256(epoch_seed + b"\x00" + info).digest()


def fisher_yates_shuffle(n: int, seed: bytes) -> list:
    rng = random.Random(int.from_bytes(seed, "big"))
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def build_group_blocks(state):
    """Split each group's keys into BLOCK_SIZE-element blocks.
    Returns: dict[group_id -> list of (key_name, start_elem, end_elem)]
    """
    groups_keys = {}
    for name, tensor in sorted(state.items()):
        if not tensor.is_floating_point():
            continue
        if ".layers." in name:
            ln = int(name.split(".layers.")[1].split(".")[0])
            gid = f"layer_{ln}"
        else:
            gid = "non_layer"
        groups_keys.setdefault(gid, []).append(name)

    group_blocks = {}
    for gid, keys in groups_keys.items():
        blocks = []
        for key_name in keys:
            n_elem = state[key_name].numel()
            for start in range(0, n_elem, BLOCK_SIZE):
                end = min(start + BLOCK_SIZE, n_elem)
                blocks.append((key_name, start, end))
        group_blocks[gid] = blocks
    return group_blocks


def build_selected_blocks(group_blocks, permutations, slot):
    """Select blocks where position % H == slot from each group.
    Returns: dict[key_name -> list of (start, end)]
    """
    selected_by_key = {}
    for gid, blocks in group_blocks.items():
        perm = permutations[gid]
        for pos, block_idx in enumerate(perm):
            if pos % COVERAGE_H == slot:
                key_name, start, end = blocks[block_idx]
                selected_by_key.setdefault(key_name, []).append((start, end))
    return selected_by_key


def encode_block_delta(to_send, selected_by_key):
    """Extract block slices from to_send (flat view).
    Returns: dict[key_name -> list of (start, end, tensor_slice_1d)]
    """
    result = {}
    for key_name, slices in selected_by_key.items():
        if key_name not in to_send or not to_send[key_name].is_floating_point():
            continue
        flat = to_send[key_name].contiguous().view(-1)
        result[key_name] = [(s, e, flat[s:e].clone()) for s, e in slices]
    return result


def apply_block_delta(global_state, client_block_deltas, weights, selected_by_key):
    """Apply weighted average of block deltas to global_state (flat view)."""
    total_w = sum(weights)
    for key_name, slices in selected_by_key.items():
        if not global_state[key_name].is_floating_point():
            continue
        gflat = global_state[key_name].contiguous().view(-1)
        for s, e in slices:
            acc = torch.zeros(e - s, dtype=torch.float32)
            for cid, w in enumerate(weights):
                if key_name in client_block_deltas[cid]:
                    for ss, ee, slice_data in client_block_deltas[cid][key_name]:
                        if ss == s and ee == e:
                            acc.add_(slice_data.to(dtype=torch.float32), alpha=w / total_w)
                            break
            gflat[s:e] = (
                gflat[s:e].to(dtype=torch.float32) + acc
            ).to(dtype=global_state[key_name].dtype)
        global_state[key_name] = gflat.view(global_state[key_name].shape)


def update_block_memory(to_send, selected_by_key, decay):
    """Update memory: uploaded blocks -> 0, non-uploaded -> to_send, then all *= decay (flat view)."""
    new_memory = {}
    for key_name in to_send:
        if not to_send[key_name].is_floating_point():
            new_memory[key_name] = to_send[key_name].clone()
            continue
        flat = to_send[key_name].contiguous().view(-1).to(dtype=torch.float32).clone()
        if key_name in selected_by_key:
            for s, e in selected_by_key[key_name]:
                flat[s:e] = 0.0
        new_memory[key_name] = (flat * decay).to(dtype=to_send[key_name].dtype).view(to_send[key_name].shape)
    return new_memory


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

print("Building block layout...", flush=True)
group_blocks = build_group_blocks(global_state)
total_blocks = sum(len(v) for v in group_blocks.values())
total_elems = sum(v.numel() for v in global_state.values() if v.is_floating_point())
print(f"  {len(group_blocks)} groups, {total_blocks} blocks, {total_elems/1e6:.1f}M elems total", flush=True)
for gid in sorted(group_blocks):
    n_blocks = len(group_blocks[gid])
    elems = sum(e - s for _, s, e in group_blocks[gid])
    print(f"    {gid}: {n_blocks} blocks, {elems/1e6:.1f}M elems", flush=True)
print(f"  H={COVERAGE_H}, ~{total_blocks // COVERAGE_H} blocks/round, ~{total_elems / COVERAGE_H / 1e6:.1f}M elems/round ({100 / COVERAGE_H:.0f}%)", flush=True)

permutations = {}
current_epoch = -1

client_memory = [zero_state_like(global_state) for _ in range(NUM_CLIENTS)]

round_log = []
for rnd in range(1, NUM_ROUNDS + 1):
    slot = (rnd - 1) % COVERAGE_H
    epoch = (rnd - 1) // COVERAGE_H

    if epoch != current_epoch:
        current_epoch = epoch
        epoch_seed = hashlib.sha256(
            f"FedScale-BlockMask-v1|{SEED}|{epoch}".encode("utf-8")
        ).digest()
        for gid in group_blocks:
            n_blocks = len(group_blocks[gid])
            group_seed = hkdf_group_seed(epoch_seed, gid)
            permutations[gid] = fisher_yates_shuffle(n_blocks, group_seed)
        print(f"  New Mask Epoch {epoch}, regenerated permutations for all {len(group_blocks)} groups", flush=True)

    selected_by_key = build_selected_blocks(group_blocks, permutations, slot)
    n_selected_blocks = sum(len(v) for v in selected_by_key.values())
    selected_elems = sum(e - s for slices in selected_by_key.values() for s, e in slices)
    pct = selected_elems / total_elems * 100

    print(f"\n=== Round {rnd}/{NUM_ROUNDS} | epoch={epoch} slot={slot} | {n_selected_blocks} blocks, {selected_elems/1e6:.1f}M elems ({pct:.1f}%) ===", flush=True)

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

        block_delta = encode_block_delta(to_send, selected_by_key)
        client_deltas.append(block_delta)

        client_memory[cid] = update_block_memory(to_send, selected_by_key, MEMORY_DECAY)

        del trainer, trained_state, delta, to_send, block_delta
        torch.cuda.empty_cache()
        print(f"    train_loss={last_loss:.4f}", flush=True)

    print(f"  Server: global += avg(block deltas)...", flush=True)
    apply_block_delta(global_state, client_deltas, [len(c) for c in client_train], selected_by_key)
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
    entry = {
        "round": rnd, "avg_train_loss": round(avg_train, 4), "eval_loss": round(eval_loss, 4),
        "epoch": epoch, "slot": slot,
        "n_selected_blocks": n_selected_blocks,
        "selected_elems_M": round(selected_elems / 1e6, 1),
        "pct_of_total": round(pct, 1),
    }
    round_log.append(entry)
    print(f"  Round {rnd}: avg_train={avg_train:.4f} eval_loss={eval_loss:.4f} upload={pct:.1f}% ({selected_elems/1e6:.1f}M, {n_selected_blocks} blocks)", flush=True)
    with open(LOG_PATH, "w") as f:
        json.dump(round_log, f, indent=2)

torch.save(global_state, OUTPUT_DIR / "final_global.pt")
with open(OUTPUT_DIR / "round_log.json", "w") as f:
    json.dump(round_log, f, indent=2)
print(f"\n=== S3R12v3 DONE ===", flush=True)
print(json.dumps(round_log, indent=2), flush=True)
