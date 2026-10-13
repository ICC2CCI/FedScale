"""SketchOPA 实验共享组件：数据加载、模型加载、训练、eval。

仅依赖 torch + transformers + accelerate（在 ICC 节点 flwr-ft 环境运行）。
本机（Central）无 GPU/transformers，仅作代码编写与逻辑校验，实际实验在 ICC 节点跑。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader


def load_json(path: str) -> List:
    with open(path) as f:
        return json.load(f)


def build_texts(items: List[Dict]) -> List[str]:
    """把 medical_flashcards 格式转为 prompt 文本。"""
    texts = []
    for it in items:
        instr = it.get("instruction", "")
        inp = it.get("input", "")
        out = it.get("output", "")
        if inp:
            prompt = f"{instr}\n{inp}"
        else:
            prompt = instr
        text = f"### Question:\n{prompt}\n\n### Answer:\n{out}"
        texts.append(text)
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


def build_dataloader(
    data_path: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    items = load_json(data_path)
    texts = build_texts(items)
    ds = ChatDataset(texts, tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def build_eval_loader(
    eval_path: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
) -> DataLoader:
    items = load_json(eval_path)
    texts = build_texts(items)
    ds = ChatDataset(texts, tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def patch_qwen2_attn_fp32_qk() -> None:
    """V100 无 bf16：fp16 QK matmul 会 inf。把 QK 升到 fp32，权重仍 fp16。"""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    if getattr(Qwen2Attention.forward, "_fedscale_fp32_qk", False):
        return
    orig_fwd = Qwen2Attention.forward

    def forward(self, *args, **kwargs):
        orig_matmul = torch.matmul
        head_dim = int(getattr(self, "head_dim", 0) or 0)

        def matmul_qk_fp32(a, b):
            if (
                head_dim
                and torch.is_tensor(a)
                and torch.is_tensor(b)
                and a.dim() == 4
                and b.dim() == 4
                and a.shape[-1] == head_dim
            ):
                with torch.autocast(device_type="cuda", enabled=False):
                    return orig_matmul(a.float(), b.float())
            return orig_matmul(a, b)

        torch.matmul = matmul_qk_fp32
        try:
            return orig_fwd(self, *args, **kwargs)
        finally:
            torch.matmul = orig_matmul

    forward._fedscale_fp32_qk = True
    Qwen2Attention.forward = forward


def load_model(model_path: str, torch_dtype=torch.float16):
    from transformers import AutoModelForCausalLM
    patch_qwen2_attn_fp32_qk()
    kwargs = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="eager",
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config._fedscale_attn_impl = "eager"
    model.config._fedscale_attn_qk = "fp32"
    return model


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True,
    )


def causal_lm_loss_fp32(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def train_local_steps(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    local_steps: int,
    device,
    grad_accum: int = 1,
    accelerator=None,
) -> Tuple[float, List[float]]:
    """本地训练 local_steps 步，返回 (avg_loss, step_losses)。"""
    model.train()
    step = 0
    total_loss = 0.0
    step_losses: List[float] = []
    data_iter = iter(dataloader)
    import math

    while step < local_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = causal_lm_loss_fp32(outputs.logits, batch["labels"]) / grad_accum
            accelerator.backward(loss) if accelerator else loss.backward()
            l = loss.item()
            if math.isnan(l) or math.isinf(l):
                l = 0.0
            accum_loss += l
        if accelerator is not None:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1
        avg = accum_loss / grad_accum
        total_loss += avg
        step_losses.append(avg)
    return total_loss / max(local_steps, 1), step_losses


@torch.no_grad()
def evaluate(model, eval_loader: DataLoader, device) -> float:
    """评估，返回平均 eval loss。"""
    model.eval()
    total_loss = 0.0
    n = 0
    import math
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = causal_lm_loss_fp32(outputs.logits, batch["labels"])
        l = loss.item()
        if math.isnan(l) or math.isinf(l):
            l = 0.0
        total_loss += l * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
    return total_loss / max(n, 1)


def get_model_state_dict(model) -> Dict[str, torch.Tensor]:
    """获取模型 state_dict（CPU，float32）。FSDP 下需用 gather；单卡直接 state_dict。"""
    sd = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
    return sd


def load_model_state_dict(model, state_dict: Dict[str, torch.Tensor]) -> None:
    """加载 state_dict 到模型（兼容 dtype）。"""
    sd = {k: v.to(model.state_dict()[k].dtype if k in model.state_dict() else torch.float32)
          for k, v in state_dict.items()}
    model.load_state_dict(sd, strict=True)


def get_full_state_fsdp(model):
    """获取完整 state_dict。单卡直接用 model.state_dict()。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def setup_accelerate():
    """初始化 Accelerator。"""
    from accelerate import Accelerator
    accelerator = Accelerator()
    return accelerator
