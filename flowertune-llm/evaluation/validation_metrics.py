"""Teacher-forced assistant-only validation loss and perplexity (Path A).

Implements the evaluation principles document §4-6:

    "prompt + reference 一起输入模型；不自由生成；
     只统计 output token 的预测误差。"

**P0 fix**: Uses ``return_offsets_mapping=True`` to precisely locate the
boundary between prompt and response tokens, instead of re-tokenizing the
prompt text and relying on token-count equality.  This avoids off-by-one
errors when tokenizer special-token behaviour differs between full-text and
prompt-only tokenization.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


# Response markers supported by the training pipeline.
RESPONSE_MARKERS = (
    "### Response:",
    "<|im_start|>assistant\n",
)


def _find_response_marker(text: str) -> str | None:
    """Return the first response marker found in *text*, or ``None``."""
    for marker in RESPONSE_MARKERS:
        if marker in text:
            return marker
    return None


def _split_prompt_reference(text: str) -> tuple[str, str]:
    """Split a formatted sample into ``(prompt, reference)``.

    Falls back to returning the full text as prompt and empty reference when
    no known marker is found.
    """
    marker = _find_response_marker(text)
    if marker is None:
        return text, ""
    idx = text.index(marker) + len(marker)
    # Skip leading whitespace after marker.
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return text[:idx], text[idx:]


def compute_validation_metrics(
    model,
    tokenizer,
    evalset,
    device: str,
    max_samples: int = 50,
    max_length: int = 512,
) -> dict:
    """Compute assistant-only validation loss and perplexity.

    Uses teacher forcing: the full text (prompt + reference) is fed to the
    model, but cross-entropy is computed only on reference/output tokens.

    **Offset-mapping approach** (P0 fix): Each token's character span is
    compared against the prompt boundary.  Tokens whose start offset falls
    within the prompt prefix are masked to ``-100`` and excluded from loss.

    Args:
        model: Unwrapped model (PEFT or full) in eval mode.
        tokenizer: HuggingFace tokenizer.
        evalset: Dataset with a ``text`` column.
        device: CUDA device string.
        max_samples: Maximum samples to evaluate.
        max_length: Maximum tokenized sequence length.

    Returns:
        ``{"val_loss": float, "perplexity": float, "loss_scope": str, ...}``
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    evaluated = 0
    skipped = 0

    with torch.inference_mode():
        for i in range(min(len(evalset), max_samples)):
            text = evalset[i]["text"]
            prompt, reference = _split_prompt_reference(text)

            if not reference:
                skipped += 1
                continue

            encodings = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            offsets = encodings.pop("offset_mapping")[0]
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            if input_ids.shape[1] < 2:
                skipped += 1
                continue

            # Build labels: mask prompt tokens to -100 using character offsets.
            labels = input_ids.clone()
            answer_start = len(prompt)
            for tok_idx, (start, end) in enumerate(offsets.tolist()):
                if start < answer_start or start == end:
                    labels[0, tok_idx] = -100

            # Shift for next-token prediction: labels[t] predicts input_ids[t+1].
            shifted_labels = labels[:, 1:]
            answer_tokens = int((shifted_labels != -100).sum().item())

            if answer_tokens == 0:
                skipped += 1
                continue

            device_batch = {
                key: val.to(device) for key, val in encodings.items()
            }
            logits = model(**device_batch, use_cache=False).logits[:, :-1, :]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shifted_labels.to(logits.device).reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += answer_tokens
            evaluated += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        "loss_scope": "assistant_response_only",
        "evaluated_samples": evaluated,
        "skipped_samples": skipped,
        "answer_tokens": total_tokens,
    }
