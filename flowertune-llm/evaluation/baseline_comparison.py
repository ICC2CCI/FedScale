"""Base-model baseline evaluation and federated-vs-base delta computation.

Implements the evaluation principles document §14:

    | Method               | Test Loss ↓ | Test PPL ↓ | ROUGE-L ↑ | BERTScore-F1 ↑ |
    | Centralized FSDP     | ...         | ...        | ...       | ...            |
    | DDP baseline         | ...         | ...        | ...       | ...            |
    | Plain FedAvg         | ...         | ...        | ...       | ...            |
    | FedScale             | ...         | ...        | ...       | ...            |

The baseline (unmodified base model) is evaluated with the same test set,
same generation parameters, and same metrics as the federated model.  The
delta quantifies how much utility the federated training pipeline retained
or lost relative to the starting model.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import torch

from evaluation.validation_metrics import compute_validation_metrics
from evaluation.generation_metrics import compute_generation_metrics


def _load_base_model(
    model_name: str,
    device: str,
    quantization: int = 0,
    local_files_only: bool = True,
):
    """Load an unmodified base model without any PEFT adapters."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.eval()
    model.config.use_cache = True
    return model


def evaluate_with_baseline(
    model_name: str,
    federated_model,
    tokenizer,
    evalset,
    device: str,
    max_samples: int = 50,
    max_length: int = 512,
    max_new_tokens: int = 128,
    bertscore_model: str | None = None,
    quantization: int = 0,
    local_files_only: bool = True,
    skip_base: bool = False,
) -> dict:
    """Evaluate both base model and federated model, then compute delta.

    Args:
        model_name: HuggingFace model name for loading the base model.
        federated_model: The trained/federated model to evaluate.
        tokenizer: HuggingFace tokenizer.
        evalset: Dataset with ``text`` column.
        device: CUDA device string.
        max_samples: Maximum samples to evaluate.
        max_length: Maximum tokenization length.
        max_new_tokens: Maximum generation tokens.
        bertscore_model: Optional BERTScore backbone model type.
        quantization: Quantization level for base model (0/4/8).
        local_files_only: Whether to restrict to local cache.
        skip_base: If True, skip base model evaluation.

    Returns:
        Dict with ``base``, ``federated``, and ``delta`` sections.
    """
    results: dict = {}

    # --- Federated model evaluation ---
    fed_started = time.perf_counter()
    fed_val = compute_validation_metrics(
        federated_model, tokenizer, evalset, device,
        max_samples=max_samples, max_length=max_length,
    )
    fed_gen = compute_generation_metrics(
        federated_model, tokenizer, evalset, device,
        max_samples=max_samples,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        bertscore_model=bertscore_model,
    )
    fed_seconds = time.perf_counter() - fed_started
    results["federated"] = {
        "validation": fed_val,
        "generation": fed_gen,
        "evaluation_seconds": round(fed_seconds, 4),
    }

    # --- Base model evaluation ---
    if not skip_base:
        base_started = time.perf_counter()
        base_model = _load_base_model(
            model_name, device,
            quantization=quantization,
            local_files_only=local_files_only,
        )
        base_val = compute_validation_metrics(
            base_model, tokenizer, evalset, device,
            max_samples=max_samples, max_length=max_length,
        )
        base_gen = compute_generation_metrics(
            base_model, tokenizer, evalset, device,
            max_samples=max_samples,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            bertscore_model=bertscore_model,
        )
        base_seconds = time.perf_counter() - base_started
        results["base"] = {
            "validation": base_val,
            "generation": base_gen,
            "evaluation_seconds": round(base_seconds, 4),
        }

        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results["delta"] = compute_delta(results["base"], results["federated"])

    return results


def compute_delta(base: dict, federated: dict) -> dict:
    """Compute the delta (federated - base) for all key metrics.

    Returns a flat dict of signed differences.  Positive delta means the
    federated model improved over the base; negative means degradation.

    For BERTScore, if either side is ``None`` (library unavailable), the
    delta is ``None``.
    """
    base_val = base["validation"]
    fed_val = federated["validation"]
    base_gen = base["generation"]
    fed_gen = federated["generation"]

    delta = {
        "val_loss": round(fed_val["val_loss"] - base_val["val_loss"], 4),
        "perplexity": round(fed_val["perplexity"] - base_val["perplexity"], 4),
        "rouge_l_f1": round(
            fed_gen["rouge_l"]["f1"] - base_gen["rouge_l"]["f1"], 4
        ),
        "token_overlap_accuracy": round(
            fed_gen["token_overlap_accuracy"]
            - base_gen["token_overlap_accuracy"],
            4,
        ),
        "macro_f1": round(
            fed_gen["macro_f1"] - base_gen["macro_f1"], 4
        ),
        "exact_match": round(
            fed_gen["exact_match"] - base_gen["exact_match"], 4
        ),
    }

    base_bs = base_gen["bertscore"]["f1"]
    fed_bs = fed_gen["bertscore"]["f1"]
    if base_bs is not None and fed_bs is not None:
        delta["bertscore_f1"] = round(fed_bs - base_bs, 4)
    else:
        delta["bertscore_f1"] = None

    return delta
