"""Federated final evaluation: integrate all evaluation modules.

This module is the improved replacement for the legacy ``final_evaluator.py``.
It runs in a short-lived, single-GPU Kubernetes Job launched by the Flower
ClientApp's evaluate handler.

Improvements over the legacy evaluator:

1. **P0 fix**: Uses offset_mapping for precise assistant-only loss masking.
2. **P0 fix**: BERTScore returns ``None`` on ImportError instead of 0.0.
3. **P1**: Left padding for decoder-only model generation.
4. **P1**: Base-model baseline evaluation and federated-vs-base delta.
5. **P1**: Three-way train/validation/test split (uses test split for final eval).
6. **P2**: Optional MMLU/MMLU-Pro external accuracy evaluation.
7. **P2**: Compression report (bytes/round, quantization error).
8. **P2**: DP report (RDP accounting for privacy budget).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluation.data_split import create_train_val_test_split
from evaluation.validation_metrics import compute_validation_metrics
from evaluation.generation_metrics import compute_generation_metrics
from evaluation.baseline_comparison import evaluate_with_baseline, compute_delta
from evaluation.compression_report import compute_compression_report
from evaluation.dp_report import compute_rdp_epsilon


def _config() -> DictConfig:
    return DictConfig(
        {
            "name": os.environ["MODEL_NAME"],
            "finetuning_type": os.environ.get("FINETUNING_TYPE", "lora"),
            "quantization": int(os.environ.get("QUANTIZATION", "4")),
            "gradient_checkpointing": False,
            "lora": {
                "peft_lora_r": int(os.environ.get("LORA_R", "32")),
                "peft_lora_alpha": int(os.environ.get("LORA_ALPHA", "64")),
            },
        }
    )


def _model_path(model_name: str) -> str:
    cache_dir = Path(os.environ.get("HF_HOME", "/app/.cache/huggingface"))
    snapshots = (
        cache_dir / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    )
    candidates = sorted(path for path in snapshots.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No local model snapshot found under {snapshots}")
    return str(candidates[0])


def _load_evaluation_model(cfg: DictConfig, device: str):
    """Load the model directly onto the evaluation GPU."""
    quantization = int(cfg.quantization)
    quantization_config = None
    if quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != 0:
        raise ValueError(f"Unsupported evaluation quantization: {quantization}")

    model = AutoModelForCausalLM.from_pretrained(
        _model_path(cfg.name),
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if str(cfg.finetuning_type).lower() == "lora":
        if quantization in {4, 8}:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora.peft_lora_r,
                lora_alpha=cfg.lora.peft_lora_alpha,
                lora_dropout=0.075,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    return model


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        _model_path(model_name),
        use_fast=True,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def main() -> None:
    started = time.perf_counter()
    output_path = os.environ["EVALUATION_OUTPUT_PATH"]
    device = "cuda:0"
    cfg = _config()

    # --- Load federated model state ---
    state = torch.load(
        os.environ["MODEL_STATE_PATH"],
        map_location="cpu",
        weights_only=True,
    )
    model = _load_evaluation_model(cfg, device)

    from flowertune_llm.model_state import set_federated_state_dict
    set_federated_state_dict(model, cfg, state)
    del state

    tokenizer = _load_tokenizer(cfg.name)
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Load and split data ---
    from flowertune_llm.train_dataset import format_dataset_for_sft, load_data
    dataset = load_data(
        int(os.environ["PARTITION_ID"]),
        int(os.environ["NUM_PARTITIONS"]),
        os.environ["DATASET_NAME"],
    )
    dataset = format_dataset_for_sft(dataset, tokenizer, cfg.name)
    if "text" not in dataset.column_names:
        raise ValueError("Final evaluation requires a text column")

    val_ratio = float(os.environ.get("EVAL_VAL_RATIO", "0.05"))
    test_ratio = float(os.environ.get("EVAL_TEST_RATIO", "0.05"))
    split = create_train_val_test_split(
        dataset, val_ratio=val_ratio, test_ratio=test_ratio,
    )
    # Use the test split for final evaluation (not the validation split).
    evalset = split.test
    max_samples = int(os.environ.get("NUM_EVAL_SAMPLES", "50") or 50)
    max_length = int(os.environ.get("EVAL_MAX_LENGTH", "512") or 512)
    max_new_tokens = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "128") or 128)

    # --- Run evaluation ---
    skip_base = os.environ.get("EVAL_SKIP_BASE", "false").strip().lower() == "true"

    results = evaluate_with_baseline(
        model_name=cfg.name,
        federated_model=model,
        tokenizer=tokenizer,
        evalset=evalset,
        device=device,
        max_samples=max_samples,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        quantization=int(cfg.quantization),
        local_files_only=True,
        skip_base=skip_base,
    )

    # --- Optional MMLU evaluation ---
    mmlu_result = None
    mmlu_dataset = os.environ.get("EVAL_MMLU_DATASET", "")
    if mmlu_dataset:
        from evaluation.mmlu_evaluator import evaluate_mmlu
        is_pro = "mmlu-pro" in mmlu_dataset.lower()
        try:
            mmlu_result = evaluate_mmlu(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dataset_name=mmlu_dataset,
                max_samples=int(os.environ.get("EVAL_MMLU_SAMPLES", "100")),
                is_mmlu_pro=is_pro,
            )
        except Exception as exc:
            mmlu_result = {"error": f"{type(exc).__name__}: {exc}"}

    # --- Optional compression report ---
    compression_result = None
    if os.environ.get("EVAL_COMPRESSION_REPORT", "false").strip().lower() == "true":
        from flowertune_llm.model_state import get_federated_state_dict
        fed_state = get_federated_state_dict(model, cfg)
        block_size = int(os.environ.get("FEDSCALE_BLOCK_SIZE", "1048576"))
        compression_result = [
            {
                "config_name": r.config_name,
                "original_bytes": r.original_bytes,
                "compressed_bytes": r.compressed_bytes,
                "compression_ratio": r.compression_ratio,
                "quantization_error_l2": r.quantization_error_l2,
                "quantization_error_relative": r.quantization_error_relative,
                "mask_ratio": r.mask_ratio,
                "int8_enabled": r.int8_enabled,
                "residual_enabled": r.residual_enabled,
                "extra": r.extra,
            }
            for r in compute_compression_report(fed_state, block_size=block_size)
        ]
        del fed_state

    # --- Optional DP report ---
    dp_result = None
    dp_sigma = os.environ.get("EVAL_DP_SIGMA_CLIENT", "")
    if dp_sigma:
        rdp = compute_rdp_epsilon(
            num_rounds=int(os.environ.get("EVAL_DP_NUM_ROUNDS", "50")),
            clip_norm=float(os.environ.get("EVAL_DP_CLIP_NORM", "1.0")),
            sigma_client=float(dp_sigma),
            cohort_size=int(os.environ.get("EVAL_DP_COHORT_SIZE", "2")),
            delta=float(os.environ.get("EVAL_DP_DELTA", "1e-5")),
        )
        dp_result = {
            "epsilon": rdp.epsilon,
            "delta": rdp.delta,
            "alpha_optimal": rdp.alpha_optimal,
            "rdp_per_round": rdp.rdp_per_round,
            "clip_norm": rdp.clip_norm,
            "sigma_client": rdp.sigma_client,
            "min_cohort_size": rdp.min_cohort_size,
            "parameters": rdp.parameters,
        }

    evaluation_seconds = round(time.perf_counter() - started, 4)

    payload = {
        "status": "completed",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "partition_id": int(os.environ["PARTITION_ID"]),
        "split_info": split.split_info,
        "evaluation_seconds": evaluation_seconds,
        "federated": results.get("federated"),
        "base": results.get("base"),
        "delta": results.get("delta"),
        "mmlu": mmlu_result,
        "compression": compression_result,
        "dp": dp_result,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure_path = os.environ.get("EVALUATION_OUTPUT_PATH")
        failure = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        if failure_path:
            try:
                with open(failure_path, "w", encoding="utf-8") as handle:
                    json.dump(failure, handle, ensure_ascii=False, indent=2)
            except Exception as write_exc:
                print(f"Failed to write evaluation failure artifact: {write_exc}")
        import traceback
        traceback.print_exc()
        raise
