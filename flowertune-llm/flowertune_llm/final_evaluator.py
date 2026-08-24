"""Evaluate the final global model on a client-local held-out split.

This module runs in a short-lived, single-GPU Kubernetes Job launched by the
Flower ClientApp's evaluate handler.  It deliberately writes its elapsed time
to a separate artifact: final-model evaluation is a quality result, not part
of the federated training or round-performance timings.
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
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flowertune_llm.evaluator import (
    compute_downstream_metrics,
    compute_validation_metrics,
    create_eval_split,
)
from flowertune_llm.model_state import set_federated_state_dict
from flowertune_llm.models import get_model, get_tokenizer
from flowertune_llm.train_dataset import format_dataset_for_sft, load_data


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
        cache_dir
        / "hub"
        / f"models--{model_name.replace('/', '--')}"
        / "snapshots"
    )
    candidates = sorted(path for path in snapshots.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No local model snapshot found under {snapshots}")
    return str(candidates[0])


def _load_evaluation_model(cfg: DictConfig, device: str):
    """Load the model directly onto the evaluation GPU.

    bitsandbytes models should be placed with ``device_map`` at load time;
    moving a 4-bit model afterwards with ``Module.to`` is unsupported in some
    versions of Transformers/bitsandbytes.
    """
    quantization = int(cfg.quantization)
    quantization_config = None
    if quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != 0:
        raise ValueError(f"Unsupported evaluation quantization: {quantization}")

    # The federated full-model state is exchanged in FP16.  Loading the base
    # model directly on the evaluator GPU in the same dtype avoids first
    # materialising an FP32 3B model in host memory and then duplicating it in
    # ``model.to(device)``.
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


def main() -> None:
    started = time.perf_counter()
    output_path = os.environ["EVALUATION_OUTPUT_PATH"]
    device = "cuda:0"
    cfg = _config()

    state = torch.load(
        os.environ["MODEL_STATE_PATH"],
        map_location="cpu",
        weights_only=True,
    )
    model = _load_evaluation_model(cfg, device)
    set_federated_state_dict(model, cfg, state)
    del state

    tokenizer = get_tokenizer(cfg.name)
    if tokenizer.pad_token is None:
        # OpenLLaMA has no dedicated padding token.  Evaluation batches are
        # padded for the validation forward pass, so reuse EOS exactly as the
        # training data collator does for this causal-LM setup.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    dataset = load_data(
        int(os.environ["PARTITION_ID"]),
        int(os.environ["NUM_PARTITIONS"]),
        os.environ["DATASET_NAME"],
    )
    dataset = format_dataset_for_sft(dataset, tokenizer, cfg.name)
    max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", "0") or 0)
    if max_train_samples > 0 and len(dataset) > max_train_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_train_samples))
    if "text" not in dataset.column_names:
        raise ValueError(
            "Final evaluation requires the federated dataset to contain a text column"
        )

    _, evalset = create_eval_split(
        dataset,
        split_ratio=float(os.environ.get("EVAL_SPLIT_RATIO", "0.1")),
    )
    max_samples = int(os.environ.get("NUM_EVAL_SAMPLES", "50") or 50)
    validation = compute_validation_metrics(
        model, tokenizer, evalset, device, max_samples=max_samples
    )
    downstream = compute_downstream_metrics(
        model, tokenizer, evalset, device, max_samples=max_samples
    )
    evaluation_seconds = round(time.perf_counter() - started, 4)

    payload = {
        "status": "completed",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "partition_id": int(os.environ["PARTITION_ID"]),
        "evaluated_samples": min(len(evalset), max_samples),
        "evaluation_seconds": evaluation_seconds,
        "validation": validation,
        "downstream": downstream,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Keep the precise failure on the shared PVC even when Kubernetes marks
        # the Job failed and the ClientApp removes the Job object.
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
