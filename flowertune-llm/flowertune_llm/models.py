# Configure HuggingFace BEFORE any imports
# Disable hf_transfer to avoid XET protocol issues with mirrors
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import math
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from flowertune_llm.model_state import finetuning_type


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_tokenizer(model_name: str):
    """Load tokenizer with local_files_only=True for offline environment.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        tokenizer
    """
    print(f"\n{'='*60}")
    print(f"Loading tokenizer for: {model_name}")
    print(f"{'='*60}")

    # Print HuggingFace environment configuration
    print(f"\nHuggingFace Environment:")
    print(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'not set')}")
    print(f"  HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', 'not set')}")
    print(f"  HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'not set')}")

    try:
        model_path_to_load = model_name
        cache_dir = os.environ.get("HF_HOME", "/app/.cache/huggingface")
        model_cache_path = Path(cache_dir) / "hub" / f"models--{model_name.replace('/', '--')}"
        snapshots_path = model_cache_path / "snapshots"
        if snapshots_path.exists():
            snapshot_dirs = list(snapshots_path.iterdir())
            if snapshot_dirs:
                model_path_to_load = str(snapshot_dirs[0])
                print(f"  Using absolute cache path: {model_path_to_load}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path_to_load,
            use_fast=True,
            padding_side="right",
            legacy=False,
            local_files_only=True,
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Model max length: {tokenizer.model_max_length}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  BOS token: {tokenizer.bos_token}")
        print(f"{'='*60}\n")

        return tokenizer
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {type(e).__name__}: {str(e)[:300]}")
        raise


def _model_source(model_cfg, source_path=None):
    """Resolve a local model snapshot or an explicit full checkpoint."""
    if source_path is not None:
        source = Path(source_path)
        if not source.is_dir():
            raise FileNotFoundError(f"Model checkpoint directory not found: {source}")
        return source

    cache_dir = Path(os.environ.get("HF_HOME", "/app/.cache/huggingface"))
    snapshots_path = (
        cache_dir
        / "hub"
        / f"models--{model_cfg.name.replace('/', '--')}"
        / "snapshots"
    )
    snapshot_dirs = sorted(path for path in snapshots_path.glob("*") if path.is_dir())
    if not snapshot_dirs:
        raise FileNotFoundError(
            f"No local snapshot found for {model_cfg.name} under {snapshots_path}"
        )
    incomplete_files = list(
        snapshots_path.parent.joinpath("blobs").glob("*.incomplete")
    )
    if incomplete_files:
        raise RuntimeError(
            f"Model cache for {model_cfg.name} is incomplete: "
            f"{len(incomplete_files)} unfinished blob(s)"
        )
    return snapshot_dirs[0]


def build_lora_config(model_cfg) -> LoraConfig:
    """Return the one LoRA adapter layout used by clients and ServerApp."""
    return LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        # Keep target selection explicit for Qwen2/Qwen3 with the pinned PEFT
        # version; these projection names also match OpenLLaMA.
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )


def get_model(model_cfg: DictConfig, source_path=None):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    tuning = finetuning_type(model_cfg)
    quantization = int(model_cfg.get("quantization", 4))
    if tuning == "full" and quantization != 0:
        raise ValueError("Full fine-tuning requires model.quantization=0 (FP16)")

    quantization_config = None
    if quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != 0:
        raise ValueError(
            f"Use quantization 0, 4, or 8. You passed: {quantization}"
        )

    # Resolve the snapshot explicitly. Transformers 4.53 can miss a valid
    # Hugging Face cache on the shared PVC when loading by repository name.
    # Training must never mutate or redownload the pre-warmed model cache.
    model_path_to_load = _model_source(model_cfg, source_path)

    print(f"  Using absolute cache path: {model_path_to_load}")
    print("  Local files only: True")
    model_dtype = torch.float32 if tuning == "full" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path_to_load),
        quantization_config=quantization_config,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print("✓ Model loaded successfully from local cache")

    gradient_checkpointing = bool(
        model_cfg.get("gradient_checkpointing", False)
    )
    if tuning == "full":
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        return model

    if quantization in {4, 8}:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        # LoRA freezes the base model.  Re-entrant activation checkpointing
        # needs at least one input to require gradients, otherwise the
        # checkpointed decoder layers return a detached loss and backward()
        # fails on the first optimizer step.  prepare_model_for_kbit_training
        # performs this for 4/8-bit paths; FP16 LoRA needs it explicitly.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return get_peft_model(model, build_lora_config(model_cfg))
