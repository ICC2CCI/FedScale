# Configure HuggingFace BEFORE any imports
# Disable hf_transfer to avoid XET protocol issues with mirrors
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import math
import shutil
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    distributed_strategy = str(
        model_cfg.get("distributed_strategy", "ddp")
    ).lower()
    ddp_cpu_offload = bool(model_cfg.get("ddp_cpu_offload", False))
    tuning = finetuning_type(model_cfg)
    quantization = int(model_cfg.get("quantization", 4))
    if tuning == "full" and quantization != 0:
        raise ValueError("Full fine-tuning requires model.quantization=0 (FP16)")

    quantization_config = None
    if quantization == 4:
        if distributed_strategy == "fsdp":
            # FSDP can only shard floating-point tensors. bitsandbytes keeps the
            # values quantized to 4-bit, but exposes the packed storage as FP16
            # so FSDP can flatten and shard it. FP16 is used instead of BF16
            # because the training workers use V100 GPUs.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage=torch.float16,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != 0:
        raise ValueError(
            f"Use quantization 0, 4, or 8. You passed: {quantization}"
        )

    # Try to load model from local cache only (training pods have no network access)
    max_retries = 1
    for attempt in range(max_retries):
        try:
            # Always use local cache only
            local_only = True

            # Check if we should use absolute path from cache (workaround for Transformers 4.53.0 + CFS)
            cache_dir = os.environ.get("HF_HOME", "/app/.cache/huggingface")
            model_cache_path = Path(cache_dir) / "hub" / f"models--{model_cfg.name.replace('/', '--')}"

            # If cache exists, use absolute path to avoid Transformers cache detection issues
            model_path_to_load = model_cfg.name
            if model_cache_path.exists():
                snapshots_path = model_cache_path / "snapshots"
                if snapshots_path.exists():
                    snapshot_dirs = list(snapshots_path.iterdir())
                    if snapshot_dirs:
                        latest_snapshot = snapshot_dirs[0]
                        model_path_to_load = str(latest_snapshot)
                        print(f"  Using absolute cache path: {model_path_to_load}")

            print(f"\n{'='*60}")
            print(f"[Attempt {attempt+1}/{max_retries}] Loading model from cache...")
            print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'not set')}")
            print(f"HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', 'not set')}")
            print(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
            print(f"Local files only: {local_only}")
            print(f"Model path: {model_path_to_load}")
            print(f"{'='*60}\n")

            # Normal full tuning keeps FP32 master parameters and uses FP16 AMP.
            # The explicit DDP CPU-offload baseline is different: DDP otherwise
            # keeps an FP32 model replica plus an equally large reduction bucket
            # on every V100 and exhausts 32 GiB before the first forward pass.
            # Its optimizer already maintains a CPU mirror, so keep only FP16
            # parameters/gradients/reduction buckets on the GPU.
            model_dtype = (
                torch.float16
                if tuning != "full" or ddp_cpu_offload
                else torch.float32
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path_to_load,
                quantization_config=quantization_config,
                # For FSDP-QLoRA this must match bnb_4bit_quant_storage.
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_only,
                # Don't force safetensors - let transformers auto-detect format
            )
            print("✓ Model loaded successfully")
            break  # Success
        except (OSError, AttributeError, ConnectionError, TimeoutError) as e:
            error_msg = str(e)
            error_type = type(e).__name__

            print(f"✗ Model loading failed ({error_type})")
            print(f"Error: {error_msg[:500]}")
            print(f"\nPlease check:")
            print(f"  1. Model cache exists at: {os.environ.get('HF_HOME', '/app/.cache/huggingface')}")
            print(f"  2. Model files are complete (no .incomplete files)")
            print(f"  3. PVC is properly mounted")
            raise

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

    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        # Explicitly select the common attention projections.  PEFT 0.6.2
        # predates Qwen3's model_type and cannot infer its targets reliably.
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, peft_config)
