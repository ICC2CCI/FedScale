"""Utilities for resuming federated LoRA or full-model training."""

from pathlib import Path

import torch
from flwr.app import ArrayRecord
from peft import get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from flowertune_llm.models import _model_source, build_lora_config, get_model
from flowertune_llm.model_state import (
    finetuning_type,
    get_federated_state_dict,
    set_federated_state_dict,
)


def get_resume_peft_path(cfg):
    """Return the configured PEFT checkpoint path, or None if resume is disabled."""
    path = getattr(cfg.train, "resume_peft_path", "")
    if path is None:
        return None

    path = str(path).strip()
    return path or None


def get_resume_round(cfg) -> int:
    """Return the completed global round represented by the resume checkpoint."""
    return int(getattr(cfg.train, "resume_round", 0) or 0)


def _meta_lora_model(model_cfg):
    """Build the adapter graph without materializing the frozen base weights."""
    model_source = _model_source(model_cfg)
    model_config = AutoConfig.from_pretrained(
        str(model_source), local_files_only=True
    )
    with torch.device("meta"):
        base_model = AutoModelForCausalLM.from_config(model_config)
    return get_peft_model(base_model, build_lora_config(model_cfg))


def _materialize_lora_parameters(model) -> None:
    """Allocate and initialise only LoRA A/B matrices on CPU.

    A 14B FP16 base needs roughly 28 GiB merely to construct the ServerApp's
    initial ArrayRecord.  The federated payload contains only LoRA tensors, so
    keep the frozen architecture on ``meta`` and materialize just its adapters.
    """
    adapter_count = 0
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        for adapter_name in lora_a.keys():
            lora_a[adapter_name].to_empty(device="cpu")
            lora_b[adapter_name].to_empty(device="cpu")
            module.reset_lora_parameters(adapter_name)
            adapter_count += 1
    if adapter_count == 0:
        raise RuntimeError("LoRA meta model did not expose any adapter matrices")


def _load_lora_arrays_without_base(model_cfg, adapter_state=None) -> ArrayRecord:
    model = _meta_lora_model(model_cfg)
    _materialize_lora_parameters(model)
    if adapter_state is not None:
        set_federated_state_dict(model, model_cfg, adapter_state)
    state = get_federated_state_dict(model, model_cfg)
    if not state or any(value.is_meta for value in state.values()):
        raise RuntimeError("LoRA initial state was not materialized on CPU")
    return ArrayRecord(state)


def load_initial_arrays(model_cfg, resume_peft_path=None) -> ArrayRecord:
    """Create initial Flower arrays from a fresh or durable checkpoint."""
    tuning = finetuning_type(model_cfg)

    if not resume_peft_path:
        if tuning == "lora":
            print("Starting fresh LoRA adapters without loading frozen base weights")
            return _load_lora_arrays_without_base(model_cfg)
        model = get_model(model_cfg)
        print(f"Starting training from fresh initial {tuning} weights")
        return ArrayRecord(get_federated_state_dict(model, model_cfg))

    checkpoint_dir = Path(resume_peft_path)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {checkpoint_dir}")

    if tuning == "full":
        print(f"Resuming full-model training from: {checkpoint_dir}")
        state_path = checkpoint_dir / "model_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Full-model federated state not found: {state_path}"
            )
        # A FedScale run stores its canonical-block INT8 global delta here;
        # ServerApp validates and reconstructs it against the verified base.
        # An uncompressed full-model run stores the dense federated state.
        return ArrayRecord(torch.load(state_path, map_location="cpu", weights_only=True))

    adapter_model_path = checkpoint_dir / "adapter_model.bin"
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_model_path.is_file():
        raise FileNotFoundError(f"PEFT adapter weights not found: {adapter_model_path}")
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"PEFT adapter config not found: {adapter_config_path}")
    print(f"Resuming LoRA training from: {checkpoint_dir}")
    adapter_state = torch.load(adapter_model_path, map_location="cpu")
    return _load_lora_arrays_without_base(model_cfg, adapter_state)


def absolute_round(server_round: int, resume_round: int) -> int:
    """Translate the current run round to the original global round number."""
    return resume_round + server_round
