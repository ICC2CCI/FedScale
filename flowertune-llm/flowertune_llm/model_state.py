"""State-dict and checkpoint helpers shared by LoRA and full fine-tuning."""

from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


FULL_LOCAL_INIT_KEY = "__flower_full_local_initialization__"
SPARSE_DELTA_FORMAT_KEY = "__flower_sparse_delta_topk_int8__"
SPARSE_DELTA_INDEX_SUFFIX = ".__topk_index"
SPARSE_DELTA_VALUE_SUFFIX = ".__topk_value"
SPARSE_DELTA_SCALE_SUFFIX = ".__topk_scale"


def finetuning_type(model_cfg) -> str:
    """Return the validated federated fine-tuning type."""
    value = str(model_cfg.get("finetuning_type", "lora")).strip().lower()
    if value not in {"lora", "full"}:
        raise ValueError(
            "model.finetuning-type must be either 'lora' or 'full', "
            f"got {value!r}"
        )
    return value


def get_federated_state_dict(model, model_cfg, state_dict=None):
    """Return exactly the tensors exchanged and averaged by Flower."""
    if finetuning_type(model_cfg) == "lora":
        return get_peft_model_state_dict(model, state_dict=state_dict)
    source = model.state_dict() if state_dict is None else state_dict
    # Full tuning uses FP32 master parameters for stable optimization, but the
    # federation boundary exchanges FP16 tensors. This halves each OpenLLaMA
    # update from ~13.7 GB to ~6.85 GB and keeps two-client FedAvg within the
    # existing centre/client memory envelopes. Loading into an FP32 model on
    # the next round casts values back to its parameter dtype.
    return {
        key: (
            value.detach().to(device="cpu", dtype=torch.float16)
            if value.is_floating_point()
            else value.detach().to(device="cpu")
        )
        for key, value in source.items()
    }


def set_federated_state_dict(model, model_cfg, state_dict) -> None:
    """Load a Flower state dict into a LoRA or full model."""
    if finetuning_type(model_cfg) == "lora":
        set_peft_model_state_dict(model, state_dict)
        return
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Full-model state mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def encode_topk_int8_delta(final_state, initial_state, ratio: float):
    """Encode a full-model update as per-tensor Top-K INT8 deltas.

    The federation transport is the bottleneck in the cross-cloud topology:
    uploading an entire OpenLLaMA FP16 state takes hours.  This representation
    keeps only the largest update entries of each floating-point tensor.  Each
    selected value is symmetric-INT8 quantized with a per-tensor scale; indices
    are flattened int32 offsets.  Non-floating buffers are deliberately omitted
    because they are unchanged by the full-tuning optimizer in this workload.
    """
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("Top-K delta ratio must be in (0, 1]")

    encoded = {
        SPARSE_DELTA_FORMAT_KEY: torch.tensor([1], dtype=torch.uint8),
    }
    selected_total = 0
    source_total = 0
    for key, final_value in final_state.items():
        if not final_value.is_floating_point():
            continue
        initial_value = initial_state.get(key)
        if initial_value is None:
            raise KeyError(f"Initial full-model state is missing {key!r}")
        if tuple(initial_value.shape) != tuple(final_value.shape):
            raise ValueError(f"State shape changed for {key!r}")

        flat_delta = (
            final_value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            - initial_value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        )
        numel = flat_delta.numel()
        if numel == 0:
            continue
        count = min(numel, max(1, int(numel * float(ratio))))
        # topk is deterministic for a fixed tensor and avoids materializing a
        # dense mask the size of the model.
        _, indices = torch.topk(flat_delta.abs(), k=count, sorted=False)
        values = flat_delta.index_select(0, indices)
        max_abs = values.abs().max()
        scale = max_abs / 127.0 if max_abs.item() > 0.0 else torch.tensor(1.0)
        quantized = torch.clamp(torch.round(values / scale), -127, 127).to(torch.int8)
        encoded[key + SPARSE_DELTA_INDEX_SUFFIX] = indices.to(torch.int32)
        encoded[key + SPARSE_DELTA_VALUE_SUFFIX] = quantized
        encoded[key + SPARSE_DELTA_SCALE_SUFFIX] = scale.reshape(1).to(torch.float32)
        selected_total += count
        source_total += numel

    encoded["__flower_sparse_selected_values__"] = torch.tensor(
        [selected_total], dtype=torch.int64
    )
    encoded["__flower_sparse_source_values__"] = torch.tensor(
        [source_total], dtype=torch.int64
    )
    return encoded


def decode_topk_int8_delta(base_state, encoded_state):
    """Reconstruct a dense state from a sparse INT8 state relative to base."""
    if not is_topk_int8_delta(encoded_state):
        raise ValueError("Expected a Top-K INT8 delta state")

    decoded = {
        key: value.clone() if hasattr(value, "clone") else value
        for key, value in base_state.items()
    }
    for key, target in decoded.items():
        if not hasattr(target, "is_floating_point") or not target.is_floating_point():
            continue
        index_key = key + SPARSE_DELTA_INDEX_SUFFIX
        value_key = key + SPARSE_DELTA_VALUE_SUFFIX
        scale_key = key + SPARSE_DELTA_SCALE_SUFFIX
        if index_key not in encoded_state:
            continue
        indices = encoded_state[index_key].to(device=target.device, dtype=torch.long)
        quantized = encoded_state[value_key].to(device=target.device, dtype=torch.float32)
        scale = encoded_state[scale_key].reshape(-1)[0].to(
            device=target.device, dtype=torch.float32
        )
        target.reshape(-1).index_add_(
            0, indices, (quantized * scale).to(dtype=target.dtype)
        )
    return decoded


def is_topk_int8_delta(state_dict) -> bool:
    """Return whether *state_dict* is a Top-K INT8 federated update."""
    return SPARSE_DELTA_FORMAT_KEY in state_dict


def checkpoint_name(round_number: int, model_cfg) -> str:
    """Return the durable checkpoint directory name for one global round."""
    prefix = "peft" if finetuning_type(model_cfg) == "lora" else "full"
    return f"{prefix}_{round_number}"


def checkpoint_path(save_path, round_number: int, model_cfg) -> str:
    return str(Path(save_path) / checkpoint_name(round_number, model_cfg))
