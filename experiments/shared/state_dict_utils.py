"""State dict 工具：差值 / 累加 / FSDP 全量提取与加载。"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def cpu_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().to(device="cpu") for k, v in state.items()}


def sub_state(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, av in a.items():
        bv = b.get(k)
        if bv is None:
            out[k] = av.clone() if hasattr(av, "clone") else av
        elif av.is_floating_point():
            out[k] = (av.to(dtype=torch.float32) - bv.to(dtype=torch.float32)).to(dtype=av.dtype)
        else:
            out[k] = av.clone()
    return out


def add_state(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, av in a.items():
        bv = b.get(k)
        if bv is None:
            out[k] = av.clone() if hasattr(av, "clone") else av
        elif av.is_floating_point():
            out[k] = (av.to(dtype=torch.float32) + bv.to(dtype=torch.float32)).to(dtype=av.dtype)
        else:
            out[k] = av.clone()
    return out


def zero_state_like(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) if v.is_floating_point() else v.clone() for k, v in state.items()}


def floating_elem_count(state: Dict[str, torch.Tensor]) -> int:
    return sum(v.numel() for v in state.values() if v.is_floating_point())


def _as_fsdp(model: torch.nn.Module):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:
        return None
    if isinstance(model, FSDP):
        return model
    for mod in model.modules():
        if isinstance(mod, FSDP):
            return mod
    return None


def broadcast_object(obj, src: int = 0):
    """在已初始化的 process group 上广播任意可 pickle 对象。"""
    import torch.distributed as dist

    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def get_full_state_fsdp(model: torch.nn.Module) -> Optional[Dict[str, torch.Tensor]]:
    """FSDP FULL_STATE_DICT；rank0_only 时非 0 号进程可能得到空 dict。"""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    fsdp_model = _as_fsdp(model)
    if fsdp_model is not None:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with fsdp_model.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state = fsdp_model.state_dict()
        if not state:
            return None
        return cpu_state(state)
    return cpu_state(model.state_dict())


def load_full_state_fsdp(model: torch.nn.Module, state: Optional[Dict[str, torch.Tensor]]) -> None:
    """加载完整 state。

    对 FSDP：要求 **每个 rank 都持有完整 state**（先 broadcast），再用
    rank0_only=False 的 FULL_STATE_DICT 上下文加载。空 dict 会导致 Missing key。
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    if state is None:
        raise ValueError("load_full_state_fsdp requires a full state dict on every rank")

    fsdp_model = _as_fsdp(model)
    if fsdp_model is not None:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with fsdp_model.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            fsdp_model.load_state_dict(state)
        return

    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device=device) for k, v in state.items()}, strict=True)
