"""S3R12v3 block 级选择 / 编解码 / memory（client & server 共用）。"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Tuple, Union

import torch

from .protocol import DEFAULT_BLOCK_SIZE, DEFAULT_COVERAGE_H, SelectedByKey

GroupBlocks = Dict[str, List[Tuple[str, int, int]]]
Permutations = Dict[str, List[int]]
BlockDelta = Dict[str, List[Tuple[int, int, torch.Tensor]]]

TransferDtypeName = str  # auto | fp16 | float16 | fp32 | float32 | bf16 | bfloat16 | int8


def infer_model_floating_dtype(state: Dict[str, torch.Tensor]) -> torch.dtype:
    for tensor in state.values():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            return tensor.dtype
    return torch.float16


def resolve_transfer_dtype(
    name: TransferDtypeName,
    *,
    ref_dtype: Optional[torch.dtype] = None,
    ref_state: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.dtype:
    """解析通信精度。

    - auto: 跟随 ref_dtype / ref_state 中模型浮点 dtype（通常为 fp16）
    - fp16/fp32/bf16: 强制该精度落盘与传输
    - int8: 预留，当前未实现
    """
    key = (name or "auto").strip().lower()
    if key in ("auto", "model", "native"):
        if ref_dtype is not None:
            return ref_dtype
        if ref_state is not None:
            return infer_model_floating_dtype(ref_state)
        return torch.float16
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if key == "int8":
        return _resolve_int8_dtype()
    if key not in mapping:
        raise ValueError(f"unsupported transfer-dtype: {name}")
    return mapping[key]


def _resolve_int8_dtype() -> torch.dtype:
    # int8 用 torch.int8 存储量化值；scale 由 encode 时 per-block 计算
    # 返回 int8 只是占位标记（实际编码在 encode_int8_block_delta 中处理）
    return torch.int8


def _quantize_int8_block(t: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """对称量化到 int8：返回 (quantized_int8, scale)。

    scale = max(abs(t)) / 127；反量化 = int8 * scale。
    """
    t32 = t.to(dtype=torch.float32)
    amax = float(t32.abs().max().item())
    if amax == 0.0:
        return torch.zeros_like(t32, dtype=torch.int8), 0.0
    scale = amax / 127.0
    q = torch.round(t32 / scale).clamp(-127, 127).to(dtype=torch.int8)
    return q, scale


def _dequantize_int8_block(q: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 0.0:
        return torch.zeros_like(q, dtype=torch.float32)
    return q.to(dtype=torch.float32) * scale


def cast_block_delta(block_delta: BlockDelta, dtype: torch.dtype) -> BlockDelta:
    """把 block_delta 中的 slice 转到指定通信 dtype（计算仍可在别处用 fp32）。

    int8: 对称量化，scale 存为元组第 4 元素 (s, e, q_int8, scale)。
    """
    if dtype == torch.int8:
        out: BlockDelta = {}
        for key_name, blocks in block_delta.items():
            out[key_name] = []
            for s, e, slice_data in blocks:
                q, scale = _quantize_int8_block(slice_data)
                out[key_name].append((s, e, q, scale))  # type: ignore[arg-type]
        return out
    out: BlockDelta = {}  # type: ignore[no-redef]
    for key_name, blocks in block_delta.items():
        out[key_name] = [
            (s, e, slice_data.detach().to(device="cpu", dtype=dtype).contiguous())
            for s, e, slice_data in blocks
        ]
    return out


def _slice_to_fp32(item: Tuple) -> torch.Tensor:
    """把 block slice 元组还原成 fp32（处理 int8 反量化）。"""
    slice_data = item[2]
    if slice_data.dtype == torch.int8:
        scale = float(item[3]) if len(item) > 3 else 0.0
        return _dequantize_int8_block(slice_data, scale)
    return slice_data.to(dtype=torch.float32)


def hkdf_group_seed(epoch_seed: bytes, group_id: str) -> bytes:
    info = b"FedScale-GroupMask-v1|" + group_id.encode("utf-8")
    return hashlib.sha256(epoch_seed + b"\x00" + info).digest()


def fisher_yates_shuffle(n: int, seed: bytes) -> List[int]:
    rng = random.Random(int.from_bytes(seed, "big"))
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def build_group_blocks(state: Dict[str, torch.Tensor], block_size: int = DEFAULT_BLOCK_SIZE) -> GroupBlocks:
    """按 layer / non_layer 分组，并把每个 floating tensor 切成 block。"""
    groups_keys: Dict[str, List[str]] = {}
    for name, tensor in sorted(state.items()):
        if not tensor.is_floating_point():
            continue
        if ".layers." in name:
            ln = int(name.split(".layers.")[1].split(".")[0])
            gid = f"layer_{ln}"
        else:
            gid = "non_layer"
        groups_keys.setdefault(gid, []).append(name)

    group_blocks: GroupBlocks = {}
    for gid, keys in groups_keys.items():
        blocks: List[Tuple[str, int, int]] = []
        for key_name in keys:
            n_elem = state[key_name].numel()
            for start in range(0, n_elem, block_size):
                end = min(start + block_size, n_elem)
                blocks.append((key_name, start, end))
        group_blocks[gid] = blocks
    return group_blocks


def build_permutations(
    group_blocks: GroupBlocks,
    epoch: int,
    seed: int,
) -> Permutations:
    epoch_seed = hashlib.sha256(f"FedScale-BlockMask-v1|{seed}|{epoch}".encode("utf-8")).digest()
    permutations: Permutations = {}
    for gid, blocks in group_blocks.items():
        group_seed = hkdf_group_seed(epoch_seed, gid)
        permutations[gid] = fisher_yates_shuffle(len(blocks), group_seed)
    return permutations


def build_selected_blocks(
    group_blocks: GroupBlocks,
    permutations: Permutations,
    slot: int,
    coverage_h: int = DEFAULT_COVERAGE_H,
) -> SelectedByKey:
    selected_by_key: SelectedByKey = {}
    for gid, blocks in group_blocks.items():
        perm = permutations[gid]
        for pos, block_idx in enumerate(perm):
            if pos % coverage_h == slot:
                key_name, start, end = blocks[block_idx]
                selected_by_key.setdefault(key_name, []).append((start, end))
    return selected_by_key


def count_selected_elems(selected_by_key: SelectedByKey) -> int:
    return sum(e - s for slices in selected_by_key.values() for s, e in slices)


def encode_block_delta(
    to_send: Dict[str, torch.Tensor],
    selected_by_key: SelectedByKey,
    *,
    dtype: Optional[torch.dtype] = None,
) -> BlockDelta:
    """抽取选中 block。dtype 指定通信落盘精度；None 则保持 to_send 原 dtype。

    int8: 对称量化 per-block，scale 存为元组第 4 元素。
    """
    result: BlockDelta = {}
    for key_name, slices in selected_by_key.items():
        if key_name not in to_send or not to_send[key_name].is_floating_point():
            continue
        flat = to_send[key_name].contiguous().view(-1)
        out_dtype = dtype if dtype is not None else to_send[key_name].dtype
        if out_dtype == torch.int8:
            result[key_name] = []
            for s, e in slices:
                q, scale = _quantize_int8_block(flat[s:e])
                result[key_name].append((s, e, q.detach().to(device="cpu").contiguous(), scale))  # type: ignore[arg-type]
        else:
            result[key_name] = [
                (s, e, flat[s:e].detach().to(device="cpu", dtype=out_dtype).contiguous())
                for s, e in slices
            ]
    return result


def add_block_delta(state: Dict[str, torch.Tensor], block_delta: BlockDelta) -> None:
    """原地：state[key][s:e] += delta_slice（支持 int8 反量化）。"""
    for key_name, blocks in block_delta.items():
        if key_name not in state or not state[key_name].is_floating_point():
            continue
        flat = state[key_name].contiguous().view(-1)
        for item in blocks:
            s, e = item[0], item[1]
            slice_fp32 = _slice_to_fp32(item)
            flat[s:e] = (flat[s:e].to(dtype=torch.float32) + slice_fp32).to(
                dtype=state[key_name].dtype
            )
        state[key_name] = flat.view(state[key_name].shape)


def apply_block_delta(
    global_state: Dict[str, torch.Tensor],
    client_block_deltas: List[BlockDelta],
    weights: List[float],
    selected_by_key: SelectedByKey,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> BlockDelta:
    """原地：global += 加权平均(block deltas)；并返回该聚合增量（供客户端增量下发）。

    累加在 fp32 中做；写出的 block_delta 使用 out_dtype（默认跟随 global_state）。
    """
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("weights sum must be positive")
    aggregated: BlockDelta = {}
    for key_name, slices in selected_by_key.items():
        if key_name not in global_state or not global_state[key_name].is_floating_point():
            continue
        gflat = global_state[key_name].contiguous().view(-1)
        store_dtype = out_dtype if out_dtype is not None else global_state[key_name].dtype
        out_blocks: List[Tuple[int, int, torch.Tensor]] = []
        for s, e in slices:
            acc = torch.zeros(e - s, dtype=torch.float32)
            for cid, w in enumerate(weights):
                if cid >= len(client_block_deltas):
                    continue
                blocks = client_block_deltas[cid].get(key_name, [])
                for item in blocks:
                    if item[0] == s and item[1] == e:
                        slice_fp32 = _slice_to_fp32(item)
                        acc.add_(slice_fp32, alpha=float(w) / total_w)
                        break
            gflat[s:e] = (gflat[s:e].to(dtype=torch.float32) + acc).to(dtype=global_state[key_name].dtype)
            # 写出聚合增量：int8 时反量化目标在客户端 add_block_delta 处理，这里仍按 store_dtype
            if store_dtype == torch.int8:
                q, scale = _quantize_int8_block(acc)
                out_blocks.append((s, e, q.detach().to(device="cpu").contiguous(), scale))  # type: ignore[arg-type]
            else:
                out_blocks.append(
                    (s, e, acc.detach().to(device="cpu", dtype=store_dtype).contiguous())
                )
        if out_blocks:
            aggregated[key_name] = out_blocks
        global_state[key_name] = gflat.view(global_state[key_name].shape)
    return aggregated


def update_block_memory(
    to_send: Dict[str, torch.Tensor],
    selected_by_key: SelectedByKey,
    decay: float,
) -> Dict[str, torch.Tensor]:
    """已上传 block 置 0，其余保留，再整体 * decay。"""
    new_memory: Dict[str, torch.Tensor] = {}
    for key_name, tensor in to_send.items():
        if not tensor.is_floating_point():
            new_memory[key_name] = tensor.clone()
            continue
        flat = tensor.contiguous().view(-1).to(dtype=torch.float32).clone()
        if key_name in selected_by_key:
            for s, e in selected_by_key[key_name]:
                flat[s:e] = 0.0
        new_memory[key_name] = (flat * decay).to(dtype=tensor.dtype).view(tensor.shape)
    return new_memory
