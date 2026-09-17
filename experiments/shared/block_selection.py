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


# Threshold: tensors with <= this many elements are designated "always_on"
# (spec section 7: small params like LayerNorm/bias/gate scalars)
DEFAULT_ALWAYS_ON_THRESHOLD = 4096


def build_group_blocks(
    state: Dict[str, torch.Tensor],
    block_size: int = DEFAULT_BLOCK_SIZE,
    always_on_threshold: int = DEFAULT_ALWAYS_ON_THRESHOLD,
) -> Tuple[GroupBlocks, set]:
    """按 layer / non_layer 分组，并把每个 floating tensor 切成 block。

    返回 (group_blocks, always_on_keys)：
    - group_blocks: 同前，每个 floating tensor 切成 block
    - always_on_keys: numel <= always_on_threshold 的参数 key_name 集合，
      这些参数的所有 block 在 build_selected_blocks 中每轮都会被选中，
      不参与 permutation / slot 轮转。

    always_on 的设计依据：spec 第 7 节——小参数（LayerNorm/bias/门控标量）
    不适合独立参与 H 轮轮转（会导致某些轮次完全没有这类参数更新）。
    """
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
    always_on_keys: set = set()
    for gid, keys in groups_keys.items():
        blocks: List[Tuple[str, int, int]] = []
        for key_name in keys:
            n_elem = state[key_name].numel()
            if n_elem <= always_on_threshold:
                always_on_keys.add(key_name)
            for start in range(0, n_elem, block_size):
                end = min(start + block_size, n_elem)
                blocks.append((key_name, start, end))
        group_blocks[gid] = blocks
    return group_blocks, always_on_keys


def get_rotating_blocks(
    group_blocks: GroupBlocks,
    always_on_keys: set,
) -> GroupBlocks:
    """返回只包含 rotating blocks 的 group_blocks（排除 always_on）。"""
    rotating: GroupBlocks = {}
    for gid, blocks in group_blocks.items():
        rotating[gid] = [
            (kn, s, e) for kn, s, e in blocks if kn not in always_on_keys
        ]
    return rotating


def get_always_on_blocks(
    group_blocks: GroupBlocks,
    always_on_keys: set,
) -> List[Tuple[str, int, int]]:
    """返回 always_on blocks 的 flat list（按 group 顺序）。"""
    always_on: List[Tuple[str, int, int]] = []
    for gid in sorted(group_blocks):
        for kn, s, e in group_blocks[gid]:
            if kn in always_on_keys:
                always_on.append((kn, s, e))
    return always_on


def build_permutations(
    group_blocks: GroupBlocks,
    epoch: int,
    seed: int,
    always_on_keys: Optional[set] = None,
) -> Permutations:
    """为每个 group 构建 Fisher-Yates 排列。

    always_on_keys 中的 key 的 block 不参与排列（它们每轮都选中）。
    只有 rotating blocks 参与排列，确保 H 轮内每个 rotating block 恰好被选一次。
    """
    epoch_seed = hashlib.sha256(f"FedScale-BlockMask-v1|{seed}|{epoch}".encode("utf-8")).digest()
    always_on_keys = always_on_keys or set()
    permutations: Permutations = {}
    for gid, blocks in group_blocks.items():
        # 只对 rotating blocks 做排列
        rotating = [(i, b) for i, b in enumerate(blocks) if b[0] not in always_on_keys]
        n_rotating = len(rotating)
        if n_rotating == 0:
            permutations[gid] = []
            continue
        group_seed = hkdf_group_seed(epoch_seed, gid)
        perm = fisher_yates_shuffle(n_rotating, group_seed)
        # perm 是 rotating-only 的排列；存原始 block indices
        permutations[gid] = [rotating[j][0] for j in perm]
    return permutations


def build_selected_blocks(
    group_blocks: GroupBlocks,
    permutations: Permutations,
    slot: int,
    coverage_h: int = DEFAULT_COVERAGE_H,
    always_on_keys: Optional[set] = None,
) -> SelectedByKey:
    """构建本轮选中的 blocks。

    - rotating blocks: pos % coverage_h == slot 的 blocks（参与 H 轮轮转）
    - always_on blocks: always_on_keys 中 key 的所有 blocks，每轮都选中

    always_on_keys=None 时向后兼容（无 always_on）。
    """
    selected_by_key: SelectedByKey = {}
    # rotating blocks
    for gid, blocks in group_blocks.items():
        perm = permutations.get(gid)
        if perm is None:
            continue
        for pos, block_idx in enumerate(perm):
            if pos % coverage_h == slot:
                key_name, start, end = blocks[block_idx]
                if always_on_keys and key_name in always_on_keys:
                    continue  # always_on 在下面统一处理
                selected_by_key.setdefault(key_name, []).append((start, end))
    # always_on blocks: 每轮全部加入
    if always_on_keys:
        for gid in sorted(group_blocks):
            for key_name, start, end in group_blocks[gid]:
                if key_name in always_on_keys:
                    selected_by_key.setdefault(key_name, []).append((start, end))
    return selected_by_key


def count_selected_elems(selected_by_key: SelectedByKey) -> int:
    return sum(e - s for slices in selected_by_key.values() for s, e in slices)


def recompute_selected_blocks(
    reference_state: Dict[str, torch.Tensor],
    round_idx: int,
    coverage_h: int,
    seed: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    always_on_keys: Optional[set] = None,
    always_on_threshold: int = 0,
) -> SelectedByKey:
    """客户端独立重算 selected blocks（用于验证 server 下发的 plan）。

    给定与 server 相同的 reference_state / coverage_h / seed / block_size，
    独立构建 group_blocks → permutations → selected_blocks，
    不依赖 server 下发的 selected_by_key。

    always_on_threshold > 0 时自动识别小参数为 always_on；
    always_on_keys 非 None 时直接使用（优先于 threshold）。
    """
    if always_on_keys is None and always_on_threshold > 0:
        _, always_on_keys = build_group_blocks(
            reference_state, block_size=block_size, always_on_threshold=always_on_threshold
        )
    group_blocks, _ = build_group_blocks(reference_state, block_size=block_size)
    epoch = (round_idx - 1) // coverage_h
    slot = (round_idx - 1) % coverage_h
    permutations = build_permutations(
        group_blocks, epoch=epoch, seed=seed, always_on_keys=always_on_keys,
    )
    selected = build_selected_blocks(
        group_blocks, permutations, slot=slot,
        coverage_h=coverage_h, always_on_keys=always_on_keys,
    )
    return selected


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


def compute_weighted_block_delta(
    client_block_deltas: List[BlockDelta],
    weights: List[float],
    selected_by_key: SelectedByKey,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> BlockDelta:
    """SCALE-1：只计算加权平均 block delta，不修改 global_state。

    与 apply_block_delta 相同的聚合逻辑，但不需要 global_state 参数，
    也不原地修改任何 state。适用于 server 不常驻 global_state 的场景。
    """
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("weights sum must be positive")
    store_dtype = out_dtype if out_dtype is not None else torch.float16
    aggregated: BlockDelta = {}
    for key_name, slices in selected_by_key.items():
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
            if store_dtype == torch.int8:
                q, scale = _quantize_int8_block(acc)
                out_blocks.append((s, e, q.detach().to(device="cpu").contiguous(), scale))  # type: ignore[arg-type]
            else:
                out_blocks.append(
                    (s, e, acc.detach().to(device="cpu", dtype=store_dtype).contiguous())
                )
        if out_blocks:
            aggregated[key_name] = out_blocks
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


def update_block_memory_with_quant_residual(
    to_send: Dict[str, torch.Tensor],
    selected_by_key: SelectedByKey,
    quant_residual: BlockDelta,
    decay: float,
    *,
    quant_decay: float = 1.0,
    out_dtype: Optional[torch.dtype] = torch.float32,
) -> Dict[str, torch.Tensor]:
    """未选中 block 保留 to_send * decay；选中 block 保留量化 residual * quant_decay。

    两类 residual 语义不同：
    - block-mask residual（未选中）：允许 decay（默认 0.9），避免过期更新长期累积
    - quantization residual（已选中但 INT16 没表达完）：应完整保留，quant_decay=1.0

    residual = true_delta - dequant(quant(true_delta))。
    没有 residual 的选中 block 仍置 0（与 update_block_memory 一致）。
    默认把 memory 存成 FP32，避免 residual 再被 round 回 FP16。
    """
    new_memory: Dict[str, torch.Tensor] = {}
    for key_name, tensor in to_send.items():
        if not tensor.is_floating_point():
            new_memory[key_name] = tensor.clone()
            continue
        flat = tensor.contiguous().view(-1).to(dtype=torch.float32).clone()
        if key_name in selected_by_key:
            for s, e in selected_by_key[key_name]:
                flat[s:e] = 0.0
        flat.mul_(float(decay))
        for item in quant_residual.get(key_name, []):
            s, e, res = int(item[0]), int(item[1]), item[2]
            flat[s:e] = res.to(dtype=torch.float32) * float(quant_decay)
        store_dtype = out_dtype if out_dtype is not None else tensor.dtype
        new_memory[key_name] = flat.to(dtype=store_dtype).view(tensor.shape)
    return new_memory


def _lookup_quant_residual_slice(
    quant_residual: BlockDelta,
    key_name: str,
    start: int,
    end: int,
) -> Optional[torch.Tensor]:
    for item in quant_residual.get(key_name, []):
        if int(item[0]) == int(start) and int(item[1]) == int(end):
            return item[2]
    return None


def merge_quant_residual_memory(
    old_residual: BlockDelta,
    selected_by_key: SelectedByKey,
    new_residual: BlockDelta,
    quant_decay: float = 1.0,
) -> BlockDelta:
    """量化残差独立于 block memory：选中块替换为本轮 residual，未选中块原样保留。"""
    selected_ranges = {
        key: {(int(s), int(e)) for s, e in ranges}
        for key, ranges in selected_by_key.items()
    }
    out: BlockDelta = {}
    for key_name, items in (old_residual or {}).items():
        kept = []
        sel = selected_ranges.get(key_name, set())
        for item in items:
            s, e = int(item[0]), int(item[1])
            if (s, e) in sel:
                continue
            kept.append((s, e, item[2].detach().to(dtype=torch.float32).contiguous()))
        if kept:
            out[key_name] = kept
    for key_name, items in (new_residual or {}).items():
        lst = out.setdefault(key_name, [])
        for item in items:
            s, e, res = int(item[0]), int(item[1]), item[2]
            lst.append(
                (
                    s,
                    e,
                    (res.detach().to(dtype=torch.float32) * float(quant_decay)).contiguous(),
                )
            )
    return out


def update_block_memory_from_states(
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    block_memory: Dict[str, torch.Tensor],
    selected_by_key: SelectedByKey,
    decay: float,
) -> Dict[str, torch.Tensor]:
    """按 FP32 计算 delta+block_memory，未选中块 * decay，选中块置 0。

    不把量化残差混进 block memory，也不在中间 round 回模型 dtype。
    """
    new_memory: Dict[str, torch.Tensor] = {}
    for key_name, tensor in local_state.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            src = (block_memory or {}).get(key_name, tensor)
            new_memory[key_name] = src.clone() if hasattr(src, "clone") else src
            continue
        local_flat = tensor.detach().contiguous().view(-1).to(dtype=torch.float32)
        gv = global_state.get(key_name) if global_state is not None else None
        if gv is not None and gv.numel() == local_flat.numel():
            delta_flat = local_flat - gv.detach().contiguous().view(-1).to(dtype=torch.float32)
        else:
            delta_flat = local_flat
        mem = block_memory.get(key_name) if block_memory is not None else None
        if mem is not None and mem.numel() == delta_flat.numel():
            combined = delta_flat + mem.detach().contiguous().view(-1).to(dtype=torch.float32)
        else:
            combined = delta_flat
        if key_name in selected_by_key:
            for s, e in selected_by_key[key_name]:
                combined[int(s):int(e)] = 0.0
        new_memory[key_name] = (combined * float(decay)).view(tensor.shape)
    return new_memory
