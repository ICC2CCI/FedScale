"""S3R12v3 block 级选择 / 编解码 / memory（client & server 共用）。"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Tuple

import torch

from .protocol import DEFAULT_BLOCK_SIZE, DEFAULT_COVERAGE_H, SelectedByKey

GroupBlocks = Dict[str, List[Tuple[str, int, int]]]
Permutations = Dict[str, List[int]]
BlockDelta = Dict[str, List[Tuple[int, int, torch.Tensor]]]


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


def encode_block_delta(to_send: Dict[str, torch.Tensor], selected_by_key: SelectedByKey) -> BlockDelta:
    result: BlockDelta = {}
    for key_name, slices in selected_by_key.items():
        if key_name not in to_send or not to_send[key_name].is_floating_point():
            continue
        flat = to_send[key_name].contiguous().view(-1)
        result[key_name] = [(s, e, flat[s:e].detach().cpu().clone()) for s, e in slices]
    return result


def apply_block_delta(
    global_state: Dict[str, torch.Tensor],
    client_block_deltas: List[BlockDelta],
    weights: List[float],
    selected_by_key: SelectedByKey,
) -> None:
    """原地：global += 加权平均(block deltas)。"""
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("weights sum must be positive")
    for key_name, slices in selected_by_key.items():
        if key_name not in global_state or not global_state[key_name].is_floating_point():
            continue
        gflat = global_state[key_name].contiguous().view(-1)
        for s, e in slices:
            acc = torch.zeros(e - s, dtype=torch.float32)
            for cid, w in enumerate(weights):
                if cid >= len(client_block_deltas):
                    continue
                blocks = client_block_deltas[cid].get(key_name, [])
                for ss, ee, slice_data in blocks:
                    if ss == s and ee == e:
                        acc.add_(slice_data.to(dtype=torch.float32), alpha=float(w) / total_w)
                        break
            gflat[s:e] = (gflat[s:e].to(dtype=torch.float32) + acc).to(dtype=global_state[key_name].dtype)
        global_state[key_name] = gflat.view(global_state[key_name].shape)


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
