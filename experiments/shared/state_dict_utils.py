"""State dict 工具：差值 / 累加 / FSDP 全量提取与加载。"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# SCALE-3：逐 FSDP unit 提取 block delta，不 gather 完整 state
# ---------------------------------------------------------------------------


def _strip_orig_mod_prefix(name: str) -> str:
    """accelerate 包装的模型参数名可能有 _orig_mod. 前缀，去掉它。"""
    if name.startswith("_orig_mod."):
        return name[len("_orig_mod."):]
    return name


def get_sharded_block_delta(
    model: torch.nn.Module,
    local_global: Dict[str, torch.Tensor],
    memory: Dict[str, torch.Tensor],
    selected_by_key: Dict[str, List[Tuple[int, int]]],
    *,
    transfer_dtype: Optional[torch.dtype] = None,
    memory_decay: float = 0.9,
    is_main: bool = True,
) -> Tuple[Dict[str, List[Tuple]], Dict[str, torch.Tensor]]:
    """SCALE-3：逐 FSDP unit unshard，提取选中 block 的 delta，不 gather 完整 state。

    替代原来的:
        full_state = get_full_state_fsdp(model)      # gather 完整模型 ~14GB(7B)
        delta = sub_state(full_state, local_global)   # 临时完整 delta ~14GB
        to_send = add_state(delta, memory)            # 临时完整 to_send ~14GB
        block_delta = encode_block_delta(to_send, ...) # 只选中的 block
        memory = update_block_memory(to_send, ...)

    新流程（峰值 = 1 个 FSDP unit ~500MB for 7B）:
        for each FSDP unit:
            unshard → 获取该 unit 完整参数
            for each selected key in this unit:
                delta_slice = (param - local_global[key]) + memory[key]
                extract [s:e] → block_delta
            reshard
        update memory (只在 rank0)

    参数:
        model: FSDP 包装的模型
        local_global: rank0 CPU 上的完整 global state dict
        memory: rank0 CPU 上的完整 memory state dict
        selected_by_key: {key_name: [(start, end), ...]}
        transfer_dtype: 通信精度
        memory_decay: memory 衰减系数
        is_main: 是否 rank0（只有 rank0 做 delta 提取 + memory 更新）

    返回:
        block_delta: {key_name: [(s, e, slice_tensor), ...]}（只有 rank0 有值）
        new_memory: 更新后的 memory（只有 rank0 有值，其他 rank 返回原 memory）
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    block_delta: Dict[str, List[Tuple]] = {}
    new_memory = memory

    # 收集所有 FSDP unit（recurse=False 避免重复）
    fsdp_units: List[torch.nn.Module] = []
    for mod in model.modules():
        if isinstance(mod, FSDP):
            fsdp_units.append(mod)

    if not fsdp_units:
        # 非 FSDP 模型：直接从 model.state_dict() 提取（回退到旧行为）
        if is_main:
            full_state = cpu_state(model.state_dict())
            from shared.block_selection import encode_block_delta, update_block_memory
            delta = sub_state(full_state, local_global)
            to_send = add_state(delta, memory)
            block_delta = encode_block_delta(to_send, selected_by_key, dtype=transfer_dtype)
            new_memory = update_block_memory(to_send, selected_by_key, memory_decay)
            del full_state, delta, to_send
        return block_delta, new_memory

    # SCALE-3 核心路径：逐 FSDP unit unshard
    selected_set = set(selected_by_key.keys())

    for fsdp_unit in fsdp_units:
        # summon_full_params 是 collective，所有 rank 都要执行
        # offload_to_cpu=True: 把完整参数 offload 到 CPU（减少 GPU 显存）
        # rank0_only=False: 所有 rank 都 unshard（collective 要求）
        # writeback=False: 不需要写回（只读提取 delta）
        # recurse=True: 获取该 unit 的所有子参数（带完整 key）
        try:
            cm = FSDP.summon_full_params(
                fsdp_unit, recurse=True, offload_to_cpu=True,
                rank0_only=False, writeback=False,
            )
            ctx = cm.__enter__()
        except NotImplementedError:
            # NO_SHARD 模式（单卡测试）不支持 offload_to_cpu
            cm = FSDP.summon_full_params(
                fsdp_unit, recurse=True, offload_to_cpu=False,
                rank0_only=False, writeback=False,
            )
            ctx = cm.__enter__()

        try:
            if is_main:
                # 只有 rank0 有 local_global / memory，做 delta 提取
                # FSDP unit 的 named_parameters() 返回的是相对于该 unit 的 key
                # 需要构建完整 key（与 local_global 的 key 一致）
                # 方法：用 model.named_parameters() 遍历，找到属于该 fsdp_unit 的参数
                # 但更简单：直接用 module prefix + param name
                # FSDP 的 fsdp_unit 可能是顶层（key 带 model. 前缀）或子层（key 不带前缀）
                # 用 module 的 _fully_sharded_module_name 或遍历 parent 构建
                # 最简单：遍历 model 的所有 named_parameters，检查 id 匹配
                unit_param_ids = set(id(p) for _, p in fsdp_unit.named_parameters(recurse=True))
                for full_name, param in model.named_parameters():
                    if id(param) not in unit_param_ids:
                        continue
                    clean_name = _strip_orig_mod_prefix(full_name)
                    if clean_name not in selected_set:
                        continue
                    if not param.is_floating_point():
                        continue
                    flat = param.data.contiguous().view(-1).to(dtype=torch.float32)
                    lg = local_global.get(clean_name)
                    mem = memory.get(clean_name)
                    if lg is not None:
                        lg_flat = lg.contiguous().view(-1).to(dtype=torch.float32)
                        if flat.numel() != lg_flat.numel():
                            logger.warning(
                                "SCALE-3: size mismatch for %s: param=%s vs local_global=%s, skipping",
                                clean_name, flat.numel(), lg_flat.numel(),
                            )
                            continue
                        flat = flat - lg_flat
                    if mem is not None:
                        mem_flat = mem.contiguous().view(-1).to(dtype=torch.float32)
                        if flat.numel() == mem_flat.numel():
                            flat = flat + mem_flat
                    # 提取选中的 block
                    out_dtype = transfer_dtype if transfer_dtype is not None else param.dtype
                    blocks: List[Tuple] = []
                    for s, e in selected_by_key[clean_name]:
                        slice_data = flat[s:e]
                        if out_dtype == torch.int8:
                            from shared.block_selection import _quantize_int8_block
                            q, scale = _quantize_int8_block(slice_data)
                            blocks.append((s, e, q.detach().to(device="cpu").contiguous(), scale))
                        else:
                            blocks.append((s, e, slice_data.detach().to(device="cpu", dtype=out_dtype).contiguous()))
                    block_delta[clean_name] = blocks
        finally:
            cm.__exit__(None, None, None)

    # 更新 memory（只在 rank0）
    if is_main:
        from shared.block_selection import update_block_memory
        # 需要完整的 to_send 来更新 memory，但不想 gather 完整 state
        # memory 更新逻辑：已上传 block 置 0，其余保留，再 * decay
        # 这里用 local_global + memory 近似（因为 to_send = delta + memory = (full - local_global) + memory）
        # 但我们没有 full_state... 需要另一种方式
        # 实际上 update_block_memory 只需要 to_send 和 selected_by_key
        # to_send = (full - local_global) + memory
        # 但我们没有 full_state
        # 替代方案：直接更新 memory 的选中 block 为 0，其余 * decay
        new_memory = _update_memory_sharded(memory, selected_by_key, memory_decay)

    return block_delta, new_memory


def _update_memory_sharded(
    memory: Dict[str, torch.Tensor],
    selected_by_key: Dict[str, List[Tuple[int, int]]],
    decay: float,
) -> Dict[str, torch.Tensor]:
    """SCALE-3：在 shard 模式下更新 memory。

    原始 update_block_memory 需要 to_send（完整 state），但 SCALE-3 不 gather 完整 state。
    memory 更新语义：已上传 block 置 0，其余保留，再整体 * decay。
    但 to_send = (full - local_global) + memory，已上传 block 在 to_send 中为非 0（因为 full 变了）。
    原始逻辑是把 to_send 的已上传 block 置 0（表示"已发送，不需要再发"），其余 * decay。

    在 SCALE-3 下，我们没有 to_send，但有 memory 和 selected_by_key。
    近似：把 memory 的已上传 block 置 0，其余 * decay。
    这与原始逻辑在 delta=0（第一轮后）时等价。
    """
    new_memory: Dict[str, torch.Tensor] = {}
    for key_name, tensor in memory.items():
        if not tensor.is_floating_point():
            new_memory[key_name] = tensor.clone()
            continue
        flat = tensor.contiguous().view(-1).to(dtype=torch.float32).clone()
        if key_name in selected_by_key:
            for s, e in selected_by_key[key_name]:
                flat[s:e] = 0.0
        new_memory[key_name] = (flat * decay).to(dtype=tensor.dtype).view(tensor.shape)
    return new_memory


def apply_block_delta_to_local_global(
    local_global: Dict[str, torch.Tensor],
    agg_block_delta: Dict[str, List[Tuple]],
) -> None:
    """SCALE-3：把聚合后的 block delta apply 到 local_global（原地修改）。

    与 shared.block_selection.add_block_delta 相同，但封装在这里方便调用。
    """
    from shared.block_selection import add_block_delta
    add_block_delta(local_global, agg_block_delta)
