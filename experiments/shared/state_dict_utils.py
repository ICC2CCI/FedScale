"""State dict 工具：差值 / 累加 / FSDP 全量提取与加载。"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def cpu_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().to(device="cpu") for k, v in state.items()}


def sub_state(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    *,
    keep_fp32: bool = False,
) -> Dict[str, torch.Tensor]:
    """a - b。浮点运算在 FP32 中完成。

    keep_fp32=True 时不把结果再 round 回 av.dtype（SecAgg 量化前必须保持 FP32）。
    默认 False：保持历史行为，结果 dtype 跟随 a。
    """
    out: Dict[str, torch.Tensor] = {}
    for k, av in a.items():
        bv = b.get(k)
        if bv is None:
            out[k] = av.clone() if hasattr(av, "clone") else av
        elif av.is_floating_point():
            diff = av.to(dtype=torch.float32) - bv.to(dtype=torch.float32)
            out[k] = diff if keep_fp32 else diff.to(dtype=av.dtype)
        else:
            out[k] = av.clone()
    return out


def add_state(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    *,
    keep_fp32: bool = False,
) -> Dict[str, torch.Tensor]:
    """a + b。浮点运算在 FP32 中完成。

    keep_fp32=True 时不把结果再 round 回 av.dtype。
    """
    out: Dict[str, torch.Tensor] = {}
    for k, av in a.items():
        bv = b.get(k)
        if bv is None:
            out[k] = av.clone() if hasattr(av, "clone") else av
        elif av.is_floating_point():
            summed = av.to(dtype=torch.float32) + bv.to(dtype=torch.float32)
            out[k] = summed if keep_fp32 else summed.to(dtype=av.dtype)
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
    """把完整 CPU state 写入 FSDP 分片。

    只有 rank0 需要持有 ``state``；其它 rank 传 ``None`` / ``{}``。
    按 FSDP unit 逐个 ``summon``（recurse=False），峰值约一层，而不是 8 份整模。
    禁止再用 FULL_STATE_DICT + rank0_only=False（那会让每张卡 CPU 都摊一份全量）。
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fsdp_model = _as_fsdp(model)
    if fsdp_model is None:
        if not state:
            raise ValueError("load_full_state_fsdp requires a state dict on non-FSDP models")
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device=device) for k, v in state.items()}, strict=True)
        return

    rank = _dist_rank()
    is_src = rank == 0
    if is_src and not state:
        raise ValueError("load_full_state_fsdp: rank0 must pass the full CPU state dict")

    fsdp_units = [m for m in model.modules() if isinstance(m, FSDP)]

    def _has_child_fsdp(unit) -> bool:
        return any(isinstance(c, FSDP) and c is not unit for c in unit.modules())

    leaves = [u for u in fsdp_units if not _has_child_fsdp(u)]
    parents = [u for u in fsdp_units if _has_child_fsdp(u)]
    copied = 0
    missing = 0
    processed_ids: set = set()
    # 叶子（DecoderLayer）整层 unshard；根节点 recurse=False 只碰 embed/lm_head/norm。
    for fsdp_unit, rec in [(u, True) for u in leaves] + [(u, False) for u in parents]:
        try:
            cm = FSDP.summon_full_params(
                fsdp_unit, recurse=rec, offload_to_cpu=True,
                rank0_only=False, writeback=True,
            )
            cm.__enter__()
        except NotImplementedError:
            cm = FSDP.summon_full_params(
                fsdp_unit, recurse=rec, offload_to_cpu=False,
                rank0_only=False, writeback=True,
            )
            cm.__enter__()
        try:
            unit_ids = {id(p) for _, p in fsdp_unit.named_parameters(recurse=True)}
            for full_name, param in model.named_parameters():
                pid = id(param)
                if pid in processed_ids or pid not in unit_ids:
                    continue
                processed_ids.add(pid)
                key = _canonical_param_name(full_name)
                if is_src:
                    src = state.get(key) if state is not None else None
                    if src is None:
                        missing += 1
                        if missing <= 5:
                            logger.warning("FSDP scatter-load missing key=%s (from %s)", key, full_name)
                    else:
                        param.data.copy_(
                            src.detach().to(device=param.device, dtype=param.dtype).reshape_as(param)
                        )
                        copied += 1
                _broadcast_tensor_inplace(param.data, src=0)
        finally:
            cm.__exit__(None, None, None)

    if is_src:
        logger.info(
            "FSDP scatter-load: units=%s leaves=%s copied=%s missing=%s (per-unit summon, no 8-way full dict)",
            len(fsdp_units), len(leaves), copied, missing,
        )
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        n_copied = broadcast_object(copied if is_src else None, src=0)
    else:
        n_copied = copied
    if int(n_copied or 0) <= 0:
        raise RuntimeError("FSDP scatter-load copied 0 parameters; name mapping failed")


# ---------------------------------------------------------------------------
# SCALE-3：逐 FSDP unit 提取 block delta，不 gather 完整 state
# ---------------------------------------------------------------------------


def _strip_orig_mod_prefix(name: str) -> str:
    """accelerate 包装的模型参数名可能有 _orig_mod. 前缀，去掉它。"""
    if name.startswith("_orig_mod."):
        return name[len("_orig_mod."):]
    return name


def _canonical_param_name(name: str) -> str:
    """FSDP/accelerate 包装名 → HuggingFace state_dict 键。"""
    name = _strip_orig_mod_prefix(name)
    return name.replace("._fsdp_wrapped_module", "").replace("_fsdp_wrapped_module.", "")


def _dist_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _broadcast_tensor_inplace(tensor: torch.Tensor, src: int = 0) -> None:
    """各 rank 对齐同一份 tensor。NCCL 不能广播 CPU tensor，需要时经当前 GPU 中转。"""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    if tensor.device.type == "cpu" and dist.get_backend() == "nccl":
        dev = torch.device("cuda", torch.cuda.current_device())
        gpu = tensor.to(dev, non_blocking=False)
        dist.broadcast(gpu, src=src)
        tensor.copy_(gpu.to(device="cpu"))
        del gpu
        return
    dist.broadcast(tensor, src=src)


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
    new_memory = memory if is_main else memory
    processed_keys: set = set()  # 防止重复处理（顶层 FSDP unit 包含所有子 unit 的参数）

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
                unit_param_ids = set(id(p) for _, p in fsdp_unit.named_parameters(recurse=True))
                for full_name, param in model.named_parameters():
                    if id(param) not in unit_param_ids:
                        continue
                    clean_name = _strip_orig_mod_prefix(full_name)
                    if clean_name in processed_keys:
                        continue  # 已被子 FSDP unit 处理过
                    processed_keys.add(clean_name)
                    if not param.is_floating_point():
                        continue
                    flat = param.data.contiguous().view(-1).to(dtype=torch.float32)
                    lg = local_global.get(clean_name)
                    mem = memory.get(clean_name)
                    # delta / to_send 保持 FP32，避免量化或 encode 前再 round 一次 FP16。
                    if lg is not None:
                        lg_flat = lg.contiguous().view(-1).to(dtype=torch.float32)
                        if flat.numel() != lg_flat.numel():
                            logger.warning(
                                "SCALE-3: size mismatch for %s: param=%s vs local_global=%s, skipping",
                                clean_name, flat.numel(), lg_flat.numel(),
                            )
                            continue
                        delta_flat = flat - lg_flat
                    else:
                        delta_flat = flat.clone()
                    if mem is not None:
                        mem_flat = mem.contiguous().view(-1).to(dtype=torch.float32)
                        if delta_flat.numel() == mem_flat.numel():
                            to_send_flat = delta_flat + mem_flat
                        else:
                            to_send_flat = delta_flat
                    else:
                        to_send_flat = delta_flat

                    # 提取选中的 block
                    if clean_name in selected_set:
                        out_dtype = transfer_dtype if transfer_dtype is not None else param.dtype
                        blocks: List[Tuple] = []
                        for s, e in selected_by_key[clean_name]:
                            slice_data = to_send_flat[s:e]
                            if out_dtype == torch.int8:
                                from shared.block_selection import _quantize_int8_block
                                q, scale = _quantize_int8_block(slice_data)
                                blocks.append((s, e, q.detach().to(device="cpu").contiguous(), scale))
                            else:
                                blocks.append((s, e, slice_data.detach().to(device="cpu", dtype=out_dtype).contiguous()))
                        block_delta[clean_name] = blocks

                    # 更新 memory: 已上传 block 置 0，其余保留，再 * decay
                    # 与原版 update_block_memory 语义一致
                    if mem is not None:
                        mem_updated = to_send_flat.clone()
                        if clean_name in selected_by_key:
                            for s, e in selected_by_key[clean_name]:
                                mem_updated[s:e] = 0.0
                        new_memory[clean_name] = (mem_updated * memory_decay).to(dtype=mem.dtype).view(mem.shape)
                    else:
                        # 首轮 memory 为 0，创建新 memory
                        mem_updated = to_send_flat.clone()
                        if clean_name in selected_by_key:
                            for s, e in selected_by_key[clean_name]:
                                mem_updated[s:e] = 0.0
                        new_memory[clean_name] = (mem_updated * memory_decay).to(dtype=param.dtype).view(param.shape)
        finally:
            cm.__exit__(None, None, None)

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
