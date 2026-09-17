"""通信协议常量与路径约定（HTTP REST + MinIO）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 默认超参（与 S3R12v3 单机实验对齐）
DEFAULT_SEED = 20260831
DEFAULT_COVERAGE_H = 5
DEFAULT_BLOCK_SIZE = 524288  # 元素数 ≈ 1MB @ fp16/bf16
DEFAULT_MEMORY_DECAY = 0.9
DEFAULT_NUM_ROUNDS = 20
DEFAULT_NUM_CLIENTS = 2
DEFAULT_BUCKET = "fedscale-bucket"
# 通信/落盘精度：auto=跟随模型(global_state)浮点 dtype；也可强制 fp16/fp32/bf16（int8 预留）
DEFAULT_TRANSFER_DTYPE = "auto"

# 训练超参
DEFAULT_LOCAL_STEPS = 30
DEFAULT_BATCH = 8
DEFAULT_GRAD_ACCUM = 2
DEFAULT_LR = 1e-5
DEFAULT_SEQ_LEN = 512


def global_state_key(round_idx: int) -> str:
    return f"global_state/round-{round_idx}/state.pt"


def global_delta_key(round_idx: int) -> str:
    """Round N 聚合后的增量：把 global 从 round-(N-1) 更新到 round-N（仅选中 blocks）。"""
    return f"global_delta/round-{round_idx}/blocks.pt"


def upload_blocks_key(round_idx: int, client_id: int) -> str:
    return f"uploads/round-{round_idx}/client-{client_id}/blocks.pt"


def upload_block_key(round_idx: int, client_id: int, block_idx: int) -> str:
    """流式：单个 block 的上传 key（per-block pipeline）。"""
    return f"uploads/round-{round_idx}/client-{client_id}/block-{block_idx}.pt"


def agg_block_key(round_idx: int, block_idx: int) -> str:
    """流式：server 聚合后单个 block 的 delta key（client 可逐 block 下载）。"""
    return f"global_delta/round-{round_idx}/block-{block_idx}.pt"


def agg_block_done_key(round_idx: int) -> str:
    """流式：标记本轮所有 block 聚合完成的哨兵 key。"""
    return f"global_delta/round-{round_idx}/_done"


def plan_key(round_idx: int) -> str:
    return f"plans/round-{round_idx}/plan.json"


def epoch_seed_from_plan(seed: int, epoch: int) -> bytes:
    """从 (seed, epoch) 派生 epoch_seed——server 与 client 独立计算得到相同值。

    与 block_selection.build_permutations 内部公式一致，SEC-2 用它派生 per-round 密钥。
    """
    import hashlib

    return hashlib.sha256(f"FedScale-BlockMask-v1|{seed}|{epoch}".encode("utf-8")).digest()


@dataclass
class RoundPlan:
    round: int
    epoch: int
    slot: int
    coverage_h: int
    seed: int
    selected_by_key: Dict[str, List[List[int]]]
    n_selected_blocks: int
    selected_elems: int
    total_elems: int
    # 流式 per-block pipeline：有序 block 列表，每项 [global_idx, key_name, start, end]
    block_list: List[List[int]] = field(default_factory=list)
    # A-3: 审计字段（spec v1 第 9 节）
    layout_hash: str = ""        # SHA256 of canonical-encoded layout (groups + always_on + H + block_size)
    mask_hash: str = ""          # SHA256 of canonical-encoded selected blocks
    mask_policy_id: str = ""     # e.g. "hierarchical-permute-rotate-v1"
    always_on_keys: List[str] = field(default_factory=list)  # sorted key names that are always selected
    # B-debug: 自适应 scale（server 下发给 client）
    secagg_scale: float = 0.0

    @property
    def upload_ratio(self) -> float:
        if self.total_elems <= 0:
            return 0.0
        return self.selected_elems / self.total_elems

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoundPlan":
        return cls(
            round=int(data["round"]),
            epoch=int(data["epoch"]),
            slot=int(data["slot"]),
            coverage_h=int(data["coverage_h"]),
            seed=int(data["seed"]),
            selected_by_key={
                k: [list(pair) for pair in v] for k, v in data["selected_by_key"].items()
            },
            n_selected_blocks=int(data["n_selected_blocks"]),
            selected_elems=int(data["selected_elems"]),
            total_elems=int(data["total_elems"]),
            block_list=[list(b) for b in data.get("block_list", [])],
            layout_hash=str(data.get("layout_hash", "")),
            mask_hash=str(data.get("mask_hash", "")),
            mask_policy_id=str(data.get("mask_policy_id", "")),
            always_on_keys=list(data.get("always_on_keys", [])),
            secagg_scale=float(data.get("secagg_scale", 0.0)),
        )


@dataclass
class RoundResult:
    round: int
    done: bool
    eval_loss: Optional[float] = None
    avg_train_loss: Optional[float] = None
    upload_ratio: Optional[float] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClientUploadMeta:
    client_id: int
    num_examples: int
    train_loss: float
    extra: Dict[str, Any] = field(default_factory=dict)


SelectedByKey = Dict[str, List[Tuple[int, int]]]


def selected_to_jsonable(selected: SelectedByKey) -> Dict[str, List[List[int]]]:
    return {k: [[int(s), int(e)] for s, e in slices] for k, slices in selected.items()}


def selected_from_jsonable(data: Dict[str, List[List[int]]]) -> SelectedByKey:
    return {k: [(int(s), int(e)) for s, e in slices] for k, slices in data.items()}


# ---------------------------------------------------------------------------
# B-3: SecAgg 协议对象（spec section 12）
# ---------------------------------------------------------------------------

SECAGG_PROTOCOL_ID = "fedscale-window-secagg-v1"


@dataclass
class SecAggPlan:
    """SecAgg 计划（嵌入 RoundPlan 或独立下发）。"""

    secagg_protocol_id: str = SECAGG_PROTOCOL_ID
    secagg_session_id: str = ""          # H(job_id, round, model_version, cohort, mask)
    attempt_id: int = 0                  # 重试递增；mask 全部重采样
    cohort_hash: str = ""                # H(sorted client_ids)
    mask_hash: str = ""                  # 来自 Block Mask
    window_plan_hash: str = ""           # H(all window descriptors)
    q_min: int = 2                       # 最小成功参与者数
    quantization_scale: float = 2.0 ** -14  # 全局 fallback scale（无 per-window 时使用）
    window_scales: Dict[str, float] = field(default_factory=dict)  # str(window_id) → scale
    modulus_bits: int = 16               # q = 2^modulus_bits
    modulus_q: int = 65536               # q = 2^modulus_bits
    q_max: int = 16383                   # 量化值上界
    reconstruction_threshold: int = 1    # Shamir 恢复阈值（2-client 无掉线=1）
    stochastic_rounding: bool = False    # 随机舍入

    def get_window_scale(self, window_id: int) -> float:
        """取该 window 的定点 scale；未下发时回退到全局 quantization_scale。"""
        if self.window_scales:
            val = self.window_scales.get(str(window_id))
            if val is None:
                val = self.window_scales.get(window_id)  # type: ignore[arg-type]
            if val is not None and float(val) > 0.0:
                return float(val)
        return float(self.quantization_scale)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecAggPlan":
        raw_scales = data.get("window_scales") or {}
        return cls(
            secagg_protocol_id=str(data.get("secagg_protocol_id", SECAGG_PROTOCOL_ID)),
            secagg_session_id=str(data.get("secagg_session_id", "")),
            attempt_id=int(data.get("attempt_id", 0)),
            cohort_hash=str(data.get("cohort_hash", "")),
            mask_hash=str(data.get("mask_hash", "")),
            window_plan_hash=str(data.get("window_plan_hash", "")),
            q_min=int(data.get("q_min", 2)),
            quantization_scale=float(data.get("quantization_scale", 2.0 ** -14)),
            window_scales={str(k): float(v) for k, v in raw_scales.items()},
            modulus_bits=int(data.get("modulus_bits", 16)),
            modulus_q=int(data.get("modulus_q", 65536)),
            q_max=int(data.get("q_max", 16383)),
            reconstruction_threshold=int(data.get("reconstruction_threshold", 1)),
            stochastic_rounding=bool(data.get("stochastic_rounding", False)),
        )


@dataclass
class WindowDescriptor:
    """单个 window (= 1 block) 的描述符。"""

    window_id: int               # = gidx
    gidx: int                    # global block index
    key_name: str
    start: int
    end: int
    vector_length: int           # = end - start
    window_layout_hash: str      # H(gidx || start || end || length)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindowDescriptor":
        return cls(
            window_id=int(data["window_id"]),
            gidx=int(data["gidx"]),
            key_name=str(data["key_name"]),
            start=int(data["start"]),
            end=int(data["end"]),
            vector_length=int(data["vector_length"]),
            window_layout_hash=str(data["window_layout_hash"]),
        )


def compute_window_layout_hash(gidx: int, start: int, end: int, length: int) -> str:
    """计算 window_layout_hash。"""
    import hashlib

    data = (
        b"window|"
        + str(gidx).encode("utf-8") + b"|"
        + str(start).encode("utf-8") + b"|"
        + str(end).encode("utf-8") + b"|"
        + str(length).encode("utf-8")
    )
    return hashlib.sha256(data).hexdigest()


def build_window_descriptors(block_list: List[List[int]]) -> List[WindowDescriptor]:
    """从 RoundPlan.block_list 构建 WindowDescriptor 列表。

    block_list 每项: [gidx, key_name, start, end]
    """
    windows = []
    for b in block_list:
        gidx = int(b[0])
        key_name = str(b[1])
        start = int(b[2])
        end = int(b[3])
        length = end - start
        wlh = compute_window_layout_hash(gidx, start, end, length)
        windows.append(WindowDescriptor(
            window_id=gidx,
            gidx=gidx,
            key_name=key_name,
            start=start,
            end=end,
            vector_length=length,
            window_layout_hash=wlh,
        ))
    return windows
