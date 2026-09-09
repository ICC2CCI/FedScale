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

# 训练超参
DEFAULT_LOCAL_STEPS = 30
DEFAULT_BATCH = 8
DEFAULT_GRAD_ACCUM = 2
DEFAULT_LR = 1e-5
DEFAULT_SEQ_LEN = 512


def global_state_key(round_idx: int) -> str:
    return f"global_state/round-{round_idx}/state.pt"


def upload_blocks_key(round_idx: int, client_id: int) -> str:
    return f"uploads/round-{round_idx}/client-{client_id}/blocks.pt"


def plan_key(round_idx: int) -> str:
    return f"plans/round-{round_idx}/plan.json"


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
