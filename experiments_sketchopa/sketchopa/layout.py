"""固定公开 Sketch layout 构造。

按层分区，每层 R 行 × B_l 个 bucket。
flat sketch 长度 = R × Σ_l B_l。

两种分配策略：
- uniform_param：按层参数量均匀分 bucket
- topk_count_calibrated：按外部 calibration 文件（各层 Top-k 命中计数）分配

budget 公式（§6.3）：
    topk16_payload = k × (idx_bytes + val_bytes) = k × 6
    sketch_payload = R × Σ_l(B_l) × value_bytes
    budget_ratio = sketch_payload / topk16_payload
    => Σ_l(B_l) = budget_ratio × 6k / (R × value_bytes)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class LayerLayout:
    """单层 layout 描述。"""

    name: str            # 参数名（state_dict key）
    start: int           # 全模型展平起始坐标
    end: int             # 全模型展平结束坐标（不含）
    num_params: int      # = end - start
    buckets_per_row: int # B_l
    is_dense: bool = False  # 小层走稠密 SA（不进 Sketch）

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "num_params": self.num_params,
            "buckets_per_row": self.buckets_per_row,
            "is_dense": self.is_dense,
        }


@dataclass
class SketchLayout:
    """全模型 Sketch layout。"""

    rows: int                         # R
    value_bytes: int                  # 传输值字节数（2=fp16）
    idx_bytes: int = 4                # Top-k 索引字节数
    layers: List[LayerLayout] = field(default_factory=list)
    allocation_strategy: str = "uniform_param"
    budget_ratio: float = 1.0
    min_buckets: int = 8
    bucket_factor: int = 8
    layout_hash: str = ""

    @property
    def sketch_len(self) -> int:
        """flat sketch tensor 长度 = R × Σ_l B_l（仅 Sketch 层）。"""
        return self.rows * sum(l.buckets_per_row for l in self.layers if not l.is_dense)

    @property
    def total_buckets_per_row(self) -> int:
        return sum(l.buckets_per_row for l in self.layers if not l.is_dense)

    @property
    def payload_bytes(self) -> int:
        return self.sketch_len * self.value_bytes

    def sketch_layer_offsets(self) -> List[Tuple[int, int]]:
        """返回每个 Sketch 层在 flat（per-row）布局中的 (offset, length)。"""
        offset = 0
        out: List[Tuple[int, int]] = []
        for l in self.layers:
            if l.is_dense:
                out.append((-1, -1))  # 稠密层不占 Sketch 位置
            else:
                out.append((offset, l.buckets_per_row))
                offset += l.buckets_per_row
        return out

    def compute_hash(self) -> str:
        parts = [f"{self.rows}|{self.value_bytes}|{self.allocation_strategy}|{self.budget_ratio}"]
        for l in self.layers:
            parts.append(f"{l.name}|{l.start}|{l.end}|{l.buckets_per_row}|{int(l.is_dense)}")
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "rows": self.rows,
            "value_bytes": self.value_bytes,
            "idx_bytes": self.idx_bytes,
            "allocation_strategy": self.allocation_strategy,
            "budget_ratio": self.budget_ratio,
            "min_buckets": self.min_buckets,
            "bucket_factor": self.bucket_factor,
            "layout_hash": self.layout_hash or self.compute_hash(),
            "layers": [l.to_dict() for l in self.layers],
            "sketch_len": self.sketch_len,
            "payload_bytes": self.payload_bytes,
        }


def build_layer_meta_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> List[Tuple[str, int, int]]:
    """从模型 state_dict 构建层元信息列表：(name, start, end)。

    按state_dict key 顺序展平，跳过非浮点 tensor。
    """
    layers: List[Tuple[str, int, int]] = []
    offset = 0
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        n = tensor.numel()
        if n == 0:
            continue
        layers.append((name, offset, offset + n))
        offset += n
    return layers


def _allocate_buckets(
    layers: List[Tuple[str, int, int]],
    total_buckets: int,
    weights: Sequence[float],
    min_buckets: int,
) -> List[int]:
    """按 weights 给每层分配 bucket，保底 min_buckets，总和不超过 total_buckets。

    采用先保底再按剩余权重按比例分配的方法。
    """
    n = len(layers)
    weights = [max(w, 0.0) for w in weights]
    total_w = sum(weights)

    # 第一步：所有层先给 min_buckets
    base = [min_buckets] * n
    used = sum(base)

    if total_w <= 0:
        # 无权重信息：均分剩余
        remain = max(total_buckets - used, 0)
        extra = [remain // n] * n
        for i in range(remain % n):
            extra[i] += 1
        return [base[i] + extra[i] for i in range(n)]

    # 第二步：剩余预算按权重比例分
    remain = max(total_buckets - used, 0)
    alloc = list(base)
    if remain > 0:
        ratios = [w / total_w for w in weights]
        extra = [int(remain * r) for r in ratios]
        # 把舍入误差补到最大权重层
        diff = remain - sum(extra)
        if diff != 0:
            idx = max(range(n), key=lambda i: weights[i])
            extra[idx] += diff
        alloc = [alloc[i] + extra[i] for i in range(n)]

    # 保证至少 min_buckets
    return [max(a, min_buckets) for a in alloc]


def build_layout(
    state_dict: Dict[str, torch.Tensor],
    k: int,
    rows: int = 5,
    value_bytes: int = 2,
    idx_bytes: int = 4,
    budget_ratio: float = 1.0,
    allocation_strategy: str = "uniform_param",
    min_buckets: int = 8,
    bucket_factor: int = 8,
    dense_layer_threshold: int = 0,   # <=0 表示不启用混合聚合
    calibration_path: Optional[str] = None,
) -> SketchLayout:
    """构建固定公开 Sketch layout。

    Args:
        k: global Top-k 数量
        rows: CountSketch 行数 R
        value_bytes: 传输值字节数（fp16=2）
        budget_ratio: sketch_payload / topk16_payload
        allocation_strategy: "uniform_param" | "topk_count_calibrated"
        dense_layer_threshold: 参数数 < 该值的层走稠密 SA（<=0 全进 Sketch）
        calibration_path: calibrated 模式下各层 Top-k 计数 json 路径
    """
    layer_meta = build_layer_meta_from_state_dict(state_dict)
    if not layer_meta:
        raise ValueError("state_dict 中无浮点参数")

    # 总 bucket 预算：Σ B_l = budget_ratio × 6k / (R × value_bytes)
    topk16_payload = k * (idx_bytes + value_bytes)
    total_budget = int(budget_ratio * topk16_payload / (rows * value_bytes))
    total_budget = max(total_budget, min_buckets * len(layer_meta))

    # 权重
    if allocation_strategy == "topk_count_calibrated":
        weights = _load_calibration_weights(layer_meta, calibration_path)
    elif allocation_strategy == "uniform_param":
        weights = [float(end - start) for (_, start, end) in layer_meta]
    else:
        raise ValueError(f"unknown allocation_strategy: {allocation_strategy}")

    buckets = _allocate_buckets(layer_meta, total_budget, weights, min_buckets)

    # 构建层 layout
    layers: List[LayerLayout] = []
    for (name, start, end), b in zip(layer_meta, buckets):
        n = end - start
        is_dense = (dense_layer_threshold > 0 and n < dense_layer_threshold)
        if is_dense:
            b = 0  # 稠密层不占 Sketch bucket
        layers.append(LayerLayout(
            name=name, start=start, end=end,
            num_params=n, buckets_per_row=b, is_dense=is_dense,
        ))

    layout = SketchLayout(
        rows=rows, value_bytes=value_bytes, idx_bytes=idx_bytes,
        layers=layers, allocation_strategy=allocation_strategy,
        budget_ratio=budget_ratio, min_buckets=min_buckets,
        bucket_factor=bucket_factor,
    )
    layout.layout_hash = layout.compute_hash()
    return layout


def _load_calibration_weights(
    layer_meta: List[Tuple[str, int, int]],
    calibration_path: Optional[str],
) -> List[float]:
    """从 calibration 文件加载各层 Top-k 命中计数作为权重。

    文件格式：{ "layer_name": count, ... }
    缺失层给 0（后续 min_buckets 保底）。
    """
    if calibration_path is None or not Path(calibration_path).exists():
        # 无 calibration 文件：退化为按参数量（避免全 0）
        return [float(end - start) for (_, start, end) in layer_meta]

    with open(calibration_path) as f:
        counts = json.load(f)

    weights: List[float] = []
    for (name, _, _) in layer_meta:
        weights.append(float(counts.get(name, 0.0)))
    return weights


def save_calibration(
    layer_counts: Dict[str, int],
    path: str,
    extra: Optional[Dict] = None,
) -> None:
    """保存各层 Top-k 命中计数到 calibration 文件。"""
    data = {"layer_counts": layer_counts}
    if extra:
        data.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
