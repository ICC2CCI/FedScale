"""Run 配置加载：YAML + CLI 覆盖。

三端（Server / Client / 启动脚本）共用。yaml 优先于 argparse 默认值，
但 CLI 显式传入的参数覆盖 yaml。

典型用法::

    from shared.run_config import load_run_config, apply_to_args

    cfg = load_run_config(args.config)          # 没传 --config 则返回 {}
    apply_to_args(args, cfg, schema=RUN_CONFIG_SCHEMA)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("run_config")

# 顶层 section -> (arg名, yaml键) 映射。
# value 为 None 表示该字段名在 yaml 与 argparse 之间一致，无需别名。
# 这里的 schema 同时定义了「哪些键会被识别」与「argparse dest 名」。
RUN_CONFIG_SCHEMA: Dict[str, List[str]] = {
    "federated": [
        "num_clients",
        "num_rounds",
        "coverage_h",
        "slots_per_round",
        "seed",
        "transfer_dtype",
        "memory_decay",
        "block_size",
    ],
    "train": [
        "local_steps",
        "batch_size",
        "grad_accum",
        "lr",
        "seq_len",
    ],
    "io": [
        "skip_round0_download",
        "online_eval",
        "write_full_global_every_n_rounds",
        "client_upload_timeout_s",
    ],
    "eval": [
        "eval_path",
        "eval_max_batches",
    ],
    "sync": [
        "min_clients_to_aggregate",
        "round_deadline_s",
    ],
    "security": [
        "auth_token",
        "tls",
    ],
    "resume": [
        "resume_from_round",
    ],
    "ops": [
        "minio_retention_recent_uploads",
    ],
}


def load_run_config(path: Optional[str]) -> Dict[str, Any]:
    """读 yaml 并扁平化为 {argname: value}。path 为空则返回 {}。"""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"run config not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load run config: pip install pyyaml") from exc
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"run config root must be a mapping: {path}")
    flat: Dict[str, Any] = {}
    for section, keys in RUN_CONFIG_SCHEMA.items():
        sec = raw.get(section, {})
        if sec is None:
            continue
        if not isinstance(sec, dict):
            raise ValueError(f"run config section '{section}' must be a mapping: {path}")
        for k in keys:
            if k in sec:
                flat[k] = sec[k]
    # 顶层裸键（向后兼容，允许不分区直接写 num_rounds 等）
    for k, v in raw.items():
        if isinstance(v, (dict, list)):
            continue
        flat.setdefault(k, v)
    logger.info("Loaded run config from %s keys=%s", path, sorted(flat.keys()))
    return flat


def _coerce(value: Any, ref: Any) -> Any:
    """按现有 argparse 值的类型推断，把 yaml 值转过去。"""
    if ref is None or value is None:
        return value
    if isinstance(ref, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(ref, int) and not isinstance(ref, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(ref, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def _arg_was_set(args: Any, key: str, parser=None) -> bool:
    """判断 argparse 是否显式传了该参数。

    优先用 parser 的解析记录；若没传 parser，则对布尔 BooleanOptionalAction 用「非默认」启发，
    其余类型无法区分 default vs 显式传同值——此时保守地认为「未显式传」，让 yaml 覆盖默认。
    """
    if parser is not None:
        # argparse 在 3.9+ 暴露 _get_values；这里用更稳的方式：检查 sys.argv 里的 flag
        import sys

        flags = {a.lstrip("-") for a in sys.argv[1:]}
        flags_dashed = {a for a in sys.argv[1:] if a.startswith("-")}
        # 支持 --num-rounds / --num_rounds 两种风格
        if key in flags or key.replace("_", "-") in flags:
            return True
        for a in flags_dashed:
            name = a.lstrip("-")
            if name == key or name == key.replace("_", "-"):
                return True
        return False
    return False


def apply_to_args(
    args: Any,
    cfg: Dict[str, Any],
    *,
    parser=None,
    skip: Optional[Sequence[str]] = None,
) -> Any:
    """把 yaml 值填进 args：仅当 argparse 未显式传该参数时覆盖。

    args: argparse.Namespace
    cfg: load_run_config 返回的扁平 dict
    parser: 可选 argparse.ArgumentParser（用于更精确判断是否显式传参；None 则用 sys.argv 启发）
    skip: 这些键不从 yaml 注入（例如 client_id / server_url 这类 per-role 字段）
    """
    skipset = set(skip or ())
    changed: List[str] = []
    for key, value in cfg.items():
        if key in skipset:
            continue
        if not hasattr(args, key):
            continue
        if _arg_was_set(args, key, parser):
            continue
        ref = getattr(args, key)
        new = _coerce(value, ref)
        setattr(args, key, new)
        changed.append(f"{key}={new!r}")
    if changed:
        logger.info("Run config applied to args: %s", ", ".join(changed))
    return args


def snapshot_effective_config(args: Any, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """把 args 中本 schema 关心的键导出成扁平 dict，写入 run_meta.json / run.yaml 用。"""
    out: Dict[str, Any] = {}
    if keys is None:
        all_keys: List[str] = []
        for ks in RUN_CONFIG_SCHEMA.values():
            all_keys.extend(ks)
        keys = all_keys
    for k in keys:
        if hasattr(args, k):
            v = getattr(args, k)
            out[k] = v
    return out
