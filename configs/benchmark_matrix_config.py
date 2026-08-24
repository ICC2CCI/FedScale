#!/usr/bin/env python3
"""Compile and validate the benchmark-matrix Flower run configuration.

Keeping this logic out of the supervisor shell script makes each submitted
configuration a typed mapping before it is rendered for ``flwr run``.  In
particular, a run can never receive two values for the same key (the previous
FedScale path accidentally emitted both compression modes).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass


# FedScale reuses the FSDP local trainer and changes the update wire codec;
# the user-facing supervisor still exposes it as a first-class experiment
# mode alongside the ordinary FSDP and DDP modes.
STRATEGIES = ("fsdp", "fedscale", "ddp")
MATRIX_ROUNDS = (10, 20)


@dataclass(frozen=True)
class BenchmarkDefaults:
    model_name: str = "Qwen/Qwen2.5-7B"
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    finetuning_type: str = "lora"
    # Cap each client partition to keep the tokenized Arrow cache and its mmap
    # footprint small. Even with 7B, the full 93k-sample UltraChat partition
    # produces a >2-GiB Arrow file whose mmap pages raise SIGBUS under memory
    # pressure during the first training step.
    max_train_samples: int = 300
    max_steps: int = 1
    num_eval_samples: int = 1
    federated_round_timeout_seconds: int = 43200
    fedscale_block_size: int = 1048576
    fedscale_mask_ratio: float = 0.0001


DEFAULTS = BenchmarkDefaults()


class ConfigError(ValueError):
    """Raised before a malformed or incompatible Flower configuration ships."""


def _as_mapping(entries: Iterable[tuple[str, object]]) -> dict[str, object]:
    """Convert ordered entries to a mapping while rejecting duplicate keys."""
    config: dict[str, object] = {}
    for key, value in entries:
        if key in config:
            raise ConfigError(f"duplicate Flower run-config key: {key}")
        config[key] = value
    return config


def compile_run_config(
    *,
    strategy: str,
    rounds: int,
    experiment_id: str,
    results_root: str,
    resume_round: int = 0,
    model_name: str | None = None,
    dataset_name: str | None = None,
    finetuning_type: str | None = None,
    defaults: BenchmarkDefaults = DEFAULTS,
) -> dict[str, object]:
    """Produce one valid benchmark configuration as a unique, typed mapping."""
    if strategy not in STRATEGIES:
        raise ConfigError(f"unsupported strategy {strategy!r}; expected one of {STRATEGIES}")
    if rounds <= 0:
        raise ConfigError("num-server-rounds must be positive")
    if resume_round < 0:
        raise ConfigError("resume-round cannot be negative")
    if not experiment_id:
        raise ConfigError("experiment-id cannot be empty")
    if not results_root.startswith("/"):
        raise ConfigError("results-root must be an absolute path")
    model_name = model_name or defaults.model_name
    dataset_name = dataset_name or defaults.dataset_name
    finetuning_type = (finetuning_type or defaults.finetuning_type).lower()
    if finetuning_type not in {"lora", "full"}:
        raise ConfigError("finetuning-type must be either 'lora' or 'full'")

    # FedScale uses the FSDP local trainer and replaces its update uplink with
    # public canonical-block INT8 deltas.
    distributed_strategy = "fsdp" if strategy == "fedscale" else strategy
    compression = "fedscale-int8" if strategy == "fedscale" else "none"
    entries: list[tuple[str, object]] = [
        ("model.name", model_name),
        ("dataset.name", dataset_name),
        ("train.distributed-strategy", distributed_strategy),
        ("num-server-rounds", rounds),
        ("train.save-every-round", 1),
        ("experiment-id", experiment_id),
        ("model.finetuning-type", finetuning_type),
        # quantization=0 means FP16 LoRA in the training image.
        ("model.quantization", 0),
        ("dataset.max-train-samples", defaults.max_train_samples),
        ("train.training-arguments.per-device-train-batch-size", 1),
        ("train.training-arguments.gradient-accumulation-steps", 1),
        ("train.training-arguments.max-steps", defaults.max_steps),
        ("train.evaluate-after-fit", False),
        ("train.num-eval-samples", defaults.num_eval_samples),
        ("train.full-update-compression", compression),
        ("train.federated-round-timeout-seconds", defaults.federated_round_timeout_seconds),
    ]
    if strategy == "fedscale":
        entries.extend(
            [
                ("train.fedscale-block-size", defaults.fedscale_block_size),
                ("train.fedscale-mask-ratio", defaults.fedscale_mask_ratio),
            ]
        )
        if finetuning_type == "full":
            # The sparse full-model path needs all three sites to start from
            # the same verified local base model on a fresh run.
            entries.append(("train.full-local-initialization", True))
    if resume_round > 0:
        entries.extend(
            [
                ("train.resume-peft-path", f"{results_root}/{experiment_id}/peft_{resume_round}"),
                ("train.resume-round", resume_round),
            ]
        )
    config = _as_mapping(entries)
    validate_run_config(config, benchmark_strategy=strategy)
    return config


def validate_run_config(config: dict[str, object], *, benchmark_strategy: str) -> None:
    """Enforce the invariants the benchmark supervisor relies on."""
    if benchmark_strategy not in STRATEGIES:
        raise ConfigError(f"unknown benchmark strategy: {benchmark_strategy}")
    required_positive = (
        "num-server-rounds",
        "train.training-arguments.max-steps",
        "train.num-eval-samples",
        "train.federated-round-timeout-seconds",
    )
    for key in required_positive:
        if not isinstance(config.get(key), int) or int(config[key]) <= 0:
            raise ConfigError(f"{key} must be a positive integer")
    max_train_samples = config.get("dataset.max-train-samples")
    if not isinstance(max_train_samples, int) or int(max_train_samples) < 0:
        raise ConfigError("dataset.max-train-samples must be a non-negative integer")
    if config.get("model.finetuning-type") not in {"lora", "full"} or config.get("model.quantization") != 0:
        raise ConfigError("matrix requires FP16 LoRA or full fine-tuning")

    expected_distributed = "fsdp" if benchmark_strategy == "fedscale" else benchmark_strategy
    if config.get("train.distributed-strategy") != expected_distributed:
        raise ConfigError(f"{benchmark_strategy} has an invalid distributed strategy")
    expected_compression = "fedscale-int8" if benchmark_strategy == "fedscale" else "none"
    if config.get("train.full-update-compression") != expected_compression:
        raise ConfigError(f"{benchmark_strategy} has an invalid update codec")

    has_resume = "train.resume-round" in config or "train.resume-peft-path" in config
    if "train.full-local-initialization" in config and config.get("model.finetuning-type") != "full":
        raise ConfigError("full-model local initialization requires full fine-tuning")
    if has_resume:
        if not isinstance(config.get("train.resume-round"), int) or int(config["train.resume-round"]) <= 0:
            raise ConfigError("checkpoint resume requires a positive resume-round")
        if not isinstance(config.get("train.resume-peft-path"), str) or not config["train.resume-peft-path"]:
            raise ConfigError("checkpoint resume requires resume-peft-path")

    fedscale_keys = {"train.fedscale-block-size", "train.fedscale-mask-ratio"}
    if benchmark_strategy == "fedscale":
        if not fedscale_keys.issubset(config):
            raise ConfigError("FedScale requires block-size and mask-ratio")
    elif fedscale_keys.intersection(config):
        raise ConfigError("FedScale-only options leaked into a non-FedScale run")


def render_flower_run_config(config: dict[str, object]) -> str:
    """Render a configuration using JSON literals accepted by Flower's parser."""
    return " ".join(f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in config.items())


def _compile_command(args: argparse.Namespace) -> int:
    config = compile_run_config(
        strategy=args.strategy,
        rounds=args.rounds,
        experiment_id=args.experiment_id,
        results_root=args.results_root,
        resume_round=args.resume_round,
        model_name=args.model,
        dataset_name=args.dataset,
        finetuning_type=args.finetuning_type,
    )
    print(json.dumps(config, sort_keys=True) if args.json else render_flower_run_config(config))
    return 0


def _validate_matrix_command() -> int:
    for rounds in MATRIX_ROUNDS:
        for strategy in STRATEGIES:
            config = compile_run_config(
                strategy=strategy,
                rounds=rounds,
                experiment_id=f"benchmark-{strategy}-r{rounds}",
                results_root="/app/results",
            )
            validate_run_config(config, benchmark_strategy=strategy)
    print(f"validated {len(MATRIX_ROUNDS) * len(STRATEGIES)} configurable matrix configurations")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile", help="compile one Flower run-config")
    compile_parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    compile_parser.add_argument("--rounds", type=int, required=True)
    compile_parser.add_argument("--experiment-id", required=True)
    compile_parser.add_argument("--results-root", required=True)
    compile_parser.add_argument("--resume-round", type=int, default=0)
    compile_parser.add_argument("--model", default=None)
    compile_parser.add_argument("--dataset", default=None)
    compile_parser.add_argument("--finetuning-type", choices=("lora", "full"), default=None)
    compile_parser.add_argument("--json", action="store_true", help="emit typed JSON instead of Flower syntax")
    subparsers.add_parser("validate-matrix", help="validate all 15 matrix configurations")
    args = parser.parse_args()
    try:
        return _compile_command(args) if args.command == "compile" else _validate_matrix_command()
    except ConfigError as exc:
        print(f"invalid benchmark configuration: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
