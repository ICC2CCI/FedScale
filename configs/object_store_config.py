#!/usr/bin/env python3
"""Compile-time guardrails for dense full-model object-store experiments."""

from __future__ import annotations

import argparse
import sys
from urllib.parse import urlparse


class ConfigError(ValueError):
    """Raised before an incompatible run can reserve client GPUs."""


def validate(args: argparse.Namespace) -> None:
    if args.strategy not in {"ddp", "fsdp"}:
        raise ConfigError("object-store transport supports ddp or fsdp only")
    if args.rounds < 1:
        raise ConfigError("rounds must be positive")
    if args.finetuning_type != "full" or args.quantization != 0:
        raise ConfigError("object-store transport requires full fine-tuning with quantization=0")
    if args.compression != "none":
        raise ConfigError("object-store transport requires full-update-compression=none")
    if args.save_every_round != 1:
        raise ConfigError("object-store MVP requires save-every-round=1")
    if args.resume_round < 0:
        raise ConfigError("resume-round cannot be negative")
    parsed = urlparse(args.initial_global_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ConfigError("object-store-initial-global-uri must be a non-empty s3:// URI")
    if "'" in args.initial_global_uri:
        raise ConfigError("object-store-initial-global-uri cannot contain a single quote")
    protected = {
        "model.finetuning-type": "full",
        "model.quantization": "0",
        "train.full-update-compression": "none",
        "train.full-update-transport": "object-store",
        "train.save-every-round": "1",
    }
    seen: set[str] = set()
    for item in args.override:
        if "=" not in item:
            raise ConfigError(f"invalid --set override: {item!r}")
        key, value = item.split("=", 1)
        if key in seen:
            raise ConfigError(f"duplicate --set key: {key}")
        seen.add(key)
        if key in protected:
            raise ConfigError(
                f"{key} is fixed by object-store mode; do not pass it again with --set"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--finetuning-type", required=True)
    parser.add_argument("--quantization", type=int, required=True)
    parser.add_argument("--compression", required=True)
    parser.add_argument("--save-every-round", type=int, required=True)
    parser.add_argument("--initial-global-uri", required=True)
    parser.add_argument("--resume-round", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    try:
        validate(args)
    except ConfigError as exc:
        print(f"invalid object-store configuration: {exc}", file=sys.stderr)
        return 2
    print("object-store configuration validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
