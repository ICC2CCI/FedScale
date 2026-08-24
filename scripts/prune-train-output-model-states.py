#!/usr/bin/env python3
"""Prune old per-round model states while preserving experiment metadata.

Only these top-level state files are removed: ``initial_model_state.pt``,
``initial_weights.pt``, and ``model_weights.pt``.  Metrics, logs, source files,
evaluation artifacts, and output directories are deliberately left in place.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TRAIN_DIR = re.compile(r"^train-round-(\d+)-(\d+)-(\d+)(?:-a(\d+))?$")
STATE_FILES = {
    "initial_model_state.pt",
    "initial_weights.pt",
    "model_weights.pt",
}


@dataclass(frozen=True)
class Attempt:
    path: Path
    name: str
    round_number: int
    client: str
    timestamp: int
    attempt: int
    modified: float
    successful: bool


def allocated_bytes(path: Path) -> int:
    return path.stat().st_blocks * 512


def collect_attempts(root: Path, cutoff: float) -> list[Attempt]:
    attempts: list[Attempt] = []
    for path in root.glob("train-round-*"):
        if not path.is_dir() or path.stat().st_mtime > cutoff:
            continue
        match = TRAIN_DIR.match(path.name)
        if not match:
            continue
        files = {entry.name for entry in path.iterdir() if entry.is_file()}
        attempts.append(
            Attempt(
                path=path,
                name=path.name,
                round_number=int(match.group(1)),
                client=match.group(2),
                timestamp=int(match.group(3)),
                attempt=int(match.group(4) or 1),
                modified=path.stat().st_mtime,
                successful={"model_weights.pt", "metrics.json"}.issubset(files),
            )
        )
    return attempts


def kept_successful_attempts(attempts: list[Attempt], keep_count: int) -> set[str]:
    """Return latest successful attempts for each monotonically increasing run."""
    by_logical_round: dict[tuple[int, str, int], list[Attempt]] = defaultdict(list)
    for item in attempts:
        by_logical_round[(item.round_number, item.client, item.timestamp)].append(item)

    logical_rounds: list[tuple[int, int, float, Attempt | None]] = []
    for same_round in by_logical_round.values():
        same_round.sort(key=lambda item: item.attempt)
        completed = [item for item in same_round if item.successful]
        logical_rounds.append(
            (
                same_round[0].round_number,
                same_round[0].timestamp,
                min(item.modified for item in same_round),
                completed[-1] if completed else None,
            )
        )

    logical_rounds.sort(key=lambda item: (item[2], item[1]))
    experiments: list[list[tuple[int, int, float, Attempt | None]]] = []
    current: list[tuple[int, int, float, Attempt | None]] = []
    previous_round: int | None = None
    for logical_round in logical_rounds:
        if current and logical_round[0] <= previous_round:  # a new experiment
            experiments.append(current)
            current = []
        current.append(logical_round)
        previous_round = logical_round[0]
    if current:
        experiments.append(current)

    kept: set[str] = set()
    for experiment in experiments:
        completed = [item[3] for item in experiment if item[3] is not None]
        kept.update(item.name for item in completed[-keep_count:])
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/app/outputs"))
    parser.add_argument("--min-age-hours", type=float, default=24)
    parser.add_argument("--keep-successful", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.keep_successful < 1 or args.min_age_hours < 0:
        parser.error("retention values must be positive")
    if not args.root.is_dir():
        parser.error(f"output root does not exist: {args.root}")

    attempts = collect_attempts(args.root, time.time() - args.min_age_hours * 3600)
    kept = kept_successful_attempts(attempts, args.keep_successful)
    targets: list[tuple[Path, str]] = []
    for item in attempts:
        if item.successful and item.name in kept:
            continue
        for name in STATE_FILES:
            path = item.path / name
            if path.is_file():
                targets.append((path, "failed_or_incomplete" if not item.successful else "old_success"))

    summary = {
        "dry_run": args.dry_run,
        "eligible_train_directories": len(attempts),
        "kept_successful_directories": len(kept),
        "state_files": len(targets),
        "allocated_gib": round(sum(allocated_bytes(path) for path, _ in targets) / 1024**3, 2),
        "by_reason": {},
    }
    for reason in ("failed_or_incomplete", "old_success"):
        files = [path for path, item_reason in targets if item_reason == reason]
        summary["by_reason"][reason] = {
            "files": len(files),
            "allocated_gib": round(sum(allocated_bytes(path) for path in files) / 1024**3, 2),
        }
    if not args.dry_run:
        for path, _ in targets:
            path.unlink()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
