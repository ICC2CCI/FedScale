#!/usr/bin/env python3
"""Render a compact comparison from exported benchmark experiment metadata."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def load(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {}

def metric(summary, key):
    values = summary.get("aggregated_client_train_metrics", {}).values()
    vals = [x.get(key) for x in values if isinstance(x.get(key), (int, float))]
    return sum(vals) if vals else None

parser = argparse.ArgumentParser()
parser.add_argument("matrix_dir", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
manifest = load(args.matrix_dir / "matrix-manifest.json")
lines = [f"# 基准矩阵：{manifest.get('matrix_id', args.matrix_dir.name)}", "", "| 策略 | 轮数 | 状态 | PPL | Macro-F1 | 增量字节合计 | 训练秒合计 | 非训练秒合计 |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
for entry in manifest.get("experiments", []):
    root = args.matrix_dir.parent / entry["experiment_id"]
    summary, evaluation = load(root / "experiment_summary.json"), load(root / "evaluation_summary.json")
    final = summary.get("final_evaluation_metrics", {})
    lines.append("| {strategy} | {rounds} | {status} | {ppl} | {f1} | {bytes} | {train} | {other} |".format(
        strategy=entry["strategy"], rounds=entry["rounds"], status=entry.get("status", "-"),
        ppl=final.get("perplexity", "-"), f1=final.get("macro_f1", "-"),
        bytes=metric(summary, "model_delta_bytes") or "-", train=metric(summary, "client_training_seconds") or "-", other=metric(summary, "client_non_training_seconds") or "-"))
(args.output or args.matrix_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
