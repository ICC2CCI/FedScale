#!/usr/bin/env python3
"""Render a concise Chinese Markdown report from exported experiment JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def md(value) -> str:
    text = "未记录" if value in (None, "") else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def duration_text(seconds) -> str:
    if seconds in (None, ""):
        return "未记录"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分 {secs} 秒"
    return f"{minutes} 分 {secs} 秒"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.result_dir
    output = args.output or root / "REPORT.md"
    experiment_config = load_json(root / "experiment_config.json", {})
    if not experiment_config:
        # Read-only compatibility for experiments created before the clearer
        # experiment_config.json naming was introduced.
        experiment_config = load_json(root / "experiment_manifest.json", {})
    summary = load_json(root / "experiment_summary.json", {})
    state = load_json(root / "experiment_state.json", {})
    timings = summary.get("round_timings") or load_json(
        root / "federated_timings.json", []
    )
    attempts = load_json(root / "experiment_attempts.json", [])
    config = experiment_config.get("run_config", {})

    experiment_id = (
        experiment_config.get("experiment_id")
        or summary.get("experiment_id")
        or state.get("experiment_id")
        or root.name
    )
    status = summary.get("status") or state.get("status")
    run_id = experiment_config.get("run_id") or summary.get("run_id") or state.get("run_id")
    model = experiment_config.get("model") or config.get("model.name")
    dataset = experiment_config.get("dataset") or config.get("dataset.name")
    strategy = experiment_config.get("distributed_strategy") or config.get(
        "train.distributed-strategy"
    )
    rounds = (
        summary.get("completed_global_round")
        or state.get("latest_completed_round")
        or state.get("total_rounds")
        or experiment_config.get("rounds_this_attempt")
    )
    duration = summary.get("duration_seconds", state.get("duration_seconds"))
    checkpoint = summary.get("latest_checkpoint") or state.get("latest_checkpoint")
    started_at = summary.get("started_at") or experiment_config.get("started_at") or state.get(
        "started_at"
    )
    finished_at = summary.get("finished_at") or state.get("finished_at") or state.get(
        "updated_at"
    )
    train_metrics = summary.get("aggregated_client_train_metrics", {})

    def sum_round_metric(key):
        values = [item.get(key) for item in train_metrics.values()]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        return sum(numeric) if numeric else None

    pure_training_critical_s = sum_round_metric(
        "critical_path_client_training_seconds"
    )
    in_round_evaluation_critical_s = sum_round_metric(
        "critical_path_client_evaluation_seconds"
    )
    legacy_round_timing = (
        summary.get("round_timing_scope") == "legacy_inter_evaluate_start"
    )

    lines = [
        f"# 联邦实验报告：{experiment_id}",
        "",
        "## 概览",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| Experiment ID | `{md(experiment_id)}` |",
        f"| Flower Run ID | `{md(run_id)}` |",
        f"| 状态 | `{md(status)}` |",
        f"| 模型 | `{md(model)}` |",
        f"| 数据集 | `{md(dataset)}` |",
        f"| 分布式策略 | `{md(strategy)}` |",
        f"| 已完成全局轮数 | {md(rounds)} |",
        f"| 开始时间 | `{md(started_at)}` |",
        f"| 结束时间 | `{md(finished_at)}` |",
        f"| 实验端到端墙钟总耗时 | {duration_text(duration)} |",
        f"| 纯训练关键路径合计 | {duration_text(pure_training_critical_s)} |",
        f"| 轮内评估关键路径合计 | {duration_text(in_round_evaluation_critical_s)} |",
        f"| 最终 checkpoint | `{md(checkpoint)}` |",
        "",
        "端到端墙钟时间包含调度、模型准备、纯训练、可选轮内评估、结果回传、FedAvg 和 checkpoint；纯训练耗时不包含评估。",
        "",
        "## 逐轮联邦周期耗时（非纯训练）",
        "",
        "联邦周期包含消息下发、客户端准备/训练/可选评估、结果回传和聚合；纯训练时间见下一节。",
        *(
            ["该历史实验使用旧计时器：表中周期是两次服务端回调开始时刻之差，可能混入上一轮 checkpoint 时间。"]
            if legacy_round_timing
            else []
        ),
        "",
        "| 全局轮次 | 联邦周期（秒） | 联邦周期 | 服务端聚合后处理 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    if timings:
        for item in timings:
            seconds = item.get("federated_cycle_s", item.get("t_round_total_s"))
            lines.append(
                f"| {md(item.get('round'))} | {md(seconds)} | "
                f"{duration_text(seconds)} | "
                f"{duration_text(item.get('server_post_aggregation_s'))} |"
            )
    else:
        lines.append("| - | - | 未记录 | 未记录 |")

    lines.extend(
        [
            "",
            "## 客户端训练指标（按客户端训练集大小加权）",
            "",
            "这里的 Train loss 是客户端训练阶段 loss 的加权均值，不是聚合后全局模型的验证 loss。",
            "",
            "| 全局轮次 | Train loss | 纯训练耗时/客户端均值 | 纯训练关键路径 | 轮内评估耗时/客户端 | 客户端轮次关键路径 | 数据集训练条数/客户端 | 估算样本呈现数/客户端 | 优化步 | World size | Gang attempt |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if train_metrics:
        for round_id in sorted(train_metrics, key=lambda item: int(item)):
            item = train_metrics[round_id]
            lines.append(
                f"| {md(round_id)} | {md(item.get('train_loss'))} | "
                f"{duration_text(item.get('client_training_seconds'))} | "
                f"{duration_text(item.get('critical_path_client_training_seconds'))} | "
                f"{duration_text(item.get('client_evaluation_seconds'))} | "
                f"{duration_text(item.get('critical_path_client_round_seconds'))} | "
                f"{md(item.get('dataset_train_samples_per_client'))} | "
                f"{md(item.get('estimated_sample_presentations_per_client'))} | "
                f"{md(item.get('optimizer_steps'))} | "
                f"{md(item.get('distributed_world_size'))} | "
                f"{md(item.get('gang_attempt'))} |"
            )
    else:
        lines.append(
            "| - | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | "
            "未记录 | 未记录 | 未记录 | 未记录 |"
        )

    lines.extend(["", "## 有效参数配置", "", "| 参数 | 值 |", "| --- | --- |"])
    if config:
        for key in sorted(config):
            lines.append(f"| `{md(key)}` | `{md(config[key])}` |")
    else:
        lines.append("| - | 该历史实验没有自动保存 run-config |")

    lines.extend(["", "## 恢复尝试", ""])
    if attempts:
        lines.extend(
            [
                "| Run ID | 状态 | 恢复轮次 | 本次轮数 | 耗时 |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for attempt in attempts:
            lines.append(
                f"| `{md(attempt.get('run_id'))}` | {md(attempt.get('status'))} | "
                f"{md(attempt.get('resume_round'))} | "
                f"{md(attempt.get('rounds_this_attempt'))} | "
                f"{duration_text(attempt.get('duration_seconds'))} |"
            )
    else:
        lines.append("该历史实验没有自动保存恢复尝试记录。")

    lines.extend(
        [
            "",
            "## 文件口径",
            "",
            "本报告是中心结果 PVC 中 JSON 元数据的本地可读镜像。全局 LoRA 权重仍以",
            f"`/app/results/{experiment_id}/peft_N/`（LoRA）或 `full_N/`（全参数）为准，",
            "本地导出默认不复制模型权重。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
