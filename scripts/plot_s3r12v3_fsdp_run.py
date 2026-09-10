#!/usr/bin/env python3
"""从一次运行目录绘制 train/eval/时间/传输/下载模式图。

用法:
  python scripts/plot_s3r12v3_fsdp_run.py results/202609101006
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _client_series(log: List[Dict[str, Any]], cid: str, key: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for r in log:
        clients = ((r.get("timing_s") or {}).get("clients") or {})
        ct = clients.get(str(cid)) or clients.get(cid) or {}
        # also check top-level client timings nested under timing after merge
        v = ct.get(key)
        if v is None:
            ctl = (r.get("client_train_loss") or {}).get(str(cid))
            if key == "train_loss":
                v = ctl
        out.append(float(v) if v is not None else None)
    return out


def _transfer(log: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
    out = []
    for r in log:
        tr = ((r.get("timing_s") or {}).get("transfer") or {})
        v = tr.get(key)
        out.append(float(v) if v is not None else None)
    return out


def _mode_label(v: Optional[float]) -> str:
    if v is None:
        return "?"
    iv = int(round(v))
    return {0: "cache", 1: "delta", 2: "full", 3: "local_base"}.get(iv, str(iv))


def plot_run(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "round_log.json"
    if not log_path.exists():
        raise SystemExit(f"missing {log_path}")
    log: List[Dict[str, Any]] = _load_json(log_path)
    if not log:
        raise SystemExit("round_log.json empty")

    rounds = [int(r["round"]) for r in log]
    avg = [float(r["avg_train_loss"]) for r in log]
    c0 = [float((r.get("client_train_loss") or {}).get("0", np.nan)) for r in log]
    c1 = [float((r.get("client_train_loss") or {}).get("1", np.nan)) for r in log]
    wall = [float((r.get("timing_s") or {}).get("round_wall_s") or np.nan) for r in log]

    # --- train loss ---
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.plot(rounds, avg, "o-", lw=1.8, ms=4.5, color="#1f4e79", label="avg")
    ax.plot(rounds, c0, "s--", lw=1.2, ms=3.5, color="#2a9d8f", label="ICC1 (c0)")
    ax.plot(rounds, c1, "^--", lw=1.2, ms=3.5, color="#e76f51", label="ICC2 (c1)")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("train_loss")
    ax.set_title(f"{run_dir.name} — train loss")
    ax.set_xticks(rounds)
    # 从 0 起笔，避免自动截断 y 轴造成“掉到接近 0”的错觉
    ymax = float(np.nanmax([*avg, *c0, *c1]))
    ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "train_loss.png")
    plt.close(fig)

    # --- transfer ---
    # 说明：
    # - download_global_MiB：训练前拉取（cache 时为 0）
    # - post_delta_MiB：聚合后拉取的 global_delta（稳态主要下行）
    # - 两端行为应对称；勿把 global_delta（服务器写出大小）误当成某一端独有下载
    up_total = _transfer(log, "upload_blocks_MiB_total")
    g_full = _transfer(log, "global_state_MiB")
    g_delta = _transfer(log, "global_delta_MiB")
    d0 = _client_series(log, "0", "download_global_MiB")
    d1 = _client_series(log, "1", "download_global_MiB")
    p0 = _client_series(log, "0", "post_delta_MiB")
    p1 = _client_series(log, "1", "post_delta_MiB")
    # 本轮实际下行 = 训练前下载 + 聚合后 delta
    recv0 = [
        (a or 0.0) + (b or 0.0) if (a is not None or b is not None) else None
        for a, b in zip(d0, p0)
    ]
    recv1 = [
        (a or 0.0) + (b or 0.0) if (a is not None or b is not None) else None
        for a, b in zip(d1, p1)
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.plot(rounds, up_total, "o-", label="upload total (2 clients)")
    if any(v is not None for v in g_delta):
        ax.plot(rounds, g_delta, "s-", label="global_delta (server write)")
    ax.plot(rounds, g_full, "^--", alpha=0.45, label="global_state full (server write)")
    if any(v is not None for v in recv0):
        ax.plot(rounds, recv0, "x-", label="ICC1 recv (=pre-dl + post_delta)")
    if any(v is not None for v in recv1):
        ax.plot(rounds, recv1, "+-", label="ICC2 recv (=pre-dl + post_delta)")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("MiB")
    ax.set_title(f"{run_dir.name} — transfer")
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "transfer_mib.png")
    plt.close(fig)

    # --- time breakdown (from client 0 if present) ---
    keys = [
        ("download_global_s", "download"),
        ("broadcast_load_s", "load"),
        ("train_local_s", "train"),
        ("encode_delta_s", "encode"),
        ("upload_minio_s", "upload"),
        ("wait_aggregate_s", "wait_agg"),
        ("post_delta_s", "post_delta"),
    ]
    stacks = {lab: [] for _, lab in keys}
    for r in log:
        ct = ((r.get("timing_s") or {}).get("clients") or {}).get("0") or {}
        # wait/post may be only in final timing merge on server for client
        for k, lab in keys:
            v = ct.get(k)
            if v is None and k in ("wait_aggregate_s", "post_delta_s"):
                # sometimes only in round timing after client timing POST
                v = ct.get(k)
            stacks[lab].append(float(v) if v is not None else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    bottom = np.zeros(len(rounds))
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a", "#355070"]
    for (k, lab), color in zip(keys, colors):
        vals = np.array(stacks[lab])
        ax.bar(rounds, vals, bottom=bottom, label=lab, color=color, width=0.7)
        bottom += vals
    ax.plot(rounds, wall, "k--", lw=1.2, label="round_wall")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("seconds")
    ax.set_title(f"{run_dir.name} — time (client0 stack + wall)")
    ax.set_xticks(rounds)
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "time_breakdown.png")
    plt.close(fig)

    # --- download mode ---
    modes = _client_series(log, "0", "download_mode")
    if any(v is not None for v in modes):
        fig, ax = plt.subplots(figsize=(9, 3.6), dpi=140)
        ax.step(rounds, modes, where="mid", color="#1f4e79")
        ax.scatter(rounds, modes, c="#1f4e79", s=30)
        ax.set_yticks([0, 1, 2, 3], ["cache", "delta", "full", "local_base"])
        ax.set_xlabel("Federated round")
        ax.set_title(f"{run_dir.name} — download mode (ICC1)")
        ax.set_xticks(rounds)
        ax.set_ylim(-0.2, 3.2)
        ax.grid(True, alpha=0.3)
        for r, m in zip(rounds, modes):
            ax.annotate(_mode_label(m), (r, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "download_mode.png")
        plt.close(fig)

    # --- eval loss: prefer round_log online field, else offline eval json ---
    er: List[int] = []
    el: List[float] = []
    for r in log:
        if r.get("eval_loss") is not None:
            er.append(int(r["round"]))
            el.append(float(r["eval_loss"]))
    eval_path = run_dir / "eval" / "eval_by_round.json"
    if not er and not eval_path.exists():
        eval_path = run_dir / "eval_by_round.json"
    if not er and eval_path.exists():
        ev = _load_json(eval_path)
        if isinstance(ev, list) and ev:
            er = [int(x["round"]) for x in ev if x.get("eval_loss") is not None]
            el = [float(x["eval_loss"]) for x in ev if x.get("eval_loss") is not None]
    if er:
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
        ax.plot(er, el, "o-", color="#6d597a", lw=1.8)
        ax.set_xlabel("Federated round")
        ax.set_ylabel("eval_loss")
        ax.set_title(f"{run_dir.name} — eval loss")
        # 从 0 起笔，避免自动截断 y 轴造成“掉到接近 0”的错觉
        ymax = float(max(el)) if el else 1.0
        ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "eval_loss.png")
        plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "n_rounds": len(log),
        "train_loss_first": avg[0],
        "train_loss_last": avg[-1],
        "avg_round_wall_s": float(np.nanmean(wall)),
        "figures": sorted(p.name for p in fig_dir.glob("*.png")),
    }
    (run_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    plot_run(args.run_dir)


if __name__ == "__main__":
    main()
