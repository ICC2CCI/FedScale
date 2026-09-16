"""Evaluation Dashboard — FastAPI web app for browsing training run evaluation reports.

Deploy on the central server::

    cd ~/fedscale-eval
    /home/pcllgr/miniconda3/envs/fedscale-server/bin/python -m evaluation_dashboard --port 8060

Features:
  - Lists all training runs in results/ directory
  - Shows evaluation report for each run (training perf, resources, federated timing, model accuracy)
  - Loss trends displayed as line charts (Chart.js)
  - Switch between different training versions
  - Shows offline generative evaluation results (ROUGE-L, BERTScore, PPL)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="FedScale Evaluation Dashboard")


def _scan_runs(results_dir: Path) -> list[dict]:
    """Scan the results directory for all training runs."""
    runs = []
    if not results_dir.exists():
        return runs
    for d in sorted(results_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        # Skip symlinks (e.g. "current" -> latest run) to avoid duplicates
        if d.is_symlink():
            continue
        # Only include directories that contain a round_log.json or run_meta.json
        round_log_path = d / "round_log.json"
        meta_path = d / "run_meta.json"
        if not round_log_path.exists() and not meta_path.exists():
            continue
        run_info: dict[str, Any] = {
            "run_id": d.name,
            "path": str(d),
            "has_round_log": round_log_path.exists(),
            "has_metrics_detailed": False,
            "has_offline_eval": False,
            "num_rounds": 0,
            "status": "unknown",
        }
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                run_info.update({
                    "num_rounds": meta.get("num_rounds", 0),
                    "num_clients": meta.get("num_clients", 0),
                    "model": meta.get("compressor", ""),
                    "coverage_h": meta.get("coverage_h", ""),
                    "tag": meta.get("tag", ""),
                })
            except Exception:
                pass
        if round_log_path.exists():
            try:
                rl = json.loads(round_log_path.read_text(encoding="utf-8"))
                run_info["completed_rounds"] = len(rl)
                run_info["status"] = "completed" if len(rl) >= run_info.get("num_rounds", 0) else "partial"
                if rl:
                    run_info["final_train_loss"] = rl[-1].get("avg_train_loss")
                    run_info["final_eval_loss"] = rl[-1].get("eval_loss")
            except Exception:
                pass
        # Check for metrics_detailed
        for name in ["metrics_detailed.json", "metrics_detailed_client0.json"]:
            if (d / name).exists():
                run_info["has_metrics_detailed"] = True
                break
        # Check for offline eval
        offline_eval_path = d / "offline_eval" / "offline_eval_summary.json"
        if offline_eval_path.exists():
            run_info["has_offline_eval"] = True
        runs.append(run_info)
    return runs


def _load_run_data(run_dir: Path) -> dict:
    """Load all evaluation data for a specific run."""
    data: dict[str, Any] = {"run_id": run_dir.name}

    # Round log
    round_log_path = run_dir / "round_log.json"
    if round_log_path.exists():
        data["round_log"] = json.loads(round_log_path.read_text(encoding="utf-8"))
    else:
        data["round_log"] = []

    # Run meta
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        data["run_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        data["run_meta"] = {}

    # Metrics detailed (client 0)
    md_path = run_dir / "metrics_detailed.json"
    if not md_path.exists():
        md_path = run_dir / "metrics_detailed_client0.json"
    if md_path.exists():
        data["metrics_detailed"] = json.loads(md_path.read_text(encoding="utf-8"))
    else:
        data["metrics_detailed"] = None

    # Client 1 metrics
    md1_path = run_dir / "metrics_detailed_client1.json"
    if md1_path.exists():
        data["metrics_detailed_client1"] = json.loads(md1_path.read_text(encoding="utf-8"))
    else:
        data["metrics_detailed_client1"] = None

    # Federated timings
    ft_path = run_dir / "federated_timings.json"
    if ft_path.exists():
        data["federated_timings"] = json.loads(ft_path.read_text(encoding="utf-8"))
    else:
        data["federated_timings"] = []

    # Experiment summary
    es_path = run_dir / "experiment_summary.json"
    if es_path.exists():
        data["experiment_summary"] = json.loads(es_path.read_text(encoding="utf-8"))
    else:
        data["experiment_summary"] = None

    # Offline eval
    offline_path = run_dir / "offline_eval" / "offline_eval_summary.json"
    if offline_path.exists():
        data["offline_eval"] = json.loads(offline_path.read_text(encoding="utf-8"))
    else:
        data["offline_eval"] = None

    # Report
    report_path = run_dir / "report.json"
    if report_path.exists():
        data["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        data["report"] = None

    return data


def _get_results_dir() -> Path:
    """Get the results directory path."""
    return Path(os.environ.get("EVAL_DASHBOARD_RESULTS_DIR", REPO_ROOT / "results"))


@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve the dashboard HTML page."""
    results_dir = _get_results_dir()
    runs = _scan_runs(results_dir)
    return _render_html(runs, selected_run=None, run_data=None)


@app.get("/api/runs")
async def api_list_runs():
    """API: List all training runs."""
    results_dir = _get_results_dir()
    runs = _scan_runs(results_dir)
    return JSONResponse(runs)


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: str):
    """API: Get detailed data for a specific run."""
    results_dir = _get_results_dir()
    run_dir = results_dir / run_id
    if not run_dir.exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)
    data = _load_run_data(run_dir)
    # Merge summary fields from _scan_runs
    runs = _scan_runs(results_dir)
    for r in runs:
        if r["run_id"] == run_id:
            for k in ("num_rounds", "status", "num_clients", "model",
                       "coverage_h", "tag", "completed_rounds",
                       "final_train_loss", "final_eval_loss"):
                if k in r:
                    data[k] = r[k]
            break
    return JSONResponse(data)


@app.get("/view/{run_id}", response_class=HTMLResponse)
async def view_run(run_id: str):
    """View a specific run's evaluation report."""
    results_dir = _get_results_dir()
    runs = _scan_runs(results_dir)
    run_dir = results_dir / run_id
    if not run_dir.exists():
        return HTMLResponse("<h1>Run not found</h1>", status_code=404)
    run_data = _load_run_data(run_dir)
    return _render_html(runs, selected_run=run_id, run_data=run_data)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _render_html(runs: list[dict], selected_run: str | None, run_data: dict | None) -> str:
    runs_json = json.dumps(runs, ensure_ascii=False)
    run_data_json = json.dumps(run_data, ensure_ascii=False) if run_data else "null"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FedScale 联邦训练评估面板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
.layout {{ display: flex; min-height: 100vh; }}
.sidebar {{ width: 280px; background: #161b22; border-right: 1px solid #30363d; padding: 20px; overflow-y: auto; }}
.main {{ flex: 1; padding: 24px; overflow-y: auto; }}
h1 {{ color: #58a6ff; font-size: 1.3em; margin-bottom: 16px; }}
h2 {{ color: #79c0ff; font-size: 1.1em; margin: 24px 0 10px 0; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
h3 {{ color: #d2a8ff; font-size: 0.95em; margin: 16px 0 8px 0; }}
.run-item {{ padding: 10px 12px; border-radius: 6px; cursor: pointer; margin-bottom: 6px; border: 1px solid transparent; transition: all 0.15s; }}
.run-item:hover {{ background: #21262d; border-color: #30363d; }}
.run-item.active {{ background: #1f6feb33; border-color: #1f6feb; }}
.run-id {{ font-weight: 600; color: #e6edf3; font-size: 0.85em; }}
.run-meta {{ font-size: 0.75em; color: #8b949e; margin-top: 2px; }}
.tag {{ display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 0.7em; font-weight: 600; }}
.tag-green {{ background: #1a4731; color: #3fb950; }}
.tag-yellow {{ background: #3d2e00; color: #d29922; }}
.tag-blue {{ background: #0c2d6b; color: #58a6ff; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 14px; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }}
.metric-box {{ background: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }}
.metric-label {{ color: #8b949e; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; }}
.metric-value {{ color: #e6edf3; font-size: 1.3em; font-weight: 600; margin-top: 2px; }}
.metric-unit {{ color: #8b949e; font-size: 0.75em; font-weight: 400; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.8em; }}
th {{ background: #21262d; color: #8b949e; text-align: left; padding: 8px 10px; border-bottom: 1px solid #30363d; font-weight: 600; text-transform: uppercase; font-size: 0.7em; letter-spacing: 0.5px; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #21262d; }}
.chart-container {{ position: relative; height: 280px; margin: 12px 0; }}
.note {{ background: #0c2d6b; border-left: 3px solid #58a6ff; padding: 8px 12px; border-radius: 0 6px 6px 0; margin: 8px 0; font-size: 0.8em; }}
.ok {{ background: #1a4731; border-left: 3px solid #3fb950; padding: 8px 12px; border-radius: 0 6px 6px 0; margin: 8px 0; font-size: 0.8em; }}
.warn {{ background: #3d2e00; border-left: 3px solid #d29922; padding: 8px 12px; border-radius: 0 6px 6px 0; margin: 8px 0; font-size: 0.8em; }}
.placeholder {{ text-align: center; padding: 60px 20px; color: #8b949e; }}
.placeholder h2 {{ border: none; color: #484f58; }}
.compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.param-core {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; }}
.param-full {{ margin-top: 12px; border-top: 1px solid #30363d; padding-top: 12px; }}
.param-toggle {{ display: inline-flex; align-items: center; gap: 4px; cursor: pointer; color: #58a6ff; font-size: 0.8em; background: none; border: 1px solid #30363d; border-radius: 6px; padding: 4px 12px; margin-top: 10px; transition: all 0.15s; }}
.param-toggle:hover {{ background: #21262d; }}
.param-toggle .arrow {{ transition: transform 0.2s; display: inline-block; }}
.param-toggle.open .arrow {{ transform: rotate(90deg); }}
.param-full {{ display: none; }}
.param-full.open {{ display: block; }}
.param-table {{ width: 100%; font-size: 0.78em; }}
.param-table td:first-child {{ color: #8b949e; width: 45%; }}
.param-table td:last-child {{ color: #e6edf3; font-family: monospace; }}
.param-section-label {{ color: #d2a8ff; font-size: 0.8em; margin: 10px 0 6px 0; font-weight: 600; }}
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h1>📊 评估面板</h1>
    <div id="run-list"></div>
  </div>
  <div class="main" id="report-area">
    <div class="placeholder">
      <h2>选择左侧的训练版本查看评估报告</h2>
    </div>
  </div>
</div>

<script>
const RUNS = {runs_json};
const RUN_DATA = {run_data_json};
const SELECTED = {json.dumps(selected_run)};

// --- Render run list ---
function renderRunList() {{
  const container = document.getElementById('run-list');
  if (!RUNS.length) {{
    container.innerHTML = '<p style="color:#8b949e;font-size:0.85em;">暂无训练记录</p>';
    return;
  }}
  container.innerHTML = RUNS.map(r => {{
    const active = r.run_id === SELECTED ? 'active' : '';
    const statusTag = r.status === 'completed'
      ? '<span class="tag tag-green">完成</span>'
      : r.status === 'partial'
      ? '<span class="tag tag-yellow">部分</span>'
      : '<span class="tag tag-blue">未知</span>';
    const evalTag = r.has_offline_eval ? ' <span class="tag tag-blue">离线评估</span>' : '';
    return `<div class="run-item ${{active}}" onclick="window.location.href='/view/${{r.run_id}}'">
      <div class="run-id">${{r.run_id}}</div>
      <div class="run-meta">${{statusTag}} ${{evalTag}} ${{r.completed_rounds || 0}}/${{r.num_rounds || 0}} 轮</div>
    </div>`;
  }}).join('');
}}

// --- Render report ---
function renderReport() {{
  if (!RUN_DATA) return;
  const d = RUN_DATA;
  const rl = d.round_log || [];
  const md = d.metrics_detailed;
  const md1 = d.metrics_detailed_client1;
  const oe = d.offline_eval;
  const meta = d.run_meta || {{}};
  const area = document.getElementById('report-area');

  let html = `<h1>📋 ${{d.run_id}}</h1>`;
  html += `<p style="color:#8b949e;font-size:0.85em;margin-bottom:16px;">`;
  if (meta.num_rounds) html += `${{meta.num_rounds}} 轮 | ${{meta.num_clients}} 客户端 | ${{meta.transfer_dtype || 'fp16'}} | coverage_h=${{meta.coverage_h}}`;
  html += `</p>`;

  // --- Training Parameters ---
  const ec = meta.effective_config || {{}};
  // GPU count: infer from metrics_detailed device or accelerate config
  let gpuPerClient = '—';
  if (md && md.training && md.training.num_steps !== undefined) {{
    // FSDP runs 8 GPUs per node; we can't always infer, use known config
    gpuPerClient = 8;
  }}
  const totalGPUs = (meta.num_clients || 0) * (gpuPerClient === '—' ? 0 : gpuPerClient);

  // Core params (always visible)
  html += `<h2>⚙️ 训练参数</h2><div class="card">`;
  html += `<div class="param-core">`;
  html += box('模型', 'Qwen2.5-0.5B', '');
  html += box('GPU 总数', totalGPUs > 0 ? totalGPUs : '—', gpuPerClient !== '—' ? `${{meta.num_clients}}×${{gpuPerClient}}` : '');
  html += box('客户端数', meta.num_clients || '—', '');
  html += box('联邦轮数', meta.num_rounds || '—', '');
  html += box('学习率', (ec.lr || 1e-5).toExponential(1), '');
  html += box('Batch Size', ec.batch_size || 8, '');
  html += box('Local Steps', ec.local_steps || 10, '');
  html += box('梯度累积', ec.grad_accum || 2, '');
  html += box('序列长度', ec.seq_len || 512, '');
  html += `</div>`;

  html += `<div class="param-section-label">分片 / 联邦参数</div>`;
  html += `<div class="param-core">`;
  html += box('分片模式', meta.compressor || 'public_random', '');
  html += box('Coverage H', meta.coverage_h || '—', '');
  html += box('每轮上传比例', meta.coverage_h ? (100/meta.coverage_h).toFixed(1) : '—', '%');
  html += box('Block Size', meta.block_size || '—', '元素');
  html += box('通信精度', meta.transfer_dtype || 'fp16', '');
  html += box('Memory Decay', meta.memory_decay || '—', '');
  html += box('Slots/Round', meta.slots_per_round || 1, '');
  html += box('Rho', meta.rho !== undefined ? meta.rho : '—', '');
  html += `</div>`;

  // Toggle for full params
  html += `<button class="param-toggle" onclick="toggleParams()"><span class="arrow">▶</span> 展开完整参数</button>`;

  // Full params (hidden by default)
  html += `<div class="param-full" id="paramFull">`;
  html += `<div class="compare-grid">`;
  // Left: federated / sharding params
  html += `<div>`;
  html += `<div class="param-section-label">联邦 & 分片</div>`;
  html += `<table class="param-table">`;
  const fedParams = [
    ['num_clients', meta.num_clients],
    ['num_rounds', meta.num_rounds],
    ['coverage_h', meta.coverage_h],
    ['slots_per_round', meta.slots_per_round],
    ['seed', meta.seed],
    ['compressor', meta.compressor],
    ['block_size', meta.block_size],
    ['transfer_dtype', meta.transfer_dtype],
    ['memory_decay', meta.memory_decay],
    ['rho', meta.rho],
    ['write_full_global_every_n_rounds', meta.write_full_global_every_n_rounds],
    ['min_clients_to_aggregate', ec.min_clients_to_aggregate],
    ['client_upload_timeout_s', ec.client_upload_timeout_s],
    ['round_deadline_s', ec.round_deadline_s],
    ['resume_from_round', ec.resume_from_round || meta.resume_from_round],
  ];
  fedParams.forEach(([k, v]) => html += `<tr><td>${{k}}</td><td>${{v ?? '—'}}</td></tr>`);
  html += `</table>`;
  html += `</div>`;
  // Right: training / security / eval params
  html += `<div>`;
  html += `<div class="param-section-label">训练 & 评估</div>`;
  html += `<table class="param-table">`;
  const trainParams = [
    ['local_steps', ec.local_steps || 10],
    ['batch_size', ec.batch_size || 8],
    ['grad_accum', ec.grad_accum || 2],
    ['lr', (ec.lr || 1e-5).toExponential(2)],
    ['seq_len', ec.seq_len || 512],
    ['eval_every_n_rounds', ec.eval_every_n_rounds || '—'],
    ['eval_path', ec.eval_path || 'data/medical_flashcards_eval.json'],
    ['skip_round0_download', ec.skip_round0_download],
    ['online_eval', ec.online_eval],
    ['sec_upload_privacy', ec.sec_upload_privacy],
    ['auth_token', ec.auth_token !== undefined ? (ec.auth_token ? '已设置' : '无') : '—'],
    ['tls', ec.tls],
    ['minio_retention_recent_uploads', ec.minio_retention_recent_uploads],
    ['gpu_per_client', gpuPerClient],
    ['total_gpus', totalGPUs > 0 ? totalGPUs : '—'],
  ];
  trainParams.forEach(([k, v]) => html += `<tr><td>${{k}}</td><td>${{v ?? '—'}}</td></tr>`);
  html += `</table>`;
  html += `</div>`;
  html += `</div>`;

  // Round-level sharding detail
  if (rl.length) {{
    const r0 = rl[0];
    html += `<div class="param-section-label">首轮分片详情</div>`;
    html += `<table class="param-table">`;
    html += `<tr><td>选中 block 数</td><td>${{r0.n_selected_blocks}}</td></tr>`;
    html += `<tr><td>选中元素</td><td>${{r0.selected_elems_M}}M (${{r0.pct_of_total}}%)</td></tr>`;
    html += `<tr><td>Epoch / Slot</td><td>${{r0.epoch}} / ${{r0.slot}}</td></tr>`;
    html += `<tr><td>写入全量模型</td><td>${{r0.wrote_full_global ? '是' : '否'}}</td></tr>`;
    html += `<tr><td>参与客户端</td><td>${{(r0.participated_clients || []).join(', ')}}</td></tr>`;
    html += `</table>`;
  }}
  html += `</div>`; // end param-full
  html += `</div>`; // end card

  // --- Loss trend chart ---
  if (rl.length) {{
    html += `<h2>📉 Loss 趋势</h2><div class="card"><div class="chart-container"><canvas id="lossChart"></canvas></div></div>`;
  }}

  // --- Training Performance ---
  if (md && md.training) {{
    const t = md.training;
    html += `<h2>⚡ 训练性能</h2><div class="card">`;
    html += `<div class="metric-grid">`;
    html += box('平均步时间', t.avg_step_time_ms, 'ms');
    html += box('前向时间', t.avg_forward_ms, 'ms');
    html += box('反向时间', t.avg_backward_ms, 'ms');
    html += box('优化器时间', t.avg_optimizer_ms, 'ms');
    html += box('通信时间', t.avg_comm_ms, 'ms');
    html += box('总训练时间', t.total_train_time_s, 's');
    html += `</div>`;

    // Per-step loss chart
    if (t.steps && t.steps.length) {{
      html += `<h3>Per-Step Loss (最终轮)</h3><div class="chart-container" style="height:200px;"><canvas id="stepLossChart"></canvas></div>`;
    }}

    // Compare ICC1 vs ICC2
    if (md1 && md1.training) {{
      const t2 = md1.training;
      html += `<div class="compare-grid" style="margin-top:12px;">`;
      html += `<div class="metric-box"><h3 style="color:#58a6ff;">ICC1</h3>`;
      html += `<table><tr><td>步时间</td><td>${{t.avg_step_time_ms}}ms</td></tr><tr><td>前向</td><td>${{t.avg_forward_ms}}ms</td></tr><tr><td>反向</td><td>${{t.avg_backward_ms}}ms</td></tr></table>`;
      html += `</div>`;
      html += `<div class="metric-box"><h3 style="color:#3fb950;">ICC2</h3>`;
      html += `<table><tr><td>步时间</td><td>${{t2.avg_step_time_ms}}ms</td></tr><tr><td>前向</td><td>${{t2.avg_forward_ms}}ms</td></tr><tr><td>反向</td><td>${{t2.avg_backward_ms}}ms</td></tr></table>`;
      html += `</div></div>`;
    }}
    html += `</div>`;
  }}

  // --- Resource Usage ---
  if (md && md.resources) {{
    const r = md.resources;
    html += `<h2>🖥️ 资源使用</h2><div class="card"><div class="metric-grid">`;
    html += box('GPU 峰值显存', r.gpu_memory_peak_mb ? (r.gpu_memory_peak_mb/1024).toFixed(2) : '—', 'GB');
    html += box('GPU 利用率', r.gpu_utilization_avg_pct !== null ? r.gpu_utilization_avg_pct : '—', '%');
    html += box('CPU 利用率', r.cpu_utilization_avg_pct, '%');
    html += box('CPU 峰值内存', r.cpu_memory_peak_mb ? (r.cpu_memory_peak_mb/1024).toFixed(2) : '—', 'GB');
    html += `</div>`;
    if (r.gpu_utilization_avg_pct === null) {{
      html += `<div class="warn"><strong>⚠ GPU 利用率未采集</strong>：旧版本训练未记录此指标，新版本已修复</div>`;
    }}
    html += `</div>`;
  }}

  // --- Federated Timing ---
  if (rl.length) {{
    html += `<h2>🌐 联邦时序</h2><div class="card">`;
    html += `<div class="chart-container"><canvas id="timingChart"></canvas></div>`;
    html += `<table><thead><tr><th>轮次</th><th>周期(s)</th><th>训练(s)</th><th>WAN上传(s)</th><th>聚合(s)</th><th>传输量(MB)</th></tr></thead><tbody>`;
    rl.forEach(r => {{
      const clients = (r.timing_s || {{}}).clients || {{}};
      const c0 = clients['0'] || {{}};
      const uploadAvg = c0.wan_upload_s !== undefined ? c0.wan_upload_s : '—';
      const transfer = (r.timing_s || {{}}).transfer || {{}};
      const deltaMB = transfer.global_delta_MiB ? transfer.global_delta_MiB.toFixed(1) : '—';
      html += `<tr><td>${{r.round}}</td><td>${{(r.timing_s||{{}}).round_wall_s}}</td><td>${{c0.train_local_s || '—'}}</td><td>${{uploadAvg}}</td><td>${{(r.timing_s||{{}}).agg_total_s}}</td><td>${{deltaMB}}</td></tr>`;
    }});
    html += `</tbody></table></div>`;
  }}

  // --- Model Accuracy (online eval) ---
  if (rl.length) {{
    const evalRounds = rl.filter(r => r.eval_loss !== null && r.eval_loss !== undefined);
    if (evalRounds.length) {{
      html += `<h2>🎯 在线评估指标</h2><div class="card"><div class="metric-grid">`;
      const firstEval = evalRounds[0];
      const lastEval = evalRounds[evalRounds.length - 1];
      html += box('初始 Eval Loss', firstEval.eval_loss);
      html += box('最终 Eval Loss', lastEval.eval_loss);
      const improvement = firstEval.eval_loss && lastEval.eval_loss
        ? (((firstEval.eval_loss - lastEval.eval_loss) / firstEval.eval_loss) * 100).toFixed(1)
        : '—';
      html += box('Eval Loss 下降', improvement, '%');
      html += box('初始 Train Loss', firstEval.avg_train_loss);
      html += box('最终 Train Loss', lastEval.avg_train_loss);
      html += `</div></div>`;
    }}
  }}

  // --- Offline Generative Evaluation ---
  if (oe) {{
    const v = oe.validation || {{}};
    const g = oe.generation || {{}};
    html += `<h2>🏆 离线生成式评估</h2><div class="card">`;
    html += `<div class="note">评估轮次: ${{oe.round}} | 样本数: ${{g.num_eval_samples || 0}} | 设备: ${{oe.device || '—'}}</div>`;
    html += `<div class="metric-grid">`;
    html += box('Validation Loss', v.val_loss);
    html += box('Perplexity', v.perplexity);
    html += box('ROUGE-L F1', g.rouge_l ? g.rouge_l.f1 : '—');
    html += box('BERTScore F1', g.bertscore ? g.bertscore.f1 : '—');
    html += box('Token Overlap Acc', g.token_overlap_accuracy);
    html += box('Macro F1', g.macro_f1);
    html += box('Exact Match', g.exact_match);
    html += box('Eval GPU 显存', oe.resource_usage ? (oe.resource_usage.gpu_memory_peak_mb/1024).toFixed(2) : '—', 'GB');
    html += `</div>`;

    // Sample predictions
    html += `<h3>生成样本预览</h3>`;
    html += `<div id="predictions" style="max-height:400px;overflow-y:auto;border:1px solid #30363d;border-radius:6px;padding:10px;background:#0d1117;"></div>`;
    html += `</div>`;
  }} else {{
    html += `<h2>🏆 离线生成式评估</h2><div class="card"><div class="warn">离线生成式评估未运行。需在 GPU 节点执行 <code>experiments/run_s3r12v3_offline_eval.py</code></div></div>`;
  }}

  area.innerHTML = html;

  // --- Draw charts ---
  // Loss trend chart
  if (rl.length) {{
    const ctx = document.getElementById('lossChart');
    if (ctx) {{
      const labels = rl.map(r => `Round ${{r.round}}`);
      const trainLosses = rl.map(r => r.avg_train_loss);
      const evalLosses = rl.map(r => r.eval_loss);
      const datasets = [{{
        label: 'Train Loss', data: trainLosses, borderColor: '#f85149', backgroundColor: '#f8514920',
        fill: true, tension: 0.3, pointRadius: 4,
      }}];
      const hasEval = evalLosses.some(v => v !== null && v !== undefined);
      if (hasEval) {{
        datasets.push({{
          label: 'Eval Loss', data: evalLosses, borderColor: '#3fb950', backgroundColor: '#3fb95020',
          fill: true, tension: 0.3, pointRadius: 4, spanGaps: true,
        }});
      }}
      new Chart(ctx, {{
        type: 'line',
        data: {{ labels, datasets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9' }} }},
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }}, title: {{ display: true, text: 'Loss', color: '#8b949e' }} }},
          }},
        }},
      }});
    }}
  }}

  // Per-step loss chart
  if (md && md.training && md.training.steps) {{
    const ctx = document.getElementById('stepLossChart');
    if (ctx) {{
      const steps = md.training.steps;
      new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: steps.map(s => `Step ${{s.step}}`),
          datasets: [{{
            label: 'Training Loss', data: steps.map(s => s.loss),
            borderColor: '#58a6ff', backgroundColor: '#58a6ff20',
            fill: true, tension: 0.3, pointRadius: 3,
          }}],
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
          }},
        }},
      }});
    }}
  }}

  // Timing chart
  if (rl.length) {{
    const ctx = document.getElementById('timingChart');
    if (ctx) {{
      const labels = rl.map(r => `Round ${{r.round}}`);
      const datasets = [];
      const phases = [
        ['train_local_s', '训练', '#58a6ff'],
        ['encode_delta_s', '编码', '#d29922'],
        ['upload_minio_s', 'WAN上传', '#f85149'],
        ['wait_aggregate_s', '等待聚合', '#8b949e'],
      ];
      phases.forEach(([key, label, color]) => {{
        const data = rl.map(r => {{
          const c = (r.timing_s || {{}}).clients || {{}};
          const c0 = c['0'] || {{}};
          return c0[key] || 0;
        }});
        datasets.push({{ label, data, backgroundColor: color + 'cc', borderColor: color, borderWidth: 1 }});
      }});
      new Chart(ctx, {{
        type: 'bar',
        data: {{ labels, datasets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
          scales: {{
            x: {{ stacked: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
            y: {{ stacked: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }}, title: {{ display: true, text: '秒', color: '#8b949e' }} }},
          }},
        }},
      }});
    }}
  }}

  // Load predictions if offline eval exists
  if (oe) {{
    const predContainer = document.getElementById('predictions');
    if (predContainer) {{
      fetch(`/api/runs/${{d.run_id}}`).then(r => r.json()).then(data => {{
        // predictions are in offline_eval but not included in the summary
        // We'll load them from the predictions endpoint
      }}).catch(() => {{}});
      predContainer.innerHTML = '<p style="color:#8b949e;">预测样本请在服务器上查看 predictions.jsonl</p>';
    }}
  }}
}}

function box(label, value, unit) {{
  return `<div class="metric-box"><div class="metric-label">${{label}}</div><div class="metric-value">${{value ?? '—'}} <span class="metric-unit">${{unit || ''}}</span></div></div>`;
}}

function toggleParams() {{
  const full = document.getElementById('paramFull');
  const btn = full.previousElementSibling;
  full.classList.toggle('open');
  btn.classList.toggle('open');
  if (full.classList.contains('open')) {{
    btn.innerHTML = '<span class="arrow">▶</span> 收起完整参数';
  }} else {{
    btn.innerHTML = '<span class="arrow">▶</span> 展开完整参数';
  }}
}}

renderRunList();
renderReport();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="FedScale Evaluation Dashboard")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--results-dir", default=None,
                        help="Override results directory (default: <repo>/results)")
    args = parser.parse_args()

    if args.results_dir:
        os.environ["EVAL_DASHBOARD_RESULTS_DIR"] = args.results_dir

    print(f"Dashboard results dir: {_get_results_dir()}")
    print(f"Starting dashboard on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
