const $ = (selector) => document.querySelector(selector);

const THEME_STORAGE_KEY = "federated-lab-theme";

function applyTheme(theme) {
  const resolved = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = resolved;
  const label = $("#theme-label");
  if (label) label.textContent = resolved === "dark" ? "浅色模式" : "深色模式";
  const button = $("#theme-toggle");
  if (button) {
    button.setAttribute(
      "aria-label",
      resolved === "dark" ? "切换到浅色模式" : "切换到深色模式",
    );
  }
}

function initTheme() {
  const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
  applyTheme(saved || "dark");
  $("#theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    window.localStorage.setItem(THEME_STORAGE_KEY, next);
    applyTheme(next);
  });
}

const state = {
  experiments: [],
  comparison: null,
  source: new URLSearchParams(window.location.search).get("source"),
  sources: [],
};

const selectors = {
  left: $("#left-select"),
  right: $("#right-select"),
  log: $("#log-experiment-select"),
};

const parameterFormatters = {
  finetuning_type: (value) => ({ full: "全参数微调", lora: "LoRA" }[value] || value),
  quantization: (value) => Number(value) === 0 ? "FP16（未量化）" : `${value}-bit`,
  gradient_checkpointing: formatBoolean,
  evaluate_after_fit: formatBoolean,
  full_local_initialization: formatBoolean,
  ddp_cpu_offload: formatBoolean,
  learning_rate: (value) => value == null ? "—" : Number(value).toExponential().replace("e-0", "e-"),
  compression: (value) => value === "topk-int8" ? "Top-K INT8" : value,
  topk_ratio: (value) => value == null ? "—" : String(value),
  strategy: (value) => value || "—",
};

function formatBoolean(value) {
  if (value === true) return "开启";
  if (value === false) return "关闭";
  return "—";
}

function formatParameter(key, value) {
  if (value == null || value === "") return "—";
  return parameterFormatters[key] ? parameterFormatters[key](value) : String(value);
}

function formatMetric(value, unit, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  if (unit === "seconds") return `${Number(value).toFixed(digits)} s`;
  if (unit === "milliseconds") return `${Number(value).toFixed(1)} ms`;
  if (unit === "memory_mb") return Number(value) >= 1024
    ? `${(Number(value) / 1024).toFixed(2)} GiB`
    : `${Number(value).toFixed(1)} MiB`;
  if (unit === "bytes") {
    const bytes = Number(value);
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
    return `${bytes.toFixed(0)} B`;
  }
  if (unit === "percent") return `${Number(value).toFixed(2)}%`;
  if (unit === "ratio") return `${(Number(value) * 100).toFixed(2)}%`;
  if (unit === "tokens_s") return `${Number(value).toFixed(2)} tok/s`;
  return Number(value).toFixed(4);
}

function formatDate(value) {
  if (!value) return "时间未记录";
  const normalized = /Z$|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("zh-CN", {
        year: "numeric", month: "2-digit", day: "2-digit",
        hour: "2-digit", minute: "2-digit", second: "2-digit",
        hour12: false,
      }).format(date);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function statusLabel(status) {
  return ({
    completed: "已完成",
    failed: "失败",
    stopped: "已停止",
    stale: "状态陈旧",
    running: "运行中",
    starting: "启动中",
    pending: "等待中",
    unknown: "未知",
  })[status] || status || "未知";
}

function supervisorPhaseLabel(phase) {
  return ({
    not_started: "未启动",
    starting: "启动中",
    submitting: "提交 Run",
    monitoring: "监控中",
    waiting_for_idle: "等待空闲",
    paused: "已暂停",
    stopping: "停止中",
    stopped: "已停止",
    completed: "已完成",
    failed: "失败",
  })[phase] || phase || "未知";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { Accept: "application/json", ...(options.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `请求失败：HTTP ${response.status}`);
  return body;
}

function chooseDefaults(experiments) {
  const params = new URLSearchParams(window.location.search);
  const requestedLeft = params.get("left");
  const requestedRight = params.get("right");
  const ids = new Set(experiments.map((item) => item.id));
  if (ids.has(requestedLeft) && ids.has(requestedRight) && requestedLeft !== requestedRight) {
    return [requestedLeft, requestedRight];
  }

  const preferredLeft = "fsdp-full-topk-3b-s512-s10-gc-pair-r1";
  const preferredRight = "ddp-full-cpuoffload-topk-3b-s512-s10-gc-pair-r2";
  if (ids.has(preferredLeft) && ids.has(preferredRight)) return [preferredLeft, preferredRight];

  const completed = experiments.filter((item) => item.has_summary && item.status === "completed");
  const fsdp = completed.find((item) => item.strategy === "FSDP");
  const ddp = completed.find((item) => item.strategy === "DDP");
  if (fsdp && ddp) return [fsdp.id, ddp.id];
  return [experiments[0]?.id, experiments[1]?.id || experiments[0]?.id];
}

function populateSelectors(experiments, selected) {
  const options = experiments.map((item) => {
    const strategy = item.strategy && item.strategy !== "UNKNOWN" ? ` · ${item.strategy}` : "";
    const status = item.has_summary && item.status === "completed"
      ? "✓ 可用于正式对照"
      : "日志观测 · " + statusLabel(item.status);
    return `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)}${strategy} · ${escapeHtml(status)}</option>`;
  }).join("");
  selectors.left.innerHTML = options;
  selectors.right.innerHTML = options;
  selectors.left.value = selected[0];
  selectors.right.value = selected[1];
}

function populateLogSelector(experiments) {
  const requested = new URLSearchParams(window.location.search).get("log");
  const previous = selectors.log.value;
  selectors.log.innerHTML = experiments.map((item) =>
    `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)} · ${escapeHtml(statusLabel(item.status))}</option>`
  ).join("");
  const ids = new Set(experiments.map((item) => item.id));
  selectors.log.value = ids.has(requested) ? requested : ids.has(previous) ? previous : experiments[0]?.id;
}

function renderLogs(payload) {
  const events = Array.isArray(payload.events) ? payload.events : [];
  const progress = payload.progress || {};
  const phase = progress.phase ? ` · 阶段 ${progress.phase}` : "";
  const heartbeat = progress.heartbeat_at ? ` · 心跳 ${formatDate(progress.heartbeat_at)}` : "";
  $("#log-meta").textContent = `${statusLabel(payload.status)}${phase}${heartbeat} · ${events.length} 条记录`;
  $("#log-stream").innerHTML = events.length ? events.map((event) => {
    const details = event.details || {};
    const chips = [
        details.run_id ? `RUN ${details.run_id}` : "",
        details.round != null ? `ROUND ${details.round}` : "",
        details.duration_seconds != null ? `${Number(details.duration_seconds).toFixed(2)} s` : "",
        details.phase ? `阶段 ${details.phase}` : "",
    ].filter(Boolean).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
    return `<article class="log-event level-${escapeHtml(event.level || "info")}">
      <time>${escapeHtml(formatDate(event.timestamp))}</time>
      <div><h3>${escapeHtml(event.title || "运行记录")}</h3><p>${escapeHtml(event.message || "—")}</p>${chips ? `<div class="log-chips">${chips}</div>` : ""}</div>
    </article>`;
  }).join("") : '<p class="log-empty">此实验暂未写入可显示的联邦运行记录。</p>';
}

function renderSupervisor(snapshot) {
  const status = snapshot?.status || {};
  const control = snapshot?.control || {};
  const hasDetailedStatus = Object.keys(status).length > 0;
  const phase = hasDetailedStatus ? (status.phase || "unknown") : "not_started";
  $("#supervisor-phase").textContent = supervisorPhaseLabel(phase);
  $("#supervisor-message").textContent = status.message
    || (phase === "not_started"
      ? "监督器 Job 尚未启动，请点击“启动监督器”。"
      : "监督器正在初始化，详细状态尚未写入。");
  $("#supervisor-experiment").textContent = status.experiment_id || control.matrix_id || "—";
  $("#supervisor-run").textContent = status.run_id || "—";
  $("#supervisor-job").textContent = status.job_name || snapshot?.job?.job_name || "—";
  $("#supervisor-heartbeat").textContent = status.heartbeat_at ? formatDate(status.heartbeat_at) : "—";
  const dot = $("#supervisor-dot");
  dot.dataset.state = phase;
  $("#control-strategy").value = control.strategy || status.strategy || "fedscale";
  $("#control-rounds").value = control.rounds ?? status.rounds ?? 10;
  $("#control-model").value = control.model || status.model || "Qwen/Qwen2.5-7B";
  $("#control-dataset").value = control.dataset || status.dataset || "HuggingFaceH4/ultrachat_200k";
  $("#control-finetuning-type").value = control.finetuning_type || status.finetuning_type || "lora";
  $("#control-poll").value = control.poll_seconds ?? status.poll_seconds ?? 120;
  $("#control-stall").value = control.stall_seconds ?? status.stall_seconds ?? 7200;
  $("#control-restarts").value = control.max_restarts ?? status.max_restarts ?? 3;
  $("#control-matrix").value = control.matrix_id || "";
}

async function loadSupervisor() {
  if (state.source !== "cloud") {
    $("#supervisor-message").textContent = "切换到 TKE 云端数据源后才能控制中心监督器。";
    return;
  }
  try {
    const snapshot = await fetchJson("/api/supervisor?source=cloud");
    renderSupervisor(snapshot);
  } catch (error) {
    $("#supervisor-message").textContent = `监督器状态读取失败：${error.message}`;
  }
}

async function sendSupervisorAction(action) {
  if (action === "stop" && !window.confirm("确认停止当前 Flower Run？这会中止正在进行的实验。")) return;
  if (action === "start" && !window.confirm("确认启动新的监督器 Job？如果已有监督器正在运行，系统会拒绝重复启动。")) return;
  const payload = {
    action,
    poll_seconds: Number($("#control-poll").value),
    stall_seconds: Number($("#control-stall").value),
    max_restarts: Number($("#control-restarts").value),
    matrix_id: $("#control-matrix").value.trim(),
    strategy: $("#control-strategy").value,
    rounds: Number($("#control-rounds").value),
    model: $("#control-model").value.trim(),
    dataset: $("#control-dataset").value.trim(),
    finetuning_type: $("#control-finetuning-type").value,
  };
  const feedback = $("#control-feedback");
  feedback.textContent = action === "configure"
    ? "正在保存实验配置…"
    : action === "start" ? "正在创建监督器 Job…" : "正在写入监督器控制文件…";
  try {
    const response = await fetchJson("/api/supervisor?source=cloud", {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderSupervisor(response.supervisor || {});
    feedback.textContent = action === "configure"
      ? "实验配置已保存；监督器将在下一次提交时使用。"
      : action === "start"
        ? `监督器已启动：${response.result?.start?.job_name || "Job 创建请求已提交"}。`
      : `已发送${({ resume: "恢复调度", pause: "暂停调度", stop: "停止当前 Run" })[action] || action}，监督器将在下一次轮询应用。`;
    await loadLogs();
  } catch (error) {
    feedback.textContent = `控制失败：${error.message}`;
  }
}

async function loadLogs() {
  const experimentId = selectors.log.value;
  if (!experimentId) return;
  const button = $("#log-refresh-button");
  button.disabled = true;
  try {
    const query = new URLSearchParams({ source: state.source || "center" });
    const payload = await fetchJson(`/api/experiments/${encodeURIComponent(experimentId)}/logs?${query}`);
    renderLogs(payload);
  } catch (error) {
    $("#log-meta").textContent = `日志读取失败：${error.message}`;
    $("#log-stream").innerHTML = '<p class="log-empty">无法读取该实验的运行日志。</p>';
  } finally {
    button.disabled = false;
  }
}

function updateSelectionMeta(side, experiment) {
  $(`#${side}-strategy`).textContent = experiment.strategy || "—";
  const cloudMeta = state.source === "cloud"
    ? ` · ${experiment.cloud_client_count ?? experiment.cloud?.clients?.length ?? 0} 个 TKE 客户端结果`
    : "";
  const sourceMeta = experiment.status_source === "flower-run"
    ? "Flower 实时状态"
    : experiment.status === "stale" ? "旧状态文件已超时" : "结果文件状态";
  const comparisonReady = experiment.has_summary && experiment.status === "completed";
  const scope = comparisonReady ? "可用于正式对照" : "仅用于日志观测";
  const meta = document.querySelector("#" + side + "-meta");
  meta.textContent =
    scope + " · " + statusLabel(experiment.status) + " · " + sourceMeta
    + " · RUN " + (experiment.run_id ?? "—") + cloudMeta;
}

function renderVerdict(comparison) {
  const { primary, left, right } = comparison;
  const partial = comparison.comparison_scope === "partial";
  const leftValue = primary.left;
  const rightValue = primary.right;
  const hasResult = leftValue != null && rightValue != null;
  let winner = null;
  let loser = null;
  if (primary.winner === "left") [winner, loser] = [left, right];
  if (primary.winner === "right") [winner, loser] = [right, left];

  if (partial) {
    $("#verdict-title").textContent = "阶段性结果对照（非最终）";
    $("#verdict-description").textContent = hasResult
      ? "以下只比较已完成联邦轮和已持久化中间结果，不代表完整实验的最终策略结论。"
      : "以下展示已持久化的阶段性结果；当前尚无两侧都具备的完整关键路径指标。";
    $("#winner-strategy").textContent = "INTERMEDIATE";
    $("#winner-percent").textContent = "—";
  } else if (!hasResult) {
    $("#verdict-title").textContent = "所选实验尚无完整实测结果";
    $("#verdict-description").textContent = "请选择两个已经完成并包含 experiment_summary.json 的实验。";
    $("#winner-strategy").textContent = "NO RESULT";
    $("#winner-percent").textContent = "—";
  } else if (winner) {
    const percent = primary.percent ?? 0;
    $("#verdict-title").textContent = `${winner.strategy} 赢得${comparison.verdict_scope || "客户端关键路径"}`;
    const sourceNote = comparison.source === "cloud"
      ? "数据直接来自 TKE A/B 客户端 metrics 文件。"
      : "结论基于客户端关键路径整轮耗时，不受 Run 提交排队影响。";
    $("#verdict-description").textContent = `${winner.id} 相比 ${loser.id} 少用 ${Math.abs(primary.delta).toFixed(2)} 秒；${sourceNote}`;
    $("#winner-strategy").textContent = winner.strategy;
    $("#winner-percent").textContent = `${percent.toFixed(2)}%`;
  } else {
    $("#verdict-title").textContent = "两个实验关键路径耗时相同";
    $("#verdict-description").textContent = "当前精度下没有可区分的整轮耗时差异。";
    $("#winner-strategy").textContent = "TIE";
    $("#winner-percent").textContent = "0.00%";
  }

  $("#critical-delta").textContent = primary.delta == null ? "—" : Math.abs(primary.delta).toFixed(2);
  const training = comparison.metrics.find((item) => item.key === "critical_training_seconds");
  $("#training-delta").textContent = training?.delta == null ? "—" : Math.abs(training.delta).toFixed(2);
  const controlled = comparison.parameters.filter((row) => !row.expected_difference);
  const matched = controlled.filter((row) => row.same).length;
  $("#parameter-match").textContent = `${matched}/${controlled.length}`;
}

function renderParameters(comparison) {
  $("#parameter-left-heading").textContent = comparison.left.strategy;
  $("#parameter-right-heading").textContent = comparison.right.strategy;
  $("#parameter-body").innerHTML = comparison.parameters.map((row) => {
    let statusClass = "status-different";
    let statusText = "不一致";
    if (row.same) {
      statusClass = "status-same";
      statusText = "一致";
    } else if (row.expected_difference) {
      statusClass = "status-expected";
      statusText = "策略差异";
    }
    return `<tr>
      <td>${escapeHtml(row.label)}</td>
      <td>${escapeHtml(formatParameter(row.key, row.left))}</td>
      <td>${escapeHtml(formatParameter(row.key, row.right))}</td>
      <td><span class="status-chip ${statusClass}">${statusText}</span></td>
    </tr>`;
  }).join("");
}

function renderMetrics(comparison) {
  $("#metric-grid").innerHTML = renderMetricCards(comparison.metrics, comparison, false);
  $("#quality-grid").innerHTML = renderMetricCards(
    comparison.quality_metrics || [],
    comparison,
    true,
  );
  const partial = comparison.comparison_scope === "partial";
  const qualityAvailable = [comparison.left, comparison.right].some(
    (experiment) => experiment.quality_status?.available,
  );
  $("#quality-section").hidden = partial && !qualityAvailable;
  $("#quality-title").textContent = partial ? "阶段性模型质量评测" : "正式模型质量评测";
  const statuses = [comparison.left.quality_status, comparison.right.quality_status];
  const unavailable = statuses.filter((status) => !status?.available);
  if (unavailable.length === 2 && statuses.every((status) => status && !status.requested)) {
    $("#quality-status").textContent = "本次两个实验均未启用评估（evaluate-after-fit=false 且 fraction-evaluate=0），质量指标不适用。";
  } else if (unavailable.length) {
    $("#quality-status").textContent = "所选实验没有可用评估结果；缺失项显示为 —，不会推断模型质量。";
  }
}

const federatedTimingColumns = [
  ["tke_a_training_seconds", "TKE-A FSDP 训练"],
  ["tke_b_training_seconds", "TKE-B FSDP 训练"],
  ["critical_training_seconds", "训练关键路径"],
  ["critical_client_round_seconds", "客户端整轮关键路径"],
  ["average_compression_seconds", "平均 Top-K 压缩"],
  ["federated_cycle_seconds", "联邦周期"],
  ["server_fedavg_aggregation_seconds", "FedAvg 聚合计算"],
  ["server_post_aggregation_seconds", "聚合后处理"],
  ["checkpoint_save_seconds", "Checkpoint 保存"],
  ["checkpoint_interval_seconds", "Checkpoint 间隔"],
];

function renderFederatedTiming(comparison) {
  const section = $("#federated-timing-section");
  const timing = comparison.timing || {};
  const hasRecords = ["left", "right"].some(
    (side) => Array.isArray(timing[side]) && timing[side].length,
  );
  section.hidden = !hasRecords;
  if (!hasRecords) return;

  $("#federated-timing-columns").innerHTML = ["left", "right"].map((side) => {
    const records = Array.isArray(timing[side]) ? timing[side] : [];
    const experiment = comparison[side];
    const rows = records.map((record) => `
      <tr>
        <th scope="row">第 ${escapeHtml(record.round ?? "—")} 轮</th>
        ${federatedTimingColumns.map(([key]) => `<td>${escapeHtml(formatMetric(record[key], "seconds"))}</td>`).join("")}
      </tr>
    `).join("");
    return `<article class="timing-panel">
      <header>
        <div><strong class="${side === "right" ? "right-site" : ""}">${escapeHtml(experiment.strategy)}</strong><h3 title="${escapeHtml(experiment.id)}">${escapeHtml(experiment.id)}</h3></div>
        <small>${records.length} 个联邦轮次</small>
      </header>
      <div class="timing-table-shell">
        <table class="timing-table">
          <thead><tr><th>轮次</th>${federatedTimingColumns.map(([, label]) => `<th>${escapeHtml(label)}</th>`).join("")}</tr></thead>
          <tbody>${rows || '<tr><td colspan="9" class="timing-empty">尚无逐轮时间记录</td></tr>'}</tbody>
        </table>
      </div>
    </article>`;
  }).join("");
}

function renderMetricCards(metrics, comparison, quality) {
  return metrics.map((metric) => {
    const values = [metric.left, metric.right].filter((value) => value != null).map(Number);
    const max = Math.max(...values, 1);
    const leftWidth = metric.left == null ? 0 : Math.max(2, Number(metric.left) / max * 100);
    const rightWidth = metric.right == null ? 0 : Math.max(2, Number(metric.right) / max * 100);
    const winner = metric.winner === "left" ? comparison.left : metric.winner === "right" ? comparison.right : null;
    const isContext = metric.interpretation === "context";
    let deltaText = "暂无可比数据";
    if (metric.delta != null) {
      if (quality && winner && ["lower", "higher"].includes(metric.interpretation)) {
        const direction = metric.interpretation === "lower" ? "低于" : "高出";
        deltaText = `<strong>${escapeHtml(winner.strategy)}</strong> ${direction} ${escapeHtml(formatMetric(Math.abs(metric.delta), metric.unit))}${metric.percent == null ? "" : `（${metric.percent.toFixed(2)}%）`}`;
      } else if (metric.interpretation === "lower" && winner) {
        deltaText = `<strong>${escapeHtml(winner.strategy)}</strong> 少用 ${Math.abs(metric.delta).toFixed(2)} 秒${metric.percent == null ? "" : `（${metric.percent.toFixed(2)}%）`}`;
      } else if (metric.interpretation === "higher" && winner) {
        deltaText = `<strong>${escapeHtml(winner.strategy)}</strong> 高出 ${Math.abs(metric.delta).toFixed(2)}${metric.percent == null ? "" : `（${metric.percent.toFixed(2)}%）`}`;
      } else if (metric.interpretation === "neutral") {
        deltaText = `绝对差值 ${Math.abs(metric.delta).toFixed(4)}`;
      } else {
        deltaText = "此指标可能包含集群排队，不用于正式胜负判断";
      }
    }
    return `<article class="metric-card ${isContext ? "context-card" : ""}">
      <div class="metric-title">
        <h3>${escapeHtml(metric.label)}</h3>
        ${winner ? `<span class="winner-chip">${escapeHtml(winner.strategy)} ${quality ? "BETTER" : "FASTER"}</span>` : ""}
      </div>
      <div class="metric-row left">
        <span class="metric-name" title="${escapeHtml(comparison.left.id)}">${escapeHtml(comparison.left.strategy)}</span>
        <div class="metric-bar-track"><div class="metric-bar" style="width:${leftWidth}%"></div></div>
        <span class="metric-value">${formatMetric(metric.left, metric.unit)}</span>
      </div>
      <div class="metric-row right">
        <span class="metric-name" title="${escapeHtml(comparison.right.id)}">${escapeHtml(comparison.right.strategy)}</span>
        <div class="metric-bar-track"><div class="metric-bar" style="width:${rightWidth}%"></div></div>
        <span class="metric-value">${formatMetric(metric.right, metric.unit)}</span>
      </div>
      <p class="metric-delta">${deltaText}</p>
    </article>`;
  }).join("");
}

function clientStat(label, value) {
  return `<div class="cloud-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderCloudClient(record) {
  const steps = Array.isArray(record.steps) ? record.steps : [];
  const maxStep = Math.max(...steps.map((step) => Number(step.total_ms) || 0), 1);
  const bars = steps.map((step) => {
    const height = Math.max(5, (Number(step.total_ms) || 0) / maxStep * 100);
    return `<i class="step-bar" style="height:${height}%" data-step="${escapeHtml(step.step)}" title="Step ${escapeHtml(step.step)}: ${formatMetric(step.total_ms, "milliseconds")}"></i>`;
  }).join("");
  return `<section class="cloud-client-card">
    <header>
      <div><h3 class="${record.site === "TKE-B" ? "right-site" : ""}">${escapeHtml(record.site || "TKE")}</h3><p title="${escapeHtml(record.job_id || "")}">${escapeHtml(record.job_id || "—")}</p></div>
      <time>${escapeHtml(formatDate(record.timestamp))}</time>
    </header>
    <div class="cloud-stats">
      ${clientStat("训练耗时", formatMetric(record.training_seconds, "seconds"))}
      ${clientStat("平均 step", formatMetric(record.avg_step_ms, "milliseconds"))}
      ${clientStat("平均前向", formatMetric(record.avg_forward_ms, "milliseconds"))}
      ${clientStat("平均反向", formatMetric(record.avg_backward_ms, "milliseconds"))}
      ${clientStat("集群内通信", formatMetric(record.avg_communication_ms, "milliseconds"))}
      ${clientStat("All-Reduce", formatMetric(record.avg_all_reduce_ms, "milliseconds"))}
      ${clientStat("All-Gather", formatMetric(record.avg_all_gather_ms, "milliseconds"))}
      ${clientStat("Reduce-Scatter", formatMetric(record.avg_reduce_scatter_ms, "milliseconds"))}
      ${clientStat("优化器更新", formatMetric(record.avg_optimizer_ms, "milliseconds"))}
      ${clientStat("吞吐量", formatMetric(record.throughput_tokens_s, "tokens_s"))}
      ${clientStat("GPU 峰值", formatMetric(record.gpu_memory_peak_mb, "memory_mb"))}
      ${clientStat("GPU 平均利用率", formatMetric(record.gpu_utilization_avg_pct, "percent"))}
      ${clientStat("CPU 内存峰值", formatMetric(record.cpu_memory_peak_mb, "memory_mb"))}
      ${clientStat("CPU 平均利用率", formatMetric(record.cpu_utilization_avg_pct, "percent"))}
      ${clientStat("训练 Loss", formatMetric(record.train_loss, "number"))}
      ${clientStat("样本数", record.num_examples ?? "—")}
      ${clientStat("Optimizer steps", record.optimizer_steps ?? "—")}
      ${clientStat("model delta 导出", formatMetric(record.model_delta_export_seconds, "seconds"))}
      ${clientStat("Full-state 导出", formatMetric(record.full_state_export_s, "seconds"))}
      ${clientStat("Checkpoint 保存", formatMetric(record.checkpoint_save_seconds, "seconds"))}
    </div>
    ${bars ? `<div class="step-chart" aria-label="逐步耗时柱状图">${bars}</div>` : ""}
  </section>`;
}

function renderAggregationDetails(comparison) {
  const section = $("#aggregation-section");
  section.hidden = comparison.source !== "cloud" || !comparison.aggregation;
  if (section.hidden) return;
  const aggregation = comparison.aggregation;
  const summary = ["left", "right"].map((side) => {
    const item = aggregation[side];
    return `<article class="aggregation-run-meta">
      <strong>${escapeHtml(comparison[side].strategy)}</strong>
      <span title="${escapeHtml(comparison[side].id)}">${escapeHtml(comparison[side].id)}</span>
      <small>联邦轮 ${item.observed_round ?? "—"} · 已汇集 ${item.cluster_count ?? 0} 个集群</small>
    </article>`;
  }).join("");
  $("#aggregation-meta").innerHTML = summary;
  $("#aggregation-performance-grid").innerHTML = renderMetricCards(
    aggregation.performance_metrics || [], comparison, false,
  );
  $("#aggregation-resource-grid").innerHTML = renderMetricCards(
    aggregation.resource_metrics || [], comparison, false,
  );
  $("#aggregation-network-grid").innerHTML = renderMetricCards(
    aggregation.network_metrics || [], comparison, false,
  );
  $("#aggregation-state-export-grid").innerHTML = renderMetricCards(
    aggregation.state_export_metrics || [], comparison, false,
  );
  const serverMeta = ["left", "right"].map((side) => {
    const server = aggregation[side].server || {};
    return `${comparison[side].strategy}：联邦轮 ${formatMetric(server.federated_cycle_seconds, "seconds")} · FedAvg 聚合 ${formatMetric(server.server_fedavg_aggregation_seconds, "seconds")} · 聚合后处理 ${formatMetric(server.server_post_aggregation_seconds, "seconds")} · Checkpoint 保存 ${formatMetric(server.checkpoint_save_seconds, "seconds")}`;
  }).join(" ｜ ");
  $("#aggregation-server-timing").textContent = `中心端时序（已有记录）：${serverMeta}`;
  const missing = [];
  if (!aggregation.left.coverage?.network && !aggregation.right.coverage?.network) missing.push("网络流量");
  if (!aggregation.left.coverage?.nccl && !aggregation.right.coverage?.nccl) missing.push("NCCL 集合通信");
  if (!aggregation.left.coverage?.state_export && !aggregation.right.coverage?.state_export) missing.push("FSDP 状态导出");
  const missingText = missing.length
    ? `当前两个历史实验未采集${missing.join("、")}，无法从已有文件回填精确耗时；请使用新版本训练后刷新。`
    : "网络/NCCL 与状态导出字段已从客户端原始结果汇总。";
  $("#aggregation-deferred-note").textContent = `${missingText} ${aggregation.left.cross_centre_update?.message || ""}`;
}

function renderCloudDetails(comparison) {
  const section = $("#cloud-section");
  section.hidden = comparison.source !== "cloud";
  if (section.hidden) return;
  $("#cloud-left-title").textContent = comparison.left.id;
  $("#cloud-right-title").textContent = comparison.right.id;
  for (const side of ["left", "right"]) {
    const records = comparison[side].cloud?.clients || [];
    $(`#cloud-${side}-clients`).innerHTML = records.length
      ? records.map(renderCloudClient).join("")
      : '<div class="cloud-client-empty">该实验未匹配到成功的 TKE 客户端 metrics 文件</div>';
  }
}

function renderRunDetails(comparison) {
  $("#left-run-id").textContent = comparison.left.run_id ?? "—";
  $("#right-run-id").textContent = comparison.right.run_id ?? "—";
  $("#left-timestamp").textContent = `${formatDate(comparison.left.started_at)} · ${comparison.left.status}`;
  $("#right-timestamp").textContent = `${formatDate(comparison.right.started_at)} · ${comparison.right.status}`;
}

function renderComparisonScope(comparison) {
  const scope = comparison.comparison_scope || "unavailable";
  const formal = scope === "formal";
  const ready = formal || scope === "partial";
  $("#comparison-body").hidden = !ready;
  $("#comparison-not-ready").hidden = ready;
  $("#comparison-scope-eyebrow").textContent = formal
    ? "COMPLETED EXPERIMENT COMPARISON"
    : scope === "partial" ? "PARTIAL RUN COMPARISON" : "COMPARISON SCOPE";
  $("#comparison-scope-title").textContent = formal
    ? "已完成实验正式对照"
    : scope === "partial" ? "阶段性结果对照（非最终）" : "暂不可进行结果对照";
  $("#comparison-scope-status").textContent = formal
    ? "两侧实验均已完成并写入汇总；下面指标用于正式实验对照。"
    : scope === "partial"
      ? "以下指标来自已完成的联邦轮、客户端 metrics 或已持久化 checkpoint，仅表示阶段性结果，不用于最终 10 轮结论。"
      : "当前选择没有足够的两侧中间结果；正式对照指标暂不展示。";
  if (!ready) {
    const statuses = ["left", "right"].map((side) =>
      comparison[side].strategy + "：" + statusLabel(comparison[side].status)
    ).join(" · ");
    $("#comparison-not-ready").textContent =
      statuses + "。请在上方日志区查看实时状态，或选择已有联邦轮记录的实验。";
  }
}

function render(comparison) {
  state.comparison = comparison;
  renderComparisonScope(comparison);
  updateSelectionMeta("left", comparison.left);
  updateSelectionMeta("right", comparison.right);
  renderVerdict(comparison);
  renderParameters(comparison);
  renderMetrics(comparison);
  renderFederatedTiming(comparison);
  renderAggregationDetails(comparison);
  renderCloudDetails(comparison);
  renderRunDetails(comparison);
  $("#loading").hidden = true;
  $("#error").hidden = true;
  $("#dashboard").hidden = false;
}

function showError(error) {
  $("#loading").hidden = true;
  $("#dashboard").hidden = true;
  $("#error-message").textContent = error.message;
  $("#error").hidden = false;
}

async function loadComparison() {
  const left = selectors.left.value;
  const right = selectors.right.value;
  if (!left || !right) return;
  if (left === right) {
    showError(new Error("请选择两个不同的实验 ID"));
    return;
  }
  $("#loading").hidden = false;
  $("#dashboard").hidden = true;
  $("#error").hidden = true;
  const query = new URLSearchParams({ source: state.source, left, right });
  try {
    const comparison = await fetchJson(`/api/compare?${query}`);
    window.history.replaceState(null, "", `?${query}`);
    render(comparison);
  } catch (error) {
    showError(error);
  }
}

async function loadExperiments(force = false) {
  const button = $("#refresh-button");
  button.disabled = true;
  try {
    const query = new URLSearchParams({ source: state.source });
    if (force) query.set("refresh", "1");
    const response = await fetchJson(`/api/experiments?${query}`);
    state.experiments = response.experiments;
    if (state.experiments.length < 2) throw new Error("结果目录中至少需要两个实验才能进行对比");
    const current = selectors.left.value && selectors.right.value
      ? [selectors.left.value, selectors.right.value]
      : chooseDefaults(state.experiments);
    populateSelectors(state.experiments, current);
    populateLogSelector(state.experiments);
    await loadLogs();
    await loadSupervisor();
    await loadComparison();
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
}

async function loadSources() {
  try {
    const response = await fetchJson("/api/sources");
    state.sources = response.sources;
    if (!state.source) {
      state.source = state.sources.find((item) => item.id === "cloud" && item.available)?.id
        || state.sources.find((item) => item.available)?.id
        || "center";
    }
    for (const button of document.querySelectorAll("[data-source]")) {
      const source = state.sources.find((item) => item.id === button.dataset.source);
      button.disabled = source ? !source.available : true;
      button.title = source?.description || source?.message || "";
      button.classList.toggle("active", button.dataset.source === state.source);
    }
    const selected = state.sources.find((item) => item.id === state.source);
    if (!selected?.available) {
      state.source = state.sources.find((item) => item.available)?.id || "center";
    }
    for (const button of document.querySelectorAll("[data-source]")) {
      button.classList.toggle("active", button.dataset.source === state.source);
    }
  } catch {
    state.source = "center";
  }
}

initTheme();

selectors.left.addEventListener("change", loadComparison);
selectors.right.addEventListener("change", loadComparison);
selectors.log.addEventListener("change", loadLogs);
$("#refresh-button").addEventListener("click", () => loadExperiments(true));
$("#log-refresh-button").addEventListener("click", loadLogs);
$("#supervisor-form")?.addEventListener("submit", (event) => {
  event.preventDefault();
  sendSupervisorAction("configure");
});
for (const button of document.querySelectorAll("[data-supervisor-action]")) {
  button.addEventListener("click", () => sendSupervisorAction(button.dataset.supervisorAction));
}
$("#swap-button").addEventListener("click", () => {
  const left = selectors.left.value;
  selectors.left.value = selectors.right.value;
  selectors.right.value = left;
  loadComparison();
});

for (const button of document.querySelectorAll("[data-source]")) {
  button.addEventListener("click", async () => {
    if (button.disabled || state.source === button.dataset.source) return;
    state.source = button.dataset.source;
    for (const item of document.querySelectorAll("[data-source]")) {
      item.classList.toggle("active", item === button);
    }
    selectors.left.innerHTML = "";
    selectors.right.innerHTML = "";
    await loadExperiments();
  });
}

loadSources().then(loadExperiments);

window.setInterval(() => {
  if ($("#log-auto-refresh").checked) {
    loadLogs();
    loadSupervisor();
  }
}, 10_000);
