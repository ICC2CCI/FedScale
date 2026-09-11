#!/usr/bin/env bash
# 实时查看当前 run 状态（OPS-1）。增强版：滞后检测、部分参与、失败告警。
# 用 --watch 周期刷新；退出码非 0 表示当前 run 有失败轮。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WATCH="${1:-}"
INTERVAL="${WATCH_INTERVAL:-5}"

render() {
  RESULTS="$(cat logs/current_rerun_results_dir.txt 2>/dev/null || true)"
  echo "=== aggregation server ==="
  curl -fsS http://127.0.0.1:8080/api/round/current 2>/dev/null || echo 'server unreachable'
  echo
  echo "=== results dir ==="
  echo "$RESULTS"
  if [[ -n "${RESULTS:-}" && -f "$RESULTS/round_log.json" ]]; then
    python3 - <<'PY'
import json, sys
from pathlib import Path
import os
RESULTS = os.environ.get("RESULTS", "")
p = Path(RESULTS) / "round_log.json"
d = json.loads(p.read_text())
# 从 run_meta 读 num_rounds
meta_p = Path(RESULTS) / "run_meta.json"
num_rounds = 20
if meta_p.exists():
    try:
        num_rounds = json.loads(meta_p.read_text()).get("num_rounds", 20)
    except Exception:
        pass
print(f"completed {len(d)}/{num_rounds}")
failed_rounds = [e for e in d if not e.get("avg_train_loss")]
if failed_rounds:
    print(f"!! FAILED rounds: {[e.get('round') for e in failed_rounds]}")
if d:
    last = d[-1]
    print("last round", last.get("round"), "avg_train_loss", last.get("avg_train_loss"))
    t = last.get("timing_s") or {}
    print("round_wall_s", t.get("round_wall_s"), "transfer", (t.get("transfer") or {}))
    print("client_train_loss", last.get("client_train_loss"))
    # OPS-1：部分参与 / 缺席告警
    if last.get("partial"):
        print(f"!! PARTIAL round {last.get('round')}: participated={last.get('participated_clients')} missing={last.get('missing_clients')}")
    clients = (t.get("clients") or {})
    for cid, ct in sorted(clients.items()):
        mode = ct.get("download_mode")
        mode_s = {0:"cache",1:"delta",2:"full",3:"local_base"}.get(int(round(mode)), "?") if mode is not None else "?"
        print(f"  client{cid} mode={mode_s} download_MiB={ct.get('download_global_MiB')} post_delta_MiB={ct.get('post_delta_MiB')} train_s={ct.get('train_local_s')}")
    # OPS-1：滞后检测——某 client round 落后 >= 2
    completed_per_client = {}
    for e in d:
        for cid in (e.get("client_train_loss") or {}):
            completed_per_client[cid] = e.get("round", 0)
    if completed_per_client:
        mx = max(completed_per_client.values())
        for cid, r in sorted(completed_per_client.items()):
            if mx - r >= 2:
                print(f"!! LAG: client{cid} at round {r}, others at {mx} (>=2 behind)")
PY
  else
    echo "round_log.json not ready"
  fi
  echo
  echo "=== last server log lines ==="
  if [[ -n "${RESULTS:-}" && -f "$RESULTS/logs/aggregation_server.log" ]]; then
    tail -n 20 "$RESULTS/logs/aggregation_server.log" 2>/dev/null || true
  else
    tail -n 20 logs/aggregation_server.log 2>/dev/null || true
  fi
}

if [[ "$WATCH" == "--watch" ]]; then
  while true; do
    clear
    export RESULTS="$(cat logs/current_rerun_results_dir.txt 2>/dev/null || true)"
    render
    sleep "$INTERVAL"
  done
else
  export RESULTS="$(cat logs/current_rerun_results_dir.txt 2>/dev/null || true)"
  render
fi
