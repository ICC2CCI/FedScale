#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RESULTS="$(cat logs/current_rerun_results_dir.txt 2>/dev/null || true)"
echo "=== aggregation server ==="
curl -fsS http://127.0.0.1:8080/api/round/current 2>/dev/null || echo 'server unreachable'
echo
echo "=== results dir ==="
echo "$RESULTS"
if [[ -n "${RESULTS:-}" && -f "$RESULTS/round_log.json" ]]; then
  python3 - <<PY
import json
from pathlib import Path
p=Path("$RESULTS")/"round_log.json"
d=json.loads(p.read_text())
print(f"completed {len(d)}/20")
if d:
  last=d[-1]
  print("last round", last.get("round"), "avg_train_loss", last.get("avg_train_loss"))
  t=last.get("timing_s") or {}
  print("round_wall_s", t.get("round_wall_s"), "transfer", (t.get("transfer") or {}))
  print("client_train_loss", last.get("client_train_loss"))
PY
else
  echo "round_log.json not ready"
fi
echo
echo "=== last server log lines ==="
tail -n 15 logs/aggregation_server.log 2>/dev/null || true
