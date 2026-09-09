#!/usr/bin/env bash
# 在 ICC 节点导出 Qwen 初始 state_dict 并上传到 Central MinIO（round-0）
set -euo pipefail

MODEL_PATH="${1:-model/Qwen/Qwen2.5-0.5B}"
MINIO_ENDPOINT="${2:-http://192.168.235.42:9000}"
ACCESS_KEY="${MINIO_ACCESS_KEY:-fedscale}"
SECRET_KEY="${MINIO_SECRET_KEY:-fedscale-minio-2026}"
BUCKET="${MINIO_BUCKET:-fedscale-bucket}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/experiments:${PYTHONPATH:-}"

python - <<PY
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from shared.minio_client import MinIOClient
from shared.protocol import global_state_key

model_path = Path("${MODEL_PATH}")
if not model_path.is_absolute():
    model_path = Path("${ROOT}") / model_path
print(f"Loading {model_path} on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.float16,
    trust_remote_code=False,
    attn_implementation="eager",
    local_files_only=True,
)
state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
del model
print(f"tensors={len(state)}")
client = MinIOClient(
    endpoint="${MINIO_ENDPOINT}",
    access_key="${ACCESS_KEY}",
    secret_key="${SECRET_KEY}",
    bucket="${BUCKET}",
)
key = global_state_key(0)
client.put_torch(key, state)
print("Uploaded s3://${BUCKET}/" + key)
PY
