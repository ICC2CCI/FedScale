# 共享：从 YAML 读取扁平配置键到环境变量（CFG-1/CFG-2/CFG-3）
# 用法：
#   source scripts/_yaml_env.sh
#   yaml_load "configs/s3r12v3-fsdp-run.yaml"        # 读 run 配置 -> YCFG_<KEY>
#   yaml_load "deployment/nodes.yaml" "NODE_"        # 读节点清单 -> NODE_<KEY>
#
# 依赖 python3 + pyyaml（Server env 已有）。bash 端只做扁平 dict 读取。
_yaml_read_flat() {
  local file="$1"
  local prefix="${2:-YCFG_}"
  if [[ ! -f "$file" ]]; then
    echo "Warning: yaml not found: $file" >&2
    return 0
  fi
  # 优先用带 pyyaml 的 python（Server conda env）
  if [[ -x /home/pcllgr/miniconda3/envs/fedscale-server/bin/python ]]; then
    PY_BIN=/home/pcllgr/miniconda3/envs/fedscale-server/bin/python
  else
    PY_BIN=python3
  fi
  # 输出 export 语句，由调用方 eval 进当前 shell
  "$PY_BIN" - "$file" "$prefix" <<'PY'
import sys
try:
    import yaml
except ImportError:
    sys.stderr.write("Error: PyYAML not installed. Run: pip install pyyaml\n")
    sys.exit(2)
path, prefix = sys.argv[1], sys.argv[2]
raw = yaml.safe_load(open(path, encoding="utf-8")) or {}
flat = {}
for section, val in (raw or {}).items():
    if isinstance(val, dict):
        for k, v in val.items():
            if not isinstance(v, (dict, list)):
                flat[k] = v
    elif not isinstance(val, (dict, list)):
        flat[section] = val
for k, v in flat.items():
    key = prefix + k.upper()
    if isinstance(v, bool):
        sval = "true" if v else "false"
    else:
        sval = str(v).replace("'", "'\\''")
    print(f"export {key}='{sval}'")
PY
}

yaml_load() { eval "$(_yaml_read_flat "$1" "${2:-YCFG_}")"; }

yaml_get() {
  # yaml_get <prefix> <key> [default]
  local prefix="$1" key="$2" default="${3:-}"
  local varname="${prefix}${key^^}"
  echo "${!varname:-$default}"
}
