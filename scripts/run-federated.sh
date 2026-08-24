#!/usr/bin/env bash
# 提交可配置的 Flower 联邦微调实验。
# 使用前请在另一终端运行 ./scripts/port-forward.sh。

set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL="Qwen/Qwen2.5-14B"
STRATEGY="fsdp"
DATASET="HuggingFaceH4/ultrachat_200k"
ROUNDS="50"
EXPERIMENT_ID=""
SAVE_EVERY_ROUND="1"
RESUME_PEFT_PATH=""
RESUME_ROUND="0"
EXTRA_CONFIG=()
FINETUNING_TYPE=""
QUANTIZATION=""
DDP_CPU_OFFLOAD=""
FULL_LOCAL_INITIALIZATION=""
FULL_UPDATE_TRANSPORT="flower-rpc"
OBJECT_STORE_INITIAL_GLOBAL_URI=""
DRY_RUN=false
SKIP_PREFLIGHT=false

usage() {
  cat <<'EOF'
用法：
  ./scripts/run-federated.sh [选项]

选项：
  --model MODEL          基座模型。默认：Qwen/Qwen2.5-14B
  --strategy STRATEGY    分布式方式：fsdp 或 ddp。默认：fsdp
  --dataset DATASET      Hugging Face 数据集。默认：HuggingFaceH4/ultrachat_200k
  --finetuning-type T    微调方式：lora 或 full。
  --quantization N       模型量化：0、4 或 8；full 必须使用 0。
  --ddp-cpu-offload      DDP 使用实验性的 FP16 CPU AdamW 状态卸载。
  --full-local-init      全参数第 1 轮使用三端已核验的一致本地基座。
  --object-store-initial-global-uri URI
                        使用对象存储传输完整模型，并指定第 0/恢复轮的 s3:// 全局模型。
  --rounds N             联邦训练轮数。默认：50
  --experiment-id ID     稳定实验标识；用于结果目录和断点恢复。
  --save-every-round N   每 N 个成功聚合轮保存全局 LoRA 检查点。默认：1
  --resume-peft-path P   从中心端持久化的 LoRA、全量或 FedScale 检查点恢复。
  --resume-round N       恢复检查点已完成的全局轮数。默认：0
  --set KEY=VALUE        追加 Flower run-config；可重复使用。
  --skip-preflight       跳过集群预检；仅用于已知的诊断场景。
  --dry-run              仅打印最终提交命令，不提交。
  -h, --help             显示本帮助。

示例：
  # OpenLLaMA-3B 的 FSDP LoRA 对照实验
  ./scripts/run-federated.sh \
    --model openlm-research/open_llama_3b_v2 \
    --strategy fsdp \
    --dataset vicgalle/alpaca-gpt4 \
    --rounds 50

  # 同模型的 DDP 实验，并覆盖局部训练步数
  ./scripts/run-federated.sh \
    --model openlm-research/open_llama_3b_v2 \
    --strategy ddp --rounds 50 \
    --set train.training-arguments.max-steps=50
EOF
}

while (($#)); do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --strategy) STRATEGY="${2:-}"; shift 2 ;;
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --finetuning-type) FINETUNING_TYPE="${2:-}"; shift 2 ;;
    --quantization) QUANTIZATION="${2:-}"; shift 2 ;;
    --ddp-cpu-offload) DDP_CPU_OFFLOAD="true"; shift ;;
    --full-local-init) FULL_LOCAL_INITIALIZATION="true"; shift ;;
    --object-store-initial-global-uri) FULL_UPDATE_TRANSPORT="object-store"; OBJECT_STORE_INITIAL_GLOBAL_URI="${2:-}"; shift 2 ;;
    --rounds) ROUNDS="${2:-}"; shift 2 ;;
    --experiment-id) EXPERIMENT_ID="${2:-}"; shift 2 ;;
    --save-every-round) SAVE_EVERY_ROUND="${2:-}"; shift 2 ;;
    --resume-peft-path) RESUME_PEFT_PATH="${2:-}"; shift 2 ;;
    --resume-round) RESUME_ROUND="${2:-}"; shift 2 ;;
    --set) EXTRA_CONFIG+=("${2:-}"); shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1（使用 --help 查看帮助）" ;;
  esac
done

[[ -n "$MODEL" && "$MODEL" != *"'"* ]] || die "模型名不能为空或包含单引号。"
[[ -n "$DATASET" && "$DATASET" != *"'"* ]] || die "数据集名不能为空或包含单引号。"
[[ "$STRATEGY" == "fsdp" || "$STRATEGY" == "ddp" ]] || die "--strategy 只能是 fsdp 或 ddp。"
[[ -z "$FINETUNING_TYPE" || "$FINETUNING_TYPE" == "lora" || "$FINETUNING_TYPE" == "full" ]] || die "--finetuning-type 只能是 lora 或 full。"
[[ -z "$QUANTIZATION" || "$QUANTIZATION" == "0" || "$QUANTIZATION" == "4" || "$QUANTIZATION" == "8" ]] || die "--quantization 只能是 0、4 或 8。"
[[ "$FINETUNING_TYPE" != "full" || -z "$QUANTIZATION" || "$QUANTIZATION" == "0" ]] || die "full 微调必须使用 --quantization 0。"
[[ "$DDP_CPU_OFFLOAD" != "true" || "$STRATEGY" == "ddp" ]] || die "--ddp-cpu-offload 只能与 --strategy ddp 一起使用。"
[[ "$FULL_LOCAL_INITIALIZATION" != "true" || "$FINETUNING_TYPE" == "full" ]] || die "--full-local-init 必须与 --finetuning-type full 一起使用。"
[[ "$FULL_LOCAL_INITIALIZATION" != "true" || "$RESUME_ROUND" == "0" ]] || die "--full-local-init 只能用于全新的第 1 轮。"
[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || die "--rounds 必须为正整数。"
[[ "$SAVE_EVERY_ROUND" =~ ^[1-9][0-9]*$ ]] || die "--save-every-round 必须为正整数。"
[[ "$RESUME_ROUND" =~ ^[0-9]+$ ]] || die "--resume-round 必须为非负整数。"
[[ -z "$EXPERIMENT_ID" || "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || die "--experiment-id 只能包含字母、数字、.、_ 或 -。"
if [[ "$FULL_UPDATE_TRANSPORT" == "object-store" ]]; then
  [[ "$FINETUNING_TYPE" == "full" && "$QUANTIZATION" == "0" ]] || die "对象存储传输必须指定 --finetuning-type full --quantization 0。"
  [[ -n "$OBJECT_STORE_INITIAL_GLOBAL_URI" ]] || die "对象存储传输必须指定 --object-store-initial-global-uri s3://...。"
  [[ -z "$FULL_LOCAL_INITIALIZATION" ]] || die "对象存储传输从 S3 下载首个全局模型，不能使用 --full-local-init。"
  compression="none"
  for item in "${EXTRA_CONFIG[@]}"; do
    [[ "$item" != train.full-update-compression=* ]] || compression="${item#*=}"
  done
  validator=(python3 "$FLOWER_DIR/configs/object_store_config.py" --strategy "$STRATEGY" --rounds "$ROUNDS" --finetuning-type "$FINETUNING_TYPE" --quantization "$QUANTIZATION" --compression "$compression" --save-every-round "$SAVE_EVERY_ROUND" --initial-global-uri "$OBJECT_STORE_INITIAL_GLOBAL_URI" --resume-round "$RESUME_ROUND")
  for item in "${EXTRA_CONFIG[@]}"; do validator+=(--override "$item"); done
  "${validator[@]}" || die "对象存储配置编译/校验失败。"
fi

RUN_CONFIG=(
  "model.name='$MODEL'"
  "dataset.name='$DATASET'"
  "train.distributed-strategy='$STRATEGY'"
  "num-server-rounds=$ROUNDS"
  "train.save-every-round=$SAVE_EVERY_ROUND"
)
[[ -z "$EXPERIMENT_ID" ]] || RUN_CONFIG+=("experiment-id='$EXPERIMENT_ID'")
[[ -z "$FINETUNING_TYPE" ]] || RUN_CONFIG+=("model.finetuning-type='$FINETUNING_TYPE'")
[[ -z "$QUANTIZATION" ]] || RUN_CONFIG+=("model.quantization=$QUANTIZATION")
[[ -z "$DDP_CPU_OFFLOAD" ]] || RUN_CONFIG+=("train.ddp-cpu-offload=true")
[[ -z "$FULL_LOCAL_INITIALIZATION" ]] || RUN_CONFIG+=("train.full-local-initialization=true")
[[ "$FULL_UPDATE_TRANSPORT" == "flower-rpc" ]] || RUN_CONFIG+=("train.full-update-transport='object-store'" "train.object-store-initial-global-uri='$OBJECT_STORE_INITIAL_GLOBAL_URI'")
[[ -z "$RESUME_PEFT_PATH" ]] || RUN_CONFIG+=("train.resume-peft-path='$RESUME_PEFT_PATH'")
[[ "$RESUME_ROUND" == "0" ]] || RUN_CONFIG+=("train.resume-round=$RESUME_ROUND")
RUN_CONFIG+=("${EXTRA_CONFIG[@]}")
RUN_CONFIG_STRING="${RUN_CONFIG[*]}"

echo '实验配置：'
printf '  model: %s\n  strategy: %s\n  dataset: %s\n  rounds: %s\n' \
  "$MODEL" "$STRATEGY" "$DATASET" "$ROUNDS"
printf '  checkpoint interval: %s\n' "$SAVE_EVERY_ROUND"
[[ -z "$EXPERIMENT_ID" ]] || printf '  experiment-id: %s\n' "$EXPERIMENT_ID"
[[ -z "$RESUME_PEFT_PATH" ]] || printf '  resume: round %s from %s\n' "$RESUME_ROUND" "$RESUME_PEFT_PATH"
[[ ${#EXTRA_CONFIG[@]} -eq 0 ]] || printf '  overrides: %s\n' "${EXTRA_CONFIG[*]}"
echo "  run-config: $RUN_CONFIG_STRING"

if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

require_flower_connection
if [[ "$SKIP_PREFLIGHT" == false ]]; then
  "$(dirname "$0")/preflight.sh" "$MODEL"
else
  echo '警告：已跳过预检。'
fi
cd "$FLOWER_DIR"
flwr_cli run flowertune-llm cross-cloud --run-config "$RUN_CONFIG_STRING"
