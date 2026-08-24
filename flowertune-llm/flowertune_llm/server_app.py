"""flowertune-llm: A Flower / FlowerTune app."""

import gc
import os
import time
import json
import re
from datetime import datetime

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from omegaconf import DictConfig

from flowertune_llm.dataset import replace_keys
from flowertune_llm.experiment_records import (
    aggregate_train_metrics as _aggregate_train_metrics,
    atomic_write_json as _atomic_write_json,
    ExperimentProgress as _ExperimentProgress,
    json_safe as _json_safe,
    round_metrics as _round_metrics,
    upsert_attempt as _upsert_attempt,
    write_attempt_artifact as _write_attempt_artifact,
    write_experiment_state as _write_experiment_state,
)
from flowertune_llm.models import get_model
from flowertune_llm.model_state import (
    FULL_LOCAL_INIT_KEY,
    SPARSE_DELTA_FORMAT_KEY,
    SPARSE_DELTA_INDEX_SUFFIX,
    SPARSE_DELTA_SCALE_SUFFIX,
    SPARSE_DELTA_VALUE_SUFFIX,
    checkpoint_path,
    decode_topk_int8_delta,
    encode_topk_int8_delta,
    finetuning_type,
    set_federated_state_dict,
    get_federated_state_dict,
    is_topk_int8_delta,
)
from flowertune_llm.fedscale_state import (
    apply_fedscale_int8_delta,
    build_canonical_layout,
    encode_fedscale_int8_delta,
    encoded_block_ids,
    is_fedscale_int8_delta,
    mask_hash,
    public_block_mask,
)
from flowertune_llm.resume import (
    absolute_round,
    get_resume_peft_path,
    get_resume_round,
    load_initial_arrays,
)
from flowertune_llm.model_storage import ModelStorage, ObjectStoreError, model_key
from flowertune_llm.object_store_strategy import (
    CONTROL_ARRAY_KEY,
    KubernetesAggregationLauncher,
    ObjectStoreFedAvg,
)

# Create ServerApp
app = ServerApp()


class FinalEvaluationOnlyMixin:
    """Schedule client evaluation only after the final training round."""

    def __init__(
        self,
        *args,
        final_evaluation_round=None,
        evaluation_timings=None,
        **kwargs,
    ):
        self._final_evaluation_round = final_evaluation_round
        self._evaluation_timings = (
            evaluation_timings if evaluation_timings is not None else {}
        )
        super().__init__(*args, **kwargs)

    def configure_evaluate(self, server_round, arrays, config, grid):
        if (
            self._final_evaluation_round is not None
            and server_round != self._final_evaluation_round
        ):
            return []
        if self._final_evaluation_round == server_round:
            self._evaluation_timings["started_perf"] = time.perf_counter()
            self._evaluation_timings["round"] = server_round
            self._evaluation_timings["started_at"] = datetime.now().isoformat()
        return super().configure_evaluate(server_round, arrays, config, grid)


class TimedFedAvg(FinalEvaluationOnlyMixin, FedAvg):
    """FedAvg wrapper that records pure server-side train aggregation time."""

    def __init__(self, *args, aggregation_timings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._aggregation_timings = aggregation_timings if aggregation_timings is not None else {}

    def aggregate_train(self, server_round: int, replies):
        started = time.perf_counter()
        try:
            # ``FedAvg`` can otherwise preserve ``initial_arrays`` when every
            # ClientApp reply is invalid/missing.  The evaluation callback then
            # persists that local-initialisation sentinel as if it were a real
            # global checkpoint, which makes a later resume both misleading and
            # unsafe.  Full-model matrix runs must advance only after every
            # selected client returned exactly one model ArrayRecord.
            valid_replies, failures = self._check_and_log_replies(
                replies, is_train=True, validate=True
            )
            if failures or not valid_replies:
                raise RuntimeError(
                    f"Client aggregation had {len(failures)} failures and "
                    f"{len(valid_replies)} valid results; refusing to advance "
                    "the global model from a partial aggregation"
                )
            return super().aggregate_train(server_round, valid_replies)
        finally:
            self._aggregation_timings[server_round] = round(
                time.perf_counter() - started, 4
            )


class SparseFullDeltaFedAvg(FinalEvaluationOnlyMixin, FedAvg):
    """FedAvg for Top-K INT8 full-model deltas.

    The first round starts from an identical verified client-local base model,
    so the server holds that base once and adds the weighted sparse deltas it
    receives.  This avoids transporting a complete FP16 model through the WAN.
    The server-to-client synchronization is also sparse: the server returns
    the current global state as a Top-K delta relative to the verified base.
    This keeps every round below the cross-cloud transport limit.
    """

    def __init__(self, *args, initial_state, topk_ratio, aggregation_timings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_state = {
            key: value.clone() if hasattr(value, "clone") else value
            for key, value in initial_state.items()
        }
        self._global_state = initial_state
        self._topk_ratio = float(topk_ratio)
        self._aggregation_timings = aggregation_timings if aggregation_timings is not None else {}

    def aggregate_train(self, server_round: int, replies):
        started = time.perf_counter()
        try:
            return self._aggregate_train_impl(server_round, replies)
        finally:
            self._aggregation_timings[server_round] = round(
                time.perf_counter() - started, 4
            )

    def _aggregate_train_impl(
        self, server_round: int, replies
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, failures = self._check_and_log_replies(
            replies, is_train=True, validate=True
        )
        if failures or not valid_replies:
            raise RuntimeError(
                f"Client aggregation had {len(failures)} failures and "
                f"{len(valid_replies)} valid results; refusing to advance "
                "the global model from a partial aggregation"
            )

        reply_contents = [message.content for message in valid_replies]
        client_states = [
            content[self.arrayrecord_key].to_torch_state_dict()
            for content in reply_contents
        ]
        if not all(is_topk_int8_delta(state) for state in client_states):
            raise ValueError(
                "Sparse full-update aggregation received a non-Top-K reply"
            )

        weights = [
            float(content["metrics"][self.weighted_by_key])
            for content in reply_contents
        ]
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Sparse full-update replies have no positive sample weight")

        # Mutate the retained FP16 global state in place. Only selected entries
        # are touched, so this does not create a second 6.85GB tensor copy.
        for client_state, weight in zip(client_states, weights):
            normalized_weight = weight / total_weight
            for key, target in self._global_state.items():
                if not target.is_floating_point():
                    continue
                index_key = key + SPARSE_DELTA_INDEX_SUFFIX
                value_key = key + SPARSE_DELTA_VALUE_SUFFIX
                scale_key = key + SPARSE_DELTA_SCALE_SUFFIX
                if index_key not in client_state:
                    continue
                indices = client_state[index_key].to(dtype=torch.long)
                quantized = client_state[value_key].to(dtype=torch.float32)
                scale = float(client_state[scale_key].reshape(-1)[0].item())
                update = (quantized * scale * normalized_weight).to(
                    dtype=target.dtype
                )
                target.reshape(-1).index_add_(0, indices, update)

        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return ArrayRecord(
            encode_topk_int8_delta(
                self._global_state, self._base_state, self._topk_ratio
            )
        ), metrics


class FedScaleBlockFedAvg(FinalEvaluationOnlyMixin, FedAvg):
    """FedAvg over a public canonical-block INT8 update contract.

    This class deliberately does *not* claim secure aggregation: Flower 1.28's
    installed ServerApp strategy API has no SecAgg+ workflow to attach here.
    It is the real ClientApp/ServerApp bridge and RoundPlan gate which will be
    wrapped by an aggregate-only protocol in the next security stage.
    """

    def __init__(
        self,
        *args,
        initial_state,
        base_state=None,
        initial_updated_block_ids=(),
        take_ownership=True,
        take_base_ownership=False,
        block_size,
        mask_ratio,
        encode_downlink=True,
        aggregation_timings=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        base_state = initial_state if base_state is None else base_state
        self._base_state = (
            base_state
            if take_base_ownership
            else {
                key: value.clone() if hasattr(value, "clone") else value
                for key, value in base_state.items()
            }
        )
        # During resume ``initial_state`` is already a dense reconstruction
        # which can be adopted directly.  Cloning it here would transiently
        # hold three OpenLLaMA states (base, reconstructed, clone) and exceed
        # the ServerApp memory envelope before the first client is scheduled.
        self._global_state = (
            initial_state
            if take_ownership
            else {
                key: value.clone() if hasattr(value, "clone") else value
                for key, value in initial_state.items()
            }
        )
        self._layout = build_canonical_layout(self._base_state, int(block_size))
        self._mask_ratio = float(mask_ratio)
        # A LoRA adapter is already compact. Keeping its downlink dense avoids
        # materializing the frozen 14B base just to reconstruct sparse blocks.
        self._encode_downlink = bool(encode_downlink)
        self._updated_block_ids: set[int] = set(initial_updated_block_ids)
        self._aggregation_timings = (
            aggregation_timings if aggregation_timings is not None else {}
        )

    def _set_round_plan(self, config, block_ids):
        config["fedscale-layout-hash"] = self._layout.layout_hash
        config["fedscale-mask-hash"] = mask_hash(self._layout, block_ids)
        config["fedscale-block-ids"] = ",".join(str(block_id) for block_id in block_ids)

    def configure_train(self, server_round, arrays, config, grid):
        block_ids = public_block_mask(
            self._layout, server_round, self._mask_ratio
        )
        self._set_round_plan(config, block_ids)
        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(self, server_round, arrays, config, grid):
        # Final evaluation receives the sparse global state relative to the
        # preflight-verified base.  Its public plan therefore lists the union
        # of blocks updated in completed rounds.
        block_ids = tuple(sorted(self._updated_block_ids))
        if block_ids:
            self._set_round_plan(config, block_ids)
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        started = time.perf_counter()
        try:
            return self._aggregate_train_impl(server_round, replies)
        finally:
            self._aggregation_timings[server_round] = round(
                time.perf_counter() - started, 4
            )

    def _aggregate_train_impl(self, server_round, replies):
        valid_replies, failures = self._check_and_log_replies(
            replies, is_train=True, validate=True
        )
        if failures or not valid_replies:
            raise RuntimeError(
                f"FedScale aggregation had {len(failures)} failures and "
                f"{len(valid_replies)} valid results; refusing partial update"
            )
        expected_ids = public_block_mask(
            self._layout, server_round, self._mask_ratio
        )
        reply_contents = [message.content for message in valid_replies]
        client_states = [
            content[self.arrayrecord_key].to_torch_state_dict()
            for content in reply_contents
        ]
        if not all(is_fedscale_int8_delta(state) for state in client_states):
            raise ValueError("FedScale aggregation received a non-block update")
        for client_state in client_states:
            received_ids = encoded_block_ids(client_state, self._layout)
            if received_ids != expected_ids:
                raise ValueError(
                    "FedScale client violated the public block mask: "
                    f"expected={expected_ids}, got={received_ids}"
                )

        # v1 starts with equal ICC weighting to avoid treating sensitive local
        # dataset size as metadata.  This is ordinary server-visible FedAvg
        # until SecAgg+ is integrated; do not use this mode for privacy claims.
        client_weight = 1.0 / len(client_states)
        for client_state in client_states:
            apply_fedscale_int8_delta(
                self._global_state,
                client_state,
                self._layout,
                weight=client_weight,
                clone=False,
            )
        self._updated_block_ids.update(expected_ids)
        downlink = (
            encode_fedscale_int8_delta(
                self._global_state,
                self._base_state,
                self._layout,
                tuple(sorted(self._updated_block_ids)),
            )
            if self._encode_downlink
            else self._global_state
        )
        metrics = self.train_metrics_aggr_fn(
            reply_contents, self.weighted_by_key
        )
        metrics["fedscale_public_block_count"] = len(expected_ids)
        metrics["fedscale_secagg_enabled"] = 0.0
        return ArrayRecord(downlink), metrics


def _load_full_base_state(model_cfg):
    """Load the preflight-verified base once for sparse delta aggregation."""
    model = get_model(model_cfg)
    try:
        return get_federated_state_dict(model, model_cfg)
    finally:
        del model
        gc.collect()


def _path_size(path: str | None) -> int | None:
    """Return durable checkpoint bytes without loading model tensors."""
    if not path or not os.path.exists(path):
        return None
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _write_lora_checkpoint(checkpoint_path_value, federated_state, model_cfg):
    """Persist an adapter checkpoint without loading the 3B base model.

    The arrays at the federation boundary are already the PEFT adapter state.
    Reconstructing the quantized base model just to call ``save_pretrained``
    adds several minutes and gigabytes of transient server memory to every
    checkpoint.  Resume loading only needs the adapter weights and a marker
    config; the active run config remains the source of truth for rebuilding
    the PEFT model.
    """
    os.makedirs(checkpoint_path_value, exist_ok=True)
    torch.save(
        federated_state,
        os.path.join(checkpoint_path_value, "adapter_model.bin"),
    )
    lora_cfg = model_cfg.lora
    adapter_config = {
        "base_model_name_or_path": str(model_cfg.name),
        "bias": "none",
        "inference_mode": False,
        "lora_alpha": int(lora_cfg.peft_lora_alpha),
        "lora_dropout": 0.075,
        "peft_type": "LORA",
        "r": int(lora_cfg.peft_lora_r),
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }
    with open(
        os.path.join(checkpoint_path_value, "adapter_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(adapter_config, handle, indent=2)


def _experiment_id(run_config) -> str:
    """Return a filesystem-safe experiment id supplied by the submitter."""
    value = str(run_config.get("experiment-id", "")).strip()
    if not value:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError("experiment-id must contain only letters, digits, '.', '_' or '-'")
    return value


def _round_record(records, round_number):
    """Read a Flower Result metric record using either numeric key form."""
    if not records:
        return {}
    return dict(records.get(round_number) or records.get(str(round_number)) or {})


def _write_final_evaluation_summary(
    save_path,
    run_id,
    result,
    final_round,
    evaluation_seconds,
):
    """Persist final global-model quality in the dashboard's formal schema."""
    raw = _round_record(result.evaluate_metrics_clientapp, final_round)
    completed = float(raw.get("evaluation_completed", 0.0) or 0.0) >= 0.999
    metrics = {
        "assistant_only": {
            "loss": raw.get("val_loss"),
            "ppl": raw.get("perplexity"),
            "evaluated_samples": raw.get("evaluated_samples"),
        },
        "rouge_l": {"f1": raw.get("rouge_l")},
        "bertscore": {"f1": raw.get("bertscore_f1")},
        "generation_quality": {
            "accuracy": raw.get("accuracy"),
            "macro_f1": raw.get("macro_f1"),
            "exact_match": raw.get("exact_match"),
        },
    }
    evaluation = {
        "status": "completed" if completed else "failed",
        "scope": "final_global_model",
        "excluded_from_training_timings": True,
        "evaluation_seconds": evaluation_seconds,
        "evaluated_at": datetime.now().isoformat(),
        "results": (
            [
                {
                    "label": "final_global",
                    "metrics": metrics,
                    "raw_client_aggregate": _json_safe(raw),
                }
            ]
            if raw
            else []
        ),
    }
    _write_attempt_artifact(
        save_path, "evaluation_summary.json", run_id, evaluation
    )
    return evaluation, raw, completed


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    started_at = datetime.now().isoformat()
    started_perf = time.perf_counter()
    run_id = getattr(context, "run_id", None)
    # The results directory is a PVC mount. A stable experiment-id lets an
    # external supervisor safely discover checkpoints after a failed run.
    experiment_id = _experiment_id(context.run_config)
    save_path = os.path.join("/app/results", experiment_id)
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    resume_peft_path = get_resume_peft_path(cfg)
    resume_round = get_resume_round(cfg)
    configured_round_timeout = context.run_config.get(
        "train.federated-round-timeout-seconds"
    )
    round_timeout = float(
        configured_round_timeout
        or os.environ.get("FEDERATED_ROUND_TIMEOUT_SECONDS", "3600")
    )
    state_path = os.path.join(save_path, "experiment_state.json")
    attempts_path = os.path.join(save_path, "experiment_attempts.json")
    experiment_config = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "started_at": started_at,
        "model": cfg.model.name,
        "dataset": cfg.dataset.name,
        "distributed_strategy": cfg.train.distributed_strategy,
        "finetuning_type": finetuning_type(cfg.model),
        "quantization": int(cfg.model.quantization),
        "ddp_cpu_offload": bool(cfg.train.ddp_cpu_offload),
        "rounds_this_attempt": num_rounds,
        "resume_round": resume_round,
        "resume_peft_path": resume_peft_path or "",
        "save_every_round": cfg.train.save_every_round,
        "federated_round_timeout_seconds": round_timeout,
        "run_config": _json_safe(dict(context.run_config)),
    }
    _write_attempt_artifact(
        save_path, "experiment_config.json", run_id, experiment_config
    )
    _upsert_attempt(
        attempts_path,
        run_id,
        status="running",
        started_at=started_at,
        resume_round=resume_round,
        rounds_this_attempt=num_rounds,
    )
    _write_experiment_state(
        state_path,
        replace=True,
        status="running",
        experiment_id=experiment_id,
        run_id=run_id,
        total_rounds=num_rounds,
        resume_round=resume_round,
        resume_peft_path=resume_peft_path or "",
        started_at=started_at,
        updated_at=started_at,
    )
    progress = _ExperimentProgress(
        save_path,
        state_path,
        experiment_id,
        run_id,
        heartbeat_seconds=float(os.environ.get("EXPERIMENT_HEARTBEAT_SECONDS", "30")),
    )
    progress.phase("starting", "实验已提交，正在准备初始状态", round_number=resume_round + 1)
    progress.start()

    # Track round timing across rounds
    round_timings = []
    aggregation_timings = {}
    final_evaluation_timings = {}
    initial_state_load_s = None
    server_base_state_load_s = None
    full_update_transport = str(
        getattr(cfg.train, "full_update_transport", "flower-rpc")
    ).lower()
    object_store_transport = full_update_transport == "object-store"

    # Start strategy, run FedAvg for `num_rounds`
    # timeout=None: wait indefinitely for client replies instead of the 3600s
    # default. The default caused the server to advance rounds every hour even
    # when clients hadn't finished training, creating a pile-up of pending Jobs.
    # Keep a failed client from blocking the whole experiment forever. The
    # default is longer than a healthy round but finite; deployments can tune
    # it with FEDERATED_ROUND_TIMEOUT_SECONDS.
    try:
        # Include model initialization in the recorded attempt. Cache or model
        # failures must produce a failed summary instead of leaving a stale
        # "running" state forever.
        use_full_local_init = bool(cfg.train.full_local_initialization)
        full_update_compression = str(
            cfg.train.full_update_compression).lower()
        if full_update_transport not in {"flower-rpc", "object-store"}:
            raise ValueError(
                "train.full-update-transport must be 'flower-rpc' or 'object-store'"
            )
        if object_store_transport and (
            finetuning_type(cfg.model) != "full"
            or full_update_compression != "none"
        ):
            raise ValueError(
                "object-store transport requires full fine-tuning with "
                "train.full-update-compression='none'"
            )
        if full_update_compression not in {"none", "topk-int8", "fedscale-int8"}:
            raise ValueError(
                "train.full-update-compression must be 'none', 'topk-int8', "
                "or 'fedscale-int8'"
            )
        if full_update_compression != "none":
            if (
                finetuning_type(cfg.model) != "full"
                and full_update_compression != "fedscale-int8"
            ):
                raise ValueError(
                    "Compressed federation requires full fine-tuning, except "
                    "FedScale INT8 which also supports LoRA adapters"
                )
            if (
                finetuning_type(cfg.model) == "full"
                and not use_full_local_init
                and not resume_peft_path
            ):
                raise ValueError(
                    "Compressed full-model federation requires "
                    "train.full-local-initialization=true for a fresh run or "
                    "a durable resume checkpoint"
                )

        object_storage = None
        initial_global = None
        if object_store_transport:
            progress.phase("loading_initial_model", "正在读取对象存储中的初始全局模型", round_number=resume_round + 1)
            object_storage = ModelStorage.from_env()
            initial_global_round = resume_round
            configured_uri = str(
                getattr(cfg.train, "object_store_initial_global_uri", "") or ""
            ).strip()
            initial_uri = configured_uri or object_storage.uri_for_key(
                model_key(experiment_id, initial_global_round, "global")
            )
            initial_global = object_storage.artifact_from_uri(
                initial_uri,
                experiment_id=experiment_id,
                round_number=initial_global_round,
                role="global",
            )
            arrays = ArrayRecord(
                {CONTROL_ARRAY_KEY: torch.zeros(1, dtype=torch.uint8)}
            )
            print(f"Using object-store global model: {initial_global.uri}")
            progress.phase("initial_model_ready", "初始全局模型已就绪", round_number=resume_round + 1, source="object-store")
        elif use_full_local_init:
            if finetuning_type(cfg.model) != "full":
                raise ValueError(
                    "train.full-local-initialization requires full fine-tuning"
                )
            if resume_round != 0 or resume_peft_path:
                raise ValueError(
                    "train.full-local-initialization is valid only for a fresh "
                    "first federated round"
                )
            print(
                "Using preflight-verified client-local full-model initialization"
            )
            arrays = ArrayRecord(
                {FULL_LOCAL_INIT_KEY: torch.zeros(1, dtype=torch.float32)}
            )
            progress.phase("initial_model_ready", "客户端本地初始模型校验完成", round_number=resume_round + 1, source="client-local")
        else:
            progress.phase("loading_initial_model", "正在读取中心端初始模型", round_number=resume_round + 1)
            initial_started = time.perf_counter()
            arrays = load_initial_arrays(cfg.model, resume_peft_path)
            initial_state_load_s = round(time.perf_counter() - initial_started, 4)
            progress.phase("initial_model_ready", "中心端初始模型已就绪", round_number=resume_round + 1, duration_seconds=initial_state_load_s)
        strategy_kwargs = {
            "fraction_train": cfg.strategy.fraction_train,
            "fraction_evaluate": 0.0 if object_store_transport else 1.0,
            # Flower's FedAvg constructor accepts this even when evaluation is
            # disabled, but keep its normal lower bound for API compatibility.
            "min_evaluate_nodes": 2,
            "train_metrics_aggr_fn": _aggregate_train_metrics,
        }
        if not object_store_transport:
            strategy_kwargs.update(
                {
                    "final_evaluation_round": num_rounds,
                    "evaluation_timings": final_evaluation_timings,
                }
            )
        if (
            not object_store_transport
            and finetuning_type(cfg.model) == "full"
            and full_update_compression in {"topk-int8", "fedscale-int8"}
        ):
            progress.phase("loading_base_state", "正在加载稀疏更新所需的基础模型", round_number=resume_round + 1)
            base_state_started = time.perf_counter()
            sparse_base_state = _load_full_base_state(cfg.model)
            server_base_state_load_s = round(
                time.perf_counter() - base_state_started, 4
            )
            progress.phase("base_state_ready", "稀疏更新基础模型已就绪", round_number=resume_round + 1, duration_seconds=server_base_state_load_s)
        if object_store_transport:
            strategy = ObjectStoreFedAvg(
                **strategy_kwargs,
                storage=object_storage,
                launcher=KubernetesAggregationLauncher(object_storage),
                experiment_id=experiment_id,
                initial_global=initial_global,
                resume_round=resume_round,
            )
        elif full_update_compression == "topk-int8":
            print(
                "Using Top-K INT8 sparse full-model updates: "
                f"ratio={cfg.train.full_update_topk_ratio}"
            )
            strategy = SparseFullDeltaFedAvg(
                **strategy_kwargs,
                initial_state=sparse_base_state,
                topk_ratio=float(cfg.train.full_update_topk_ratio),
                aggregation_timings=aggregation_timings,
            )
        elif full_update_compression == "fedscale-int8":
            if not 0.0 < float(cfg.train.fedscale_mask_ratio) <= 1.0:
                raise ValueError("train.fedscale-mask-ratio must be in (0, 1]")
            if int(cfg.train.fedscale_block_size) <= 0:
                raise ValueError("train.fedscale-block-size must be positive")
            print(
                "Using FedScale public canonical-block INT8 updates "
                "(SecAgg disabled for this bridge smoke): "
                f"block_size={cfg.train.fedscale_block_size}, "
                f"mask_ratio={cfg.train.fedscale_mask_ratio}"
            )
            is_lora_fedscale = finetuning_type(cfg.model) == "lora"
            initial_global_state = (
                arrays.to_torch_state_dict() if is_lora_fedscale else sparse_base_state
            )
            base_state = initial_global_state if is_lora_fedscale else sparse_base_state
            initial_updated_block_ids = ()
            take_ownership = True
            take_base_ownership = False
            if not is_lora_fedscale and not use_full_local_init:
                checkpoint_state = arrays.to_torch_state_dict()
                checkpoint_layout = build_canonical_layout(
                    sparse_base_state, int(cfg.train.fedscale_block_size)
                )
                if not is_fedscale_int8_delta(checkpoint_state):
                    raise ValueError(
                        "FedScale full-model resume requires a canonical-block "
                        "INT8 checkpoint created by this mode"
                    )
                initial_global_state = apply_fedscale_int8_delta(
                    sparse_base_state, checkpoint_state, checkpoint_layout
                )
                initial_updated_block_ids = encoded_block_ids(
                    checkpoint_state, checkpoint_layout
                )
                take_ownership = True
                take_base_ownership = True
                print(
                    "Reconstructed FedScale global state from the sparse "
                    "checkpoint for resumed aggregation"
                )
            strategy = FedScaleBlockFedAvg(
                **strategy_kwargs,
                initial_state=initial_global_state,
                base_state=base_state,
                initial_updated_block_ids=initial_updated_block_ids,
                take_ownership=take_ownership,
                take_base_ownership=take_base_ownership,
                block_size=int(cfg.train.fedscale_block_size),
                mask_ratio=float(cfg.train.fedscale_mask_ratio),
                encode_downlink=not is_lora_fedscale,
                aggregation_timings=aggregation_timings,
            )
        else:
            strategy = TimedFedAvg(
                **strategy_kwargs,
                aggregation_timings=aggregation_timings,
            )
        progress.phase(
            "waiting_for_clients",
            "已提交第一个联邦轮，等待客户端训练与回传",
            round_number=resume_round + 1,
            timeout_seconds=round_timeout,
            expected_clients=2,
        )
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({
                "save_path": save_path,
                "resume-round": resume_round,
            }),
            num_rounds=num_rounds,
            timeout=round_timeout,
            evaluate_fn=(
                get_object_store_evaluate_fn(
                    strategy,
                    save_path=save_path,
                    state_path=state_path,
                    experiment_id=experiment_id,
                    resume_round=resume_round,
                )
                if object_store_transport
                else get_evaluate_fn(
                cfg.model,
                cfg.train.save_every_round,
                num_rounds,
                save_path,
                round_timings,
                cfg,
                resume_round,
                state_path,
                experiment_id,
                sparse_base_state=getattr(strategy, "_base_state", None),
                aggregation_timings=aggregation_timings,
                initial_state_load_s=initial_state_load_s,
                server_base_state_load_s=server_base_state_load_s,
                final_evaluation_timings=final_evaluation_timings,
                progress=progress,
                )
            ),
        )
    except Exception as exc:
        progress.stop()
        progress.phase("failed", f"实验失败：{type(exc).__name__}: {exc}", round_number=resume_round + 1, emit=True)
        finished_at = datetime.now().isoformat()
        training_finished_perf = final_evaluation_timings.get(
            "started_perf", time.perf_counter()
        )
        duration_seconds = round(training_finished_perf - started_perf, 4)
        wall_clock_seconds = round(time.perf_counter() - started_perf, 4)
        summary = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "status": "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "wall_clock_seconds": wall_clock_seconds,
            "final_evaluation_status": "failed",
            "final_evaluation_seconds": final_evaluation_timings.get("seconds"),
            "resume_round": resume_round,
            "round_timing_scope": "previous_callback_end_to_current_aggregation",
            "round_timings": round_timings,
            "initial_state_load_s": initial_state_load_s,
            "server_base_state_load_s": server_base_state_load_s,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_attempt_artifact(
            save_path, "experiment_summary.json", run_id, summary
        )
        _upsert_attempt(
            attempts_path,
            run_id,
            status="failed",
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            error=summary["error"],
        )
        _write_experiment_state(
            state_path,
            status="failed",
            experiment_id=experiment_id,
            total_rounds=num_rounds,
            resume_round=resume_round,
            error=summary["error"],
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            updated_at=finished_at,
        )
        raise
    else:
        progress.stop()
        progress.phase("completed", "联邦训练已完成，正在写入最终汇总", round_number=resume_round + num_rounds, emit=True)
        finished_at = datetime.now().isoformat()
        completed_global_round = resume_round + num_rounds
        final_evaluation_seconds = final_evaluation_timings.get("seconds")
        if final_evaluation_seconds is None and final_evaluation_timings.get(
            "started_perf"
        ) is not None:
            final_evaluation_seconds = round(
                time.perf_counter() - final_evaluation_timings["started_perf"],
                4,
            )
        training_finished_perf = final_evaluation_timings.get(
            "started_perf", time.perf_counter()
        )
        duration_seconds = round(training_finished_perf - started_perf, 4)
        wall_clock_seconds = round(time.perf_counter() - started_perf, 4)
        if object_store_transport:
            final_evaluation_metrics = {}
            final_evaluation_completed = True
            final_evaluation_status = "not_requested_object_store_mvp"
            latest_checkpoint = strategy.current_global.uri
        else:
            _, final_evaluation_metrics, final_evaluation_completed = (
                _write_final_evaluation_summary(
                    save_path,
                    run_id,
                    result,
                    num_rounds,
                    final_evaluation_seconds,
                )
            )
            final_evaluation_status = (
                "completed" if final_evaluation_completed else "failed"
            )
            latest_checkpoint = checkpoint_path(
                save_path, completed_global_round, cfg.model
            )
        summary = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "status": "completed",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "wall_clock_seconds": wall_clock_seconds,
            "final_evaluation_status": final_evaluation_status,
            "final_evaluation_seconds": final_evaluation_seconds,
            "final_evaluation_metrics": final_evaluation_metrics,
            "resume_round": resume_round,
            "rounds_this_attempt": num_rounds,
            "completed_global_round": completed_global_round,
            "latest_checkpoint": latest_checkpoint,
            "round_timing_scope": "previous_callback_end_to_current_aggregation",
            "round_timings": round_timings,
            "initial_state_load_s": initial_state_load_s,
            "server_base_state_load_s": server_base_state_load_s,
            "aggregated_client_train_metrics": _round_metrics(
                result.train_metrics_clientapp, resume_round
            ),
            "aggregated_client_evaluate_metrics": _round_metrics(
                result.evaluate_metrics_clientapp, resume_round
            ),
            "server_evaluate_metrics": _round_metrics(
                result.evaluate_metrics_serverapp, resume_round
            ),
        }
        _write_attempt_artifact(
            save_path, "experiment_summary.json", run_id, summary
        )
        _upsert_attempt(
            attempts_path,
            run_id,
            status="completed",
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            completed_global_round=completed_global_round,
        )
        _write_experiment_state(
            state_path,
            status=(
                "completed"
                if final_evaluation_completed
                else "completed_with_evaluation_failure"
            ),
            experiment_id=experiment_id,
            total_rounds=num_rounds,
            resume_round=resume_round,
            latest_completed_round=completed_global_round,
            latest_checkpoint=latest_checkpoint,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            wall_clock_seconds=wall_clock_seconds,
            final_evaluation_status=final_evaluation_status,
            final_evaluation_seconds=final_evaluation_seconds,
            updated_at=finished_at,
        )


def get_object_store_evaluate_fn(
    strategy: ObjectStoreFedAvg,
    *,
    save_path: str,
    state_path: str,
    experiment_id: str,
    resume_round: int,
):
    """Persist only the immutable global URI after an object-store round.

    The MVP deliberately has no Flower final-evaluation phase because it would
    reintroduce a full global ArrayRecord. Evaluation can later download this
    artifact through the same verified object-store path.
    """

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        global_round = absolute_round(server_round, resume_round)
        artifact = strategy.current_global
        payload = {
            "round": global_round,
            "global_model_uri": artifact.uri,
            "global_model_sha256": artifact.sha256,
            "global_model_size": artifact.size,
        }
        _atomic_write_json(
            os.path.join(save_path, f"object_store_round_{global_round}.json"),
            payload,
        )
        _write_experiment_state(
            state_path,
            status="running",
            experiment_id=experiment_id,
            latest_completed_round=global_round,
            latest_checkpoint=artifact.uri,
            updated_at=datetime.now().isoformat(),
        )
        return MetricRecord({"object_store_global_bytes": artifact.size})

    return evaluate


# Get function that will be executed by the strategy
# Here we use it to save global model checkpoints and compute federated metrics
def get_evaluate_fn(
    model_cfg,
    save_every_round,
    total_round,
    save_path,
    round_timings,
    full_cfg=None,
    resume_round=0,
    state_path=None,
    experiment_id=None,
    sparse_base_state=None,
    aggregation_timings=None,
    initial_state_load_s=None,
    server_base_state_load_s=None,
    final_evaluation_timings=None,
    progress=None,
):
    """Return an evaluation function for saving global model and federated metrics."""

    prev_callback_end = [None]

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        global_round = absolute_round(server_round, resume_round)
        round_start = time.perf_counter()
        if progress is not None:
            progress.phase(
                "aggregating_round",
                f"第 {global_round} 轮客户端结果已回传，正在聚合",
                round_number=global_round,
            )

        # Client-side final evaluation happens immediately before this
        # centralized callback.  Use its start as the timing boundary so the
        # final quality evaluation is not attributed to the federated cycle.
        cycle_end = round_start
        final_evaluation_seconds = None
        if (
            final_evaluation_timings
            and final_evaluation_timings.get("round") == server_round
            and final_evaluation_timings.get("started_perf") is not None
        ):
            final_evaluation_timings["ended_perf"] = round_start
            final_evaluation_seconds = round(
                round_start - final_evaluation_timings["started_perf"], 4
            )
            final_evaluation_timings["seconds"] = final_evaluation_seconds
            cycle_end = final_evaluation_timings["started_perf"]

        # Time the federated cycle after the previous checkpoint/evaluation
        # callback completed and before this round's post-aggregation callback.
        # This excludes server-side checkpoint time and final-model evaluation.
        federated_cycle_s = None
        timing_entry = None
        if prev_callback_end[0] is not None:
            federated_cycle_s = cycle_end - prev_callback_end[0]
            timing_entry = {
                "round": global_round,
                "federated_cycle_s": round(federated_cycle_s, 4),
            }
            round_timings.append(timing_entry)
            print(
                f"Round {server_round} federated cycle: "
                f"{federated_cycle_s:.2f}s"
            )

        federated_metrics = {
            "server_round": global_round,
            "timestamp": datetime.now().isoformat(),
        }
        if aggregation_timings:
            aggregation_s = aggregation_timings.get(global_round)
            if aggregation_s is None:
                aggregation_s = aggregation_timings.get(server_round)
            if aggregation_s is not None:
                federated_metrics["server_fedavg_aggregation_s"] = aggregation_s
        if federated_cycle_s is not None:
            federated_metrics["federated_cycle_s"] = round(
                federated_cycle_s, 4
            )
        if final_evaluation_seconds is not None:
            federated_metrics["final_model_evaluation_s"] = (
                final_evaluation_seconds
            )

        # Save model checkpoint
        checkpoint_save_s = None
        checkpoint_bytes = None
        if server_round != 0 and (
            server_round == total_round or global_round % save_every_round == 0
        ):
            try:
                checkpoint_started = time.perf_counter()
                durable_checkpoint = checkpoint_path(
                    save_path, global_round, model_cfg
                )
                federated_state = arrays.to_torch_state_dict()
                if sparse_base_state is not None and is_topk_int8_delta(
                    federated_state
                ):
                    federated_state = decode_topk_int8_delta(
                        sparse_base_state, federated_state
                    )
                elif sparse_base_state is not None and is_fedscale_int8_delta(
                    federated_state
                ):
                    federated_state = apply_fedscale_int8_delta(
                        sparse_base_state,
                        federated_state,
                        build_canonical_layout(
                            sparse_base_state,
                            int(full_cfg.train.fedscale_block_size),
                        ),
                    )
                if finetuning_type(model_cfg) == "full":
                    os.makedirs(durable_checkpoint, exist_ok=True)
                    # Saving the already-aggregated tensors avoids allocating a
                    # second 6.5-GB OpenLLaMA model in the 24-GiB ServerApp.
                    torch.save(
                        federated_state,
                        os.path.join(durable_checkpoint, "model_state.pt"),
                    )
                else:
                    # The federated state is already the adapter state. Save it
                    # directly so checkpointing does not reload the 3B base
                    # model on the ServerApp CPU.
                    _write_lora_checkpoint(
                        durable_checkpoint,
                        federated_state,
                        model_cfg,
                    )
                checkpoint_bytes = _path_size(durable_checkpoint)
                checkpoint_save_s = round(time.perf_counter() - checkpoint_started, 4)
                print(
                    f"✓ Saved global model checkpoint: {durable_checkpoint} "
                    f"({checkpoint_save_s:.2f}s, {checkpoint_bytes or 0} bytes)"
                )
                if state_path:
                    _write_experiment_state(
                        state_path,
                        status="running",
                        experiment_id=experiment_id,
                        latest_completed_round=global_round,
                        latest_checkpoint=durable_checkpoint,
                        updated_at=datetime.now().isoformat(),
                    )

            except Exception as e:
                print(f"Warning: Failed to save model checkpoint: {e}")
                # A run without its advertised durable checkpoint cannot be
                # considered successful or safely resumed.
                raise

        server_post_aggregation_s = time.perf_counter() - round_start
        federated_metrics["server_post_aggregation_s"] = round(
            server_post_aggregation_s, 4
        )
        if checkpoint_save_s is not None:
            federated_metrics["checkpoint_save_s"] = checkpoint_save_s
            federated_metrics["checkpoint_bytes"] = checkpoint_bytes
        if timing_entry is not None:
            timing_entry["server_post_aggregation_s"] = round(
                server_post_aggregation_s, 4
            )

        # Save federated metrics for this round
        metrics_path = f"{save_path}/federated_metrics_round_{global_round}.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(federated_metrics, f, indent=2)
            print(f"✓ Saved federated metrics: {metrics_path}")
            if progress is not None:
                progress.phase(
                    "round_completed",
                    f"第 {global_round} 轮联邦指标已写入",
                    round_number=global_round,
                    federated_cycle_seconds=federated_metrics.get("federated_cycle_s"),
                    server_post_aggregation_seconds=federated_metrics.get("server_post_aggregation_s"),
                    checkpoint_seconds=federated_metrics.get("checkpoint_save_s"),
                )
        except Exception as e:
            print(f"Warning: Failed to save federated metrics: {e}")

        # Save cumulative round timings
        if round_timings:
            timings_path = f"{save_path}/federated_timings.json"
            try:
                with open(timings_path, 'w') as f:
                    json.dump(round_timings, f, indent=2)
            except Exception:
                pass

        prev_callback_end[0] = time.perf_counter()
        return MetricRecord()

    return evaluate
