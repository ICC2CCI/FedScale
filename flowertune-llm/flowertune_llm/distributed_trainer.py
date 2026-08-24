"""
Distributed trainer for multi-node multi-GPU training.
This module is launched as a Kubernetes Job and supports PyTorch DDP and FSDP.
"""

# Configure HuggingFace BEFORE any imports
# This must be done before importing transformers/huggingface_hub
import os
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/.cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("TRAIN_HF_HUB_OFFLINE", "1"))
os.environ.setdefault("HF_DATASETS_OFFLINE", os.environ.get("TRAIN_HF_DATASETS_OFFLINE", "1"))
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Disable hf_transfer to avoid XET issues
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import json
import gc
import functools
import time
import torch
import torch.distributed as dist
from datetime import datetime
from transformers import TrainingArguments
from trl import SFTTrainer
from omegaconf import DictConfig
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from flowertune_llm.train_models import get_model
from flowertune_llm.cpu_offload_optimizer import CPUOffloadAdamW
from flowertune_llm.model_state import (
    FULL_LOCAL_INIT_KEY,
    finetuning_type,
    get_federated_state_dict,
    set_federated_state_dict,
)
from flowertune_llm.train_dataset import (
    format_dataset_for_sft,
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
)
from flowertune_llm.metrics import StepMetricsCallback, ResourceMonitor, save_metrics_detailed
from flowertune_llm.evaluator import create_eval_split, compute_validation_metrics, compute_downstream_metrics


def _env_bool(name, default):
    """Read a boolean trainer option without treating arbitrary text as true."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _config_bool(value, default=False):
    """Read a JSON config boolean without treating "false" as truthy."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _fsdp_options():
    """Return communication-related FSDP options from the Job environment."""
    strategy_name = os.environ.get("FSDP_SHARDING_STRATEGY", "FULL_SHARD").upper()
    try:
        sharding_strategy = ShardingStrategy[strategy_name]
    except KeyError as exc:
        valid = ", ".join(item.name for item in ShardingStrategy)
        raise ValueError(
            f"Unsupported FSDP_SHARDING_STRATEGY={strategy_name!r}; choose one of {valid}"
        ) from exc

    prefetch_name = os.environ.get("FSDP_BACKWARD_PREFETCH", "BACKWARD_PRE").upper()
    if prefetch_name in {"", "NONE"}:
        backward_prefetch = None
    else:
        try:
            backward_prefetch = BackwardPrefetch[prefetch_name]
        except KeyError as exc:
            raise ValueError(
                "FSDP_BACKWARD_PREFETCH must be NONE, BACKWARD_PRE, or BACKWARD_POST"
            ) from exc

    return {
        "sharding_strategy": sharding_strategy,
        "backward_prefetch": backward_prefetch,
        "forward_prefetch": _env_bool("FSDP_FORWARD_PREFETCH", False),
        "limit_all_gathers": _env_bool("FSDP_LIMIT_ALL_GATHERS", True),
    }


def _full_model_fsdp_auto_wrap_policy(model):
    """Wrap complete transformer blocks for full-model FSDP.

    Hugging Face applies activation checkpointing at the decoder-layer
    boundary.  Splitting large Linear modules inside that boundary lets a
    re-computed layer overlap with FSDP freeing/prefetching its flat
    parameters, which is unsafe on the pinned torch/CUDA stack.  Keep both
    mechanisms on the same module boundary.
    """
    no_split_names = set(getattr(model, "_no_split_modules", ()) or ())
    transformer_layer_classes = {
        type(module)
        for module in model.modules()
        if type(module).__name__ in no_split_names
    }
    if transformer_layer_classes:
        print(
            "Full-model FSDP transformer layer classes: "
            + ", ".join(sorted(cls.__name__ for cls in transformer_layer_classes))
        )
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_classes,
        )

    print(
        "WARNING: model exposes no transformer no-split class; falling back "
        "to size-based FSDP wrapping"
    )
    return functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=10_000_000,
    )


def setup_distributed():
    """Setup distributed training environment for multi-node training."""
    # HuggingFace env vars already set at module level

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_nodes = int(os.environ.get("NUM_NODES", "1"))
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "1"))

    # Get node rank from environment or pod name
    # Priority: BATCH_JOB_COMPLETION_INDEX > NODE_RANK > extract from pod name
    if "BATCH_JOB_COMPLETION_INDEX" in os.environ:
        node_rank = int(os.environ["BATCH_JOB_COMPLETION_INDEX"])
    elif "NODE_RANK" in os.environ:
        node_rank = int(os.environ["NODE_RANK"])
    else:
        # Extract from pod name: {job_name}-{index}-{random}
        pod_name = os.environ.get("POD_HOSTNAME", "")
        if pod_name:
            # Pod name format: train-round-1-0-xxx-2-abc12
            parts = pod_name.rsplit("-", 2)  # Split from right: ['train-round-1-0-xxx', '2', 'abc12']
            if len(parts) >= 2:
                try:
                    node_rank = int(parts[-2])  # Second to last part is the index
                except ValueError:
                    node_rank = 0
            else:
                node_rank = 0
        else:
            node_rank = 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = node_rank * gpus_per_node + local_rank

    print(f"[Rank {rank}] Setting up distributed training")
    print(f"[Rank {rank}] Master: {master_addr}:{master_port}")
    print(f"[Rank {rank}] World size: {world_size}, Node rank: {node_rank}, Local rank: {local_rank}")
    print(f"[Rank {rank}] Pod hostname: {os.environ.get('POD_HOSTNAME', 'N/A')}")
    print(f"[Rank {rank}] BATCH_JOB_COMPLETION_INDEX: {os.environ.get('BATCH_JOB_COMPLETION_INDEX', 'N/A')}")

    # Disable IPv6 for NCCL (common issue in Kubernetes).  Keep the historical
    # socket fallback, but allow the launcher/deployment to opt into RDMA or a
    # selected interface without editing this module.
    import socket
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    # Wait for master address to be resolvable
    import time
    print(f"[Rank {rank}] Waiting for master address to be resolvable...")
    for attempt in range(30):  # Wait up to 60 seconds
        try:
            socket.getaddrinfo(master_addr, int(master_port), socket.AF_INET)
            print(f"[Rank {rank}] ✓ Master address resolved successfully")
            break
        except socket.gaierror as e:
            if attempt < 29:
                print(f"[Rank {rank}] Waiting for DNS resolution... ({attempt+1}/30): {e}")
                time.sleep(2)
            else:
                print(f"[Rank {rank}] ✗ Failed to resolve master address after 60 seconds")
                raise

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, node_rank


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def peft_fsdp_auto_wrap_policy(model):
    """Build PEFT's FSDP policy without relying on Accelerate internals.

    PEFT 0.6.2 calls an Accelerate helper removed by the newer Accelerate
    version in the training image. This is the equivalent policy: isolate
    trainable LoRA leaf modules and wrap each supported decoder layer.
    """
    # Qwen3 uses Qwen3DecoderLayer while OpenLLaMA uses
    # LlamaDecoderLayer.  Discover decoder blocks by their stable class-name
    # suffix so the same PEFT/FSDP path works for both model families.
    transformer_layer_cls = tuple(
        sorted(
            {
                type(module)
                for module in model.modules()
                if module.__class__.__name__.endswith("DecoderLayer")
            },
            key=lambda cls: cls.__name__,
        )
    )
    if not transformer_layer_cls:
        raise RuntimeError(
            "Could not find a *DecoderLayer class for FSDP wrapping"
        )

    def lambda_policy_fn(module):
        return (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        )

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda_policy_fn,
    )
    transformer_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )
    return functools.partial(
        _or_policy,
        policies=[lambda_policy, transformer_policy],
    )


def train_distributed():
    """Main distributed training function."""
    rank, local_rank, world_size, node_rank = setup_distributed()

    try:
        # Load configuration from environment
        model_weights_path = os.environ.get("MODEL_WEIGHTS_PATH", "")
        train_config_str = os.environ.get("TRAIN_CONFIG", "{}")
        train_config = json.loads(train_config_str)

        # Extract training parameters
        partition_id = train_config.get("partition_id", 0)
        num_partitions = train_config.get("num_partitions", 1)
        dataset_name = train_config.get("dataset_name", "vicgalle/alpaca-gpt4")
        max_train_samples = int(train_config.get("max_train_samples", 0) or 0)
        model_name = train_config.get("model_name", "Qwen/Qwen3-14B")
        learning_rate = train_config.get("learning_rate", 5e-5)
        batch_size = train_config.get("batch_size", 4)
        seq_length = train_config.get("seq_length", 512)
        output_dir = train_config.get("output_dir", "/app/outputs")
        # Training control (passed from ClientApp via pyproject.toml config)
        num_train_epochs = train_config.get("num_train_epochs", 3)
        max_steps = train_config.get("max_steps", 10)
        gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
        run_local_evaluation = _config_bool(
            train_config.get("run_local_evaluation"), False
        )
        logging_steps = train_config.get("logging_steps", 10)
        save_steps = train_config.get("save_steps", 1000)
        save_total_limit = train_config.get("save_total_limit", 1)
        lr_scheduler_type = train_config.get("lr_scheduler_type", "constant")
        distributed_strategy = str(
            train_config.get("distributed_strategy", "fsdp")
        ).lower()
        if distributed_strategy not in {"ddp", "fsdp"}:
            raise ValueError(
                "distributed_strategy must be either 'ddp' or 'fsdp', got "
                f"{distributed_strategy!r}"
            )
        configured_finetuning_type = str(
            train_config.get("finetuning_type", "lora")
        ).lower()
        quantization = int(train_config.get("quantization", 4))
        gradient_checkpointing = _config_bool(
            train_config.get("gradient_checkpointing"), True
        )
        ddp_cpu_offload = _config_bool(
            train_config.get("ddp_cpu_offload"), False
        )
        full_update_compression = str(
            train_config.get("full_update_compression", "none")
        ).lower()
        if ddp_cpu_offload and distributed_strategy != "ddp":
            raise ValueError("train.ddp-cpu-offload is valid only with DDP")

        print(f"\n{'='*60}")
        print(f"[Rank {rank}] Loading model and data...")
        print(f"{'='*60}")

        # Debug: Check PVC mount and cache status
        from pathlib import Path as PathLib
        cache_dir = os.environ.get("HF_HOME", "/app/.cache/huggingface")
        cache_path = PathLib(cache_dir)

        print(f"\n[Rank {rank}] Cache directory check:")
        print(f"  HF_HOME: {cache_dir}")
        print(f"  Cache path exists: {cache_path.exists()}")

        if cache_path.exists():
            # List top-level contents
            items = list(cache_path.iterdir())
            print(f"  Contents ({len(items)} items):")
            for item in items[:5]:
                print(f"    - {item.name}")

            # Check hub directory
            hub_path = cache_path / "hub"
            if hub_path.exists():
                print(f"  Hub directory exists: YES")
                model_cache = hub_path / f"models--{model_name.replace('/', '--')}"
                if model_cache.exists():
                    print(f"  Model cache exists: YES")
                    snapshots_path = model_cache / "snapshots"
                    if snapshots_path.exists():
                        snapshot_dirs = list(snapshots_path.iterdir())
                        if snapshot_dirs:
                            print(f"  Snapshots found: {len(snapshot_dirs)}")
                            latest_snapshot = snapshot_dirs[0]
                            files = list(latest_snapshot.iterdir())
                            print(f"  Files in snapshot: {len(files)}")
                            if files:
                                total_size = sum(f.stat().st_size for f in files if f.is_file())
                                print(f"  Total size: {total_size/1024/1024:.1f}MB")
                        else:
                            print(f"  WARNING: Snapshots directory is EMPTY!")
                    else:
                        print(f"  WARNING: Snapshots directory not found!")
                else:
                    print(f"  WARNING: Model cache not found!")
            else:
                print(f"  WARNING: Hub directory not found!")
        else:
            print(f"  WARNING: Cache directory does not exist!")

        print(f"\n{'='*60}\n")

        # Load tokenizer and data
        (
            tokenizer,
            data_collator,
            formatting_prompts_func,
        ) = get_tokenizer_and_data_collator_and_propt_formatting(model_name)

        trainset = load_data(partition_id, num_partitions, dataset_name)
        if max_train_samples > 0 and len(trainset) > max_train_samples:
            # UltraChat formatting materializes an Arrow cache.  Cap the raw
            # partition before rendering messages so a 14B FP16 worker never
            # maps all 100k+ records only to discard nearly all of them later.
            trainset = trainset.shuffle(seed=42).select(range(max_train_samples))
            print(
                f"[Rank {rank}] Capped raw client partition to "
                f"{max_train_samples} deterministic samples before formatting"
            )
        trainset = format_dataset_for_sft(trainset, tokenizer, model_name)

        # Load model
        model_cfg = DictConfig({
            "name": model_name,
            "finetuning_type": configured_finetuning_type,
            "quantization": quantization,
            "gradient_checkpointing": gradient_checkpointing,
            "distributed_strategy": distributed_strategy,
            "ddp_cpu_offload": ddp_cpu_offload,
            "lora": {
                "peft_lora_r": int(train_config.get("lora_r", 32)),
                "peft_lora_alpha": int(train_config.get("lora_alpha", 64)),
            }
        })
        tuning = finetuning_type(model_cfg)
        print(
            f"[Rank {rank}] Effective model config: tuning={tuning}, "
            f"quantization={quantization}, "
            f"gradient_checkpointing={gradient_checkpointing}, "
            f"ddp_cpu_offload={ddp_cpu_offload}"
        )

        model = get_model(model_cfg)

        # Load initial weights if provided
        if model_weights_path and os.path.exists(model_weights_path):
            print(f"[Rank {rank}] Loading initial weights from {model_weights_path}")
            initial_weights = torch.load(model_weights_path, map_location="cpu")
            if (
                tuning == "full"
                and set(initial_weights) == {FULL_LOCAL_INIT_KEY}
            ):
                print(
                    f"[Rank {rank}] Using identical preflight-verified local "
                    "base snapshot for first-round initialization"
                )
            else:
                set_federated_state_dict(model, model_cfg, initial_weights)
            del initial_weights
            gc.collect()

        # Export the known local base before moving or wrapping the model.  The
        # sparse full-update mode is restricted by ServerApp to a fresh first
        # round with an identical preflight-verified base at every client.
        # Gathering a FULL_STATE_DICT from an already wrapped FSDP model here
        # used to exercise a second unshard/reshard cycle immediately before
        # the first forward pass and produced illegal CUDA accesses on V100.
        if tuning == "full" and full_update_compression in {
            "topk-int8",
            "fedscale-int8",
        }:
            job_name_for_initial_state = os.environ.get(
                "JOB_NAME", "distributed_job"
            )
            initial_state_path = (
                f"{output_dir}/{job_name_for_initial_state}/initial_model_state.pt"
            )
            if rank == 0:
                initial_federated_state = get_federated_state_dict(
                    model, model_cfg
                )
                os.makedirs(os.path.dirname(initial_state_path), exist_ok=True)
                torch.save(initial_federated_state, initial_state_path)
                del initial_federated_state
                gc.collect()
                print(
                    "[Rank 0] Saved pre-wrap initial full-model state for "
                    f"compressed federated update: {initial_state_path}"
                )
            # Do not let another rank enter the first model collective while
            # rank 0 is still serializing the shared-PVC baseline.
            dist.barrier(device_ids=[local_rank])

        # Enable gradient checkpointing before distributed wrapping.
        if gradient_checkpointing:
            print(f"[Rank {rank}] Enabling gradient checkpointing...")
            if tuning == "lora":
                # get_model returns a PeftModel.  Enable the input-gradient
                # hook on that final wrapper immediately before Transformers
                # installs re-entrant checkpointing; otherwise every frozen
                # base-model input is detached and the first backward pass has
                # no grad_fn.
                model.enable_input_require_grads()
            checkpoint_kwargs = (
                {"use_reentrant": False}
                if distributed_strategy == "fsdp" and tuning == "full"
                else None
            )
            if checkpoint_kwargs is None:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=checkpoint_kwargs
                )
            model.config.use_cache = False  # Disable KV cache for training

        # Move model to GPU before distributed wrapping. The FSDP path uses
        # float-backed bitsandbytes storage configured in train_models.py.
        model = model.to(f"cuda:{local_rank}")

        # Initialize metrics before wrapping the model. DDP's communication
        # hook is registered immediately after DistributedDataParallel is
        # constructed and needs the callback as its state object.
        step_callback = StepMetricsCallback(seq_length=seq_length)

        if distributed_strategy == "fsdp":
            # PEFT's policy isolates trainable LoRA leaf modules from frozen
            # transformer weights, which is required with use_orig_params=False.
            # FULL_SHARD shards parameters, gradients, and optimizer states.
            fsdp_options = _fsdp_options()
            print(
                f"[Rank {rank}] FSDP communication options: "
                f"sharding={fsdp_options['sharding_strategy'].name}, "
                f"backward_prefetch={getattr(fsdp_options['backward_prefetch'], 'name', 'NONE')}, "
                f"forward_prefetch={fsdp_options['forward_prefetch']}, "
                f"limit_all_gathers={fsdp_options['limit_all_gathers']}"
            )
            auto_wrap_policy = (
                peft_fsdp_auto_wrap_policy(model)
                if tuning == "lora"
                else _full_model_fsdp_auto_wrap_policy(model)
            )
            use_orig_params = tuning == "full"
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=fsdp_options["sharding_strategy"],
                backward_prefetch=fsdp_options["backward_prefetch"],
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                ),
                device_id=torch.device("cuda", local_rank),
                sync_module_states=False,
                forward_prefetch=fsdp_options["forward_prefetch"],
                limit_all_gathers=fsdp_options["limit_all_gathers"],
                use_orig_params=use_orig_params,
            )
            # Transformers' Trainer cannot identify a PEFT model through a
            # manually-created FSDP wrapper. Preserve the marker it uses to
            # distinguish QLoRA from unsupported base-model quantized tuning.
            if tuning == "lora":
                model._hf_peft_config_loaded = True
            print(
                f"[Rank {rank}] Model wrapped with FSDP "
                f"(FULL_SHARD, use_orig_params={use_orig_params}, fp16)"
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            # DDP reducer collectives happen in C++ and do not pass through
            # dist.all_reduce. Register a small Python comm hook so NCCL
            # latency and bucket bytes are included in metrics_detailed.json.
            model.register_comm_hook(step_callback, step_callback.ddp_comm_hook)
            print(f"[Rank {rank}] Model loaded and wrapped with DDP")

        custom_optimizer = None
        if ddp_cpu_offload:
            print(
                f"[Rank {rank}] Creating FP16 CPU-offloaded AdamW baseline..."
            )
            custom_optimizer = CPUOffloadAdamW(
                model.parameters(),
                lr=learning_rate,
            )
            print(f"[Rank {rank}] CPU-offloaded AdamW ready")

        # ========================================================================
        # Dataset preparation for TRL 0.8.1
        #
        # TRL 0.8.1 has two bugs in _prepare_dataset:
        # 1. Does NOT remove original string columns after tokenization
        # 2. Does NOT support skip_prepare_dataset
        #
        # Solution:
        # - Remove non-text columns BEFORE SFTTrainer tokenization
        # - Use remove_unused_columns=False to prevent Trainer from stripping
        #   input_ids/attention_mask (DDP forward signature doesn't list them)
        # - Wrap DataCollator to strip leftover 'text' string field at batch level
        # ========================================================================
        dataset_columns = trainset.column_names
        print(f"[Rank {rank}] Dataset columns: {dataset_columns}")

        # Remove all columns except 'text' to avoid DataCollator issues
        columns_to_remove = [col for col in dataset_columns if col != "text"]
        if columns_to_remove:
            trainset = trainset.remove_columns(columns_to_remove)
            print(f"[Rank {rank}] Removed columns: {columns_to_remove}")
            print(f"[Rank {rank}] Remaining columns: {trainset.column_names}")

        # Split training set for evaluation
        eval_split_ratio = train_config.get("eval_split_ratio", 0.1)
        trainset, evalset = create_eval_split(trainset, split_ratio=eval_split_ratio)
        print(f"[Rank {rank}] Train/Eval split: {len(trainset)}/{len(evalset)} samples")

        # Wrap DataCollator to strip non-tensor fields BEFORE passing to inner collator.
        # TRL 0.8.1's tokenization leaves the original 'text' string column in the dataset.
        # DataCollatorForCompletionOnlyLM tries to pad ALL fields → crash on 'text' strings.
        class StrippingDataCollator:
            def __init__(self, collator):
                self.collator = collator
            def __call__(self, examples):
                # Strip non-tensor fields from each example BEFORE inner collator
                cleaned = []
                for ex in examples:
                    cleaned.append({
                        k: v for k, v in ex.items()
                        if not isinstance(v, str)
                    })
                return self.collator(cleaned)

        wrapped_collator = StrippingDataCollator(data_collator)

        # Initialize the resource monitor after the distributed wrapper exists
        # so each rank can select its own local GPU.
        resource_monitor = ResourceMonitor(interval=1.0, device_index=local_rank)

        # Setup training arguments
        # V100 supports FP16 AMP but not BF16. Full tuning keeps FP32 master
        # parameters while AMP performs the forward/backward compute in FP16;
        # this lets GradScaler operate safely and avoids pure-FP16 NaNs.
        # GradScaler rejects FP16 leaf gradients.  In the DDP CPU-offload
        # baseline the GPU replica is intentionally FP16 to avoid the otherwise
        # unavoidable FP32 model + DDP bucket OOM, and the optimizer owns the
        # CPU-side mirror, so do not ask Trainer to create a GradScaler.
        trainer_amp_fp16 = not ddp_cpu_offload
        amp_description = (
            "disabled (FP16 DDP replica with CPU-offloaded optimizer)"
            if ddp_cpu_offload
            else "enabled (FP32 master parameters, FP16 compute)"
        )
        print(f"[Rank {rank}] Trainer AMP GradScaler: {amp_description}")
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=logging_steps,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            # Distributed ranks share one PVC-backed output_dir. Trainer's
            # automatic checkpoint writer makes every manually launched rank
            # race on the same sharded files. Federated state is saved below
            # exactly once by rank 0, so disable the redundant Trainer save.
            save_strategy="no",
            lr_scheduler_type=lr_scheduler_type,
            fp16=trainer_amp_fp16,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,  # Prevent Trainer from stripping input_ids/attention_mask
        )

        # Construct trainer
        # SFTTrainer tokenizes 'text' → creates input_ids, attention_mask
        # remove_unused_columns=False keeps them (DDP signature doesn't list them)
        # StrippingDataCollator removes leftover 'text' strings at batch level
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_arguments,
            max_seq_length=seq_length,
            train_dataset=trainset,
            data_collator=wrapped_collator,
            dataset_text_field="text",
            callbacks=[step_callback],
            optimizers=(custom_optimizer, None),
        )

        # TRL 0.8.1 does not consistently dispatch on_pre_backward for the
        # manually wrapped DDP/FSDP path. Time compute_loss directly so the
        # forward/loss interval is measured before backward starts instead of
        # silently becoming zero in metrics_detailed.json.
        original_compute_loss = trainer.compute_loss

        def timed_compute_loss(model_instance, inputs, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_started = time.perf_counter()
            try:
                return original_compute_loss(model_instance, inputs, *args, **kwargs)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_callback.record_forward(
                    (time.perf_counter() - forward_started) * 1000
                )

        trainer.compute_loss = timed_compute_loss

        # ========================================================================
        # Distributed data split
        #
        # We manually init the process group and wrap the model with DDP, so HF
        # Trainer's `parallel_mode` stays NO (accelerate PartialState is unaware
        # of the manual DDP). As a result Trainer._get_train_sampler returns a
        # plain RandomSampler and EVERY rank iterates the FULL dataset → 3x
        # redundant compute (1098 steps instead of 366).
        #
        # Fix: override _get_train_sampler to return a DistributedSampler that
        # shards the dataset across world_size ranks. Each rank now sees
        # 23400/3 = 7800 samples, giving 122 steps/epoch (366 total for 3 epochs).
        # set_epoch is called automatically by Trainer's epoch loop and propagated
        # through accelerate's DataLoaderShard → batch_sampler.sampler.set_epoch.
        # ========================================================================
        from torch.utils.data.distributed import DistributedSampler

        def _distributed_train_sampler(dataset=None):
            if dataset is None:
                dataset = trainer.train_dataset
            print(
                f"[Rank {rank}] Using DistributedSampler: "
                f"{len(dataset)} samples / {world_size} ranks = "
                f"{len(dataset) // world_size} per rank"
            )
            return DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )

        trainer._get_train_sampler = _distributed_train_sampler

        print(f"[Rank {rank}] Starting training...")

        # Start resource monitoring
        resource_monitor.start()

        # Train
        results = trainer.train()
        optimizer_steps = int(trainer.state.global_step)
        estimated_sample_presentations = (
            optimizer_steps
            * batch_size
            * gradient_accumulation_steps
            * world_size
        )

        # Stop resource monitoring
        resource_summary = resource_monitor.stop()

        print(f"[Rank {rank}] Training completed. Loss: {results.training_loss}")

        # The DDP offload optimizer owns a full CPU parameter mirror plus AdamW
        # moments.  Keeping those tensors alive while materializing another
        # complete CPU model state exceeds the 40-GiB worker limit after a
        # multi-step run.  Training is finished, so release optimizer-only
        # state before constructing the federated state dict.
        if ddp_cpu_offload:
            if trainer.optimizer is not None:
                trainer.optimizer.zero_grad(set_to_none=True)
            trainer.optimizer = None
            trainer.lr_scheduler = None
            custom_optimizer = None
            accelerator = getattr(trainer, "accelerator", None)
            if accelerator is not None:
                getattr(accelerator, "_optimizers", []).clear()
                getattr(accelerator, "_schedulers", []).clear()
            gc.collect()
            print(
                f"[Rank {rank}] Released CPU-offloaded optimizer state before "
                "full-model export"
            )

        # FSDP state-dict collection is collective: every rank must enter it,
        # even though only rank 0 receives the full CPU state dict.
        unwrapped_model = model.module
        state_export_started = time.perf_counter()
        state_export_summary = {
            "state_export_type": "full_state_dict" if distributed_strategy == "fsdp" else "replicated_full_state_dict",
        }
        if distributed_strategy == "fsdp":
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
            ):
                full_state_dict = model.state_dict()
            state_export_summary["full_state_export_s"] = round(
                time.perf_counter() - state_export_started, 4
            )
            state_conversion_started = time.perf_counter()
            model_state_dict = (
                get_federated_state_dict(
                    unwrapped_model,
                    model_cfg,
                    state_dict=full_state_dict,
                )
                if rank == 0
                else None
            )
            if rank == 0:
                state_export_summary["state_dict_conversion_s"] = round(
                    time.perf_counter() - state_conversion_started, 4
                )
        else:
            state_conversion_started = time.perf_counter()
            model_state_dict = (
                get_federated_state_dict(unwrapped_model, model_cfg)
                if rank == 0
                else None
            )
            if rank == 0:
                state_export_summary["full_state_export_s"] = round(
                    time.perf_counter() - state_export_started, 4
                )
                state_export_summary["state_dict_conversion_s"] = round(
                    time.perf_counter() - state_conversion_started, 4
                )

        # Save results (only on rank 0).
        if rank == 0:
            device = f"cuda:{local_rank}"

            # Create output directory
            job_name = os.environ.get("JOB_NAME", "distributed_job")
            job_output_dir = f"{output_dir}/{job_name}"
            os.makedirs(job_output_dir, exist_ok=True)

            # Save weights
            weights_path = f"{job_output_dir}/model_weights.pt"
            serialization_started = time.perf_counter()
            torch.save(model_state_dict, weights_path)
            state_export_summary["state_serialization_s"] = round(
                time.perf_counter() - serialization_started, 4
            )
            state_export_summary["state_bytes"] = os.path.getsize(weights_path)
            print(f"[Rank {rank}] Saved model weights to {weights_path}")

            # Collect training performance summary
            training_summary = step_callback.get_summary()
            print(f"[Rank {rank}] Training summary: avg_step={training_summary['avg_step_time_ms']}ms, throughput={training_summary['throughput_tokens_per_s']} tokens/s")

            validation_metrics = None
            downstream_metrics = None
            evaluation_total_s = 0.0
            if run_local_evaluation:
                evaluation_start = time.perf_counter()
                # FSDP parameters are sharded after training, so reload a
                # compact unwrapped QLoRA model for rank-0-only validation and
                # generation. This avoids nested FSDP collectives in generate().
                evaluation_model = unwrapped_model
                if distributed_strategy == "fsdp":
                    del full_state_dict
                    del unwrapped_model
                    del trainer
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

                    cleanup_distributed()
                    print(f"[Rank {rank}] Reloading unwrapped model for evaluation")
                    evaluation_model = get_model(model_cfg)
                    set_federated_state_dict(
                        evaluation_model, model_cfg, model_state_dict
                    )
                    evaluation_model = evaluation_model.to(device)

                num_eval_samples = train_config.get("num_eval_samples", 50)
                print(
                    f"[Rank {rank}] Computing validation metrics on "
                    f"{min(len(evalset), num_eval_samples)} samples..."
                )
                validation_metrics = compute_validation_metrics(
                    evaluation_model,
                    tokenizer,
                    evalset,
                    device,
                    max_samples=num_eval_samples,
                )
                print(
                    f"[Rank {rank}] Validation: "
                    f"loss={validation_metrics['val_loss']}, "
                    f"perplexity={validation_metrics['perplexity']}"
                )

                print(f"[Rank {rank}] Computing downstream metrics...")
                downstream_metrics = compute_downstream_metrics(
                    evaluation_model,
                    tokenizer,
                    evalset,
                    device,
                    max_samples=num_eval_samples,
                )
                print(
                    f"[Rank {rank}] Downstream: "
                    f"accuracy={downstream_metrics['accuracy']}, "
                    f"rouge_l={downstream_metrics['rouge_l']}"
                )
                evaluation_total_s = time.perf_counter() - evaluation_start
            else:
                print(
                    f"[Rank {rank}] In-round local evaluation disabled; "
                    "run evaluation separately from the training experiment."
                )

            # Save detailed metrics
            save_metrics_detailed(
                job_output_dir, training_summary, resource_summary,
                validation_metrics=validation_metrics,
                downstream_metrics=downstream_metrics,
                federated_metrics={
                    "training_only_s": training_summary["total_train_time_s"],
                    "evaluation_s": round(evaluation_total_s, 4),
                    "evaluation_enabled": run_local_evaluation,
                    **state_export_summary,
                },
            )

            # Save simple metrics.json for backward compatibility with client_app.py
            metrics = {
                "train_loss": results.training_loss,
                "num_examples": len(trainset),
                "world_size": world_size,
                "distributed_strategy": distributed_strategy,
                "finetuning_type": tuning,
                "federated_state_dtype": (
                    "float16" if tuning == "full" else "adapter_native"
                ),
                "quantization": quantization,
                "gradient_checkpointing": gradient_checkpointing,
                "ddp_cpu_offload": ddp_cpu_offload,
                "cpu_optimizer_state_dtype": (
                    "float16" if ddp_cpu_offload else "not_applicable"
                ),
                "optimizer_steps": optimizer_steps,
                "estimated_sample_presentations": estimated_sample_presentations,
                "training_only_s": training_summary["total_train_time_s"],
                "evaluation_s": round(evaluation_total_s, 4),
                "evaluation_enabled": run_local_evaluation,
                "network_rx_bytes": resource_summary.get("network_rx_bytes"),
                "network_tx_bytes": resource_summary.get("network_tx_bytes"),
                "network_total_bytes": resource_summary.get("network_total_bytes"),
                "nccl_bytes": training_summary.get("total_nccl_bytes"),
                "nccl_comm_ms": training_summary.get("avg_nccl_comm_ms"),
                "nccl_collective_calls": training_summary.get("nccl_collective_calls"),
                **state_export_summary,
                "timestamp": datetime.now().isoformat(),
            }
            metrics_path = f"{job_output_dir}/metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"[Rank {rank}] Saved metrics to {metrics_path}")

        # DDP keeps all ranks alive through rank-0 evaluation. FSDP has already
        # completed its last collective during full-state gathering; nonzero
        # ranks may exit while rank 0 reloads an unwrapped evaluation model.
        if distributed_strategy == "ddp" and dist.is_initialized():
            dist.barrier()
        print(f"[Rank {rank}] Training job completed successfully")

    except Exception as e:
        print(f"[Rank {rank}] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    train_distributed()
