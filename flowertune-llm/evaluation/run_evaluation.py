"""CLI entry point for standalone evaluation outside the federated runtime.

Usage examples:

    # Evaluate a PEFT adapter checkpoint against the base model baseline
    python -m evaluation.run_evaluation \
        --adapter-path /path/to/checkpoint \
        --base-model openlm-research/open_llama_3b_v2 \
        --dataset vicgalle/alpaca-gpt4 \
        --output-dir ./eval_results \
        --max-samples 100 \
        --with-baseline

    # Evaluate a full-model state dict
    python -m evaluation.run_evaluation \
        --full-state-path /path/to/model_state.pt \
        --base-model openlm-research/open_llama_3b_v2 \
        --output-dir ./eval_results

    # Include MMLU evaluation
    python -m evaluation.run_evaluation \
        --adapter-path /path/to/checkpoint \
        --base-model openlm-research/open_llama_3b_v2 \
        --output-dir ./eval_results \
        --mmlu-dataset cais/mmlu \
        --mmlu-samples 200

    # Generate compression report
    python -m evaluation.run_evaluation \
        --adapter-path /path/to/checkpoint \
        --base-model openlm-research/open_llama_3b_v2 \
        --output-dir ./eval_results \
        --compression-report

    # Generate DP report
    python -m evaluation.run_evaluation \
        --adapter-path /path/to/checkpoint \
        --base-model openlm-research/open_llama_3b_v2 \
        --output-dir ./eval_results \
        --dp-sigma-client 0.5 \
        --dp-num-rounds 50 \
        --dp-clip-norm 1.0 \
        --dp-cohort-size 2
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from datasets import concatenate_datasets
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone evaluation for FedScale federated LLM fine-tuning."
    )

    # Model loading (mutually exclusive: adapter vs full state)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--adapter-path", help="PEFT adapter checkpoint directory")
    checkpoint.add_argument("--full-state-path", help="Full model state_dict file")

    # Model configuration
    parser.add_argument("--base-model", default="openlm-research/open_llama_3b_v2")
    parser.add_argument("--label", help="Stable label for result filenames")
    parser.add_argument("--quantization", type=int, default=0, choices=[0, 4, 8])
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)

    # Dataset configuration
    parser.add_argument("--dataset", default="vicgalle/alpaca-gpt4")
    parser.add_argument("--num-partitions", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)

    # Evaluation parameters
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bertscore-model", default=None)

    # Baseline comparison
    parser.add_argument("--with-baseline", action="store_true",
                        help="Evaluate the unmodified base model as a baseline")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model evaluation (override --with-baseline)")

    # MMLU evaluation
    parser.add_argument("--mmlu-dataset", default=None,
                        help="MMLU dataset name (e.g. cais/mmlu or TIGER-Lab/MMLU-Pro)")
    parser.add_argument("--mmlu-samples", type=int, default=100)

    # Compression report
    parser.add_argument("--compression-report", action="store_true")
    parser.add_argument("--block-size", type=int, default=1048576)

    # DP report
    parser.add_argument("--dp-sigma-client", type=float, default=None)
    parser.add_argument("--dp-num-rounds", type=int, default=50)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-cohort-size", type=int, default=2)
    parser.add_argument("--dp-delta", type=float, default=1e-5)

    # Output
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")

    return parser.parse_args()


def load_model(
    base_model: str,
    adapter_path: str | None,
    full_state_path: str | None,
    quantization: int,
    lora_r: int,
    lora_alpha: int,
    local_files_only: bool,
):
    """Load the evaluation model (PEFT adapter or full state dict)."""
    from transformers import BitsAndBytesConfig

    quantization_config = None
    if quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=False,
            local_files_only=local_files_only,
        )
    elif full_state_path is not None:
        print(f"Loading full state dict: {full_state_path}", flush=True)
        state_dict = torch.load(
            full_state_path, map_location="cpu", weights_only=True,
        )
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Full-model state mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del state_dict
        gc.collect()

    model.eval()
    model.config.use_cache = True
    return model


def reconstruct_held_out_test(args: argparse.Namespace):
    """Reconstruct the union of per-partition held-out test subsets."""
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import IidPartitioner
    from evaluation.data_split import create_train_val_test_split, stable_sample_key

    fds = FederatedDataset(
        dataset=args.dataset,
        partitioners={"train": IidPartitioner(num_partitions=args.num_partitions)},
    )

    test_partitions = []
    partition_sizes = []
    for partition_id in range(args.num_partitions):
        partition = fds.load_partition(partition_id, "train")
        split = create_train_val_test_split(
            partition,
            val_ratio=0.0,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
        test_partitions.append(split.test)
        partition_sizes.append({
            "partition_id": partition_id,
            "total": len(partition),
            "test": len(split.test),
        })

    held_out = concatenate_datasets(test_partitions)
    ordered_indices = sorted(
        range(len(held_out)), key=lambda i: stable_sample_key(held_out[i])
    )
    selected_count = min(args.max_samples, len(ordered_indices))
    selected = held_out.select(ordered_indices[:selected_count])

    return selected, len(held_out), partition_sizes


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation job requires a CUDA GPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or (
        "federated_peft" if args.adapter_path else "full_checkpoint"
    )

    # --- Reconstruct test set ---
    samples, held_out_size, partition_sizes = reconstruct_held_out_test(args)
    print(
        f"Held-out test union: {held_out_size}; selected: {len(samples)}; "
        f"partition details: {partition_sizes}",
        flush=True,
    )

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        padding_side="left",
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load federated model ---
    print(f"\nLoading {label}", flush=True)
    model = load_model(
        args.base_model,
        args.adapter_path,
        args.full_state_path,
        args.quantization,
        args.lora_r,
        args.lora_alpha,
        args.local_files_only,
    )

    # --- Evaluate federated model ---
    from evaluation.validation_metrics import compute_validation_metrics
    from evaluation.generation_metrics import compute_generation_metrics

    fed_started = time.perf_counter()
    fed_val = compute_validation_metrics(
        model, tokenizer, samples, "cuda:0",
        max_samples=args.max_samples,
        max_length=args.max_length,
    )
    fed_gen = compute_generation_metrics(
        model, tokenizer, samples, "cuda:0",
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        bertscore_model=args.bertscore_model,
        return_predictions=True,
    )
    fed_seconds = time.perf_counter() - fed_started

    # Save predictions
    pred_path = output_dir / f"predictions_{label}.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for record in fed_gen.pop("predictions", []):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    results = {
        "label": label,
        "adapter_path": args.adapter_path,
        "full_state_path": args.full_state_path,
        "federated": {
            "validation": fed_val,
            "generation": fed_gen,
            "evaluation_seconds": round(fed_seconds, 4),
        },
    }

    # --- Base model baseline ---
    skip_base = args.skip_base or not args.with_baseline
    if not skip_base:
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print("\nLoading base model", flush=True)
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        )
        base_model_obj.eval()
        base_model_obj.config.use_cache = True

        base_started = time.perf_counter()
        base_val = compute_validation_metrics(
            base_model_obj, tokenizer, samples, "cuda:0",
            max_samples=args.max_samples,
            max_length=args.max_length,
        )
        base_gen = compute_generation_metrics(
            base_model_obj, tokenizer, samples, "cuda:0",
            max_samples=args.max_samples,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            bertscore_model=args.bertscore_model,
        )
        base_seconds = time.perf_counter() - base_started

        results["base"] = {
            "validation": base_val,
            "generation": base_gen,
            "evaluation_seconds": round(base_seconds, 4),
        }

        # Save base predictions
        base_pred_path = output_dir / "predictions_base.jsonl"
        with base_pred_path.open("w", encoding="utf-8") as handle:
            for record in base_gen.get("predictions", []):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        from evaluation.baseline_comparison import compute_delta
        results["delta"] = compute_delta(results["base"], results["federated"])

        del base_model_obj
        gc.collect()
        torch.cuda.empty_cache()

    # --- MMLU evaluation ---
    if args.mmlu_dataset:
        from evaluation.mmlu_evaluator import evaluate_mmlu

        # Reload model for MMLU if we freed it.
        if skip_base:
            pass  # model is still loaded
        else:
            model = load_model(
                args.base_model,
                args.adapter_path,
                args.full_state_path,
                args.quantization,
                args.lora_r,
                args.lora_alpha,
                args.local_files_only,
            )

        is_pro = "mmlu-pro" in args.mmlu_dataset.lower()
        try:
            results["mmlu"] = evaluate_mmlu(
                model=model,
                tokenizer=tokenizer,
                device="cuda:0",
                dataset_name=args.mmlu_dataset,
                max_samples=args.mmlu_samples,
                is_mmlu_pro=is_pro,
            )
        except Exception as exc:
            results["mmlu"] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- Compression report ---
    if args.compression_report:
        from evaluation.compression_report import compute_compression_report as _ccr

        if args.adapter_path:
            from peft import get_peft_model_state_dict
            fed_state = get_peft_model_state_dict(model)
        else:
            fed_state = model.state_dict()

        results["compression"] = [
            {
                "config_name": r.config_name,
                "original_bytes": r.original_bytes,
                "compressed_bytes": r.compressed_bytes,
                "compression_ratio": r.compression_ratio,
                "quantization_error_l2": r.quantization_error_l2,
                "quantization_error_relative": r.quantization_error_relative,
                "mask_ratio": r.mask_ratio,
                "int8_enabled": r.int8_enabled,
                "residual_enabled": r.residual_enabled,
                "extra": r.extra,
            }
            for r in _ccr(fed_state, block_size=args.block_size)
        ]

    # --- DP report ---
    if args.dp_sigma_client is not None and args.dp_sigma_client > 0:
        from evaluation.dp_report import compute_rdp_epsilon

        rdp = compute_rdp_epsilon(
            num_rounds=args.dp_num_rounds,
            clip_norm=args.dp_clip_norm,
            sigma_client=args.dp_sigma_client,
            cohort_size=args.dp_cohort_size,
            delta=args.dp_delta,
        )
        results["dp"] = {
            "epsilon": rdp.epsilon,
            "delta": rdp.delta,
            "alpha_optimal": rdp.alpha_optimal,
            "rdp_per_round": rdp.rdp_per_round,
            "clip_norm": rdp.clip_norm,
            "sigma_client": rdp.sigma_client,
            "min_cohort_size": rdp.min_cohort_size,
            "parameters": rdp.parameters,
        }

    # --- Save summary ---
    metadata = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "full_state_path": args.full_state_path,
        "dataset": args.dataset,
        "num_partitions": args.num_partitions,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "held_out_union_samples": held_out_size,
        "selected_samples": len(samples),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
    }

    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "results": results,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nSaved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
