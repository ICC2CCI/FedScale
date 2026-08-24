"""Evaluate an Alpaca SFT PEFT or full-model checkpoint with held-out data.

This script intentionally keeps the two evaluation paths separate:

* teacher forcing over prompt + reference for assistant-only loss/PPL;
* deterministic generation from the prompt for ROUGE-L/BERTScore.

The held-out set is reconstructed in the same order as federated training:
IID partition first, then a seed-42 train/test split inside every partition.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from datasets import concatenate_datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


RESPONSE_MARKER = "### Response:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--adapter-path")
    checkpoint.add_argument(
        "--full-state-path",
        help="Full Hugging Face model state_dict saved by the Flower ServerApp.",
    )
    parser.add_argument(
        "--label",
        help="Stable result label used in summary and prediction filenames.",
    )
    parser.add_argument(
        "--base-model", default="openlm-research/open_llama_3b_v2"
    )
    parser.add_argument("--dataset", default="vicgalle/alpaca-gpt4")
    parser.add_argument("--num-partitions", type=int, default=2)
    parser.add_argument("--eval-split-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bertscore-layers", type=int, default=17)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Do not evaluate the unmodified base-model baseline.",
    )
    return parser.parse_args()


def stable_sample_key(sample: dict) -> str:
    source = f"{sample.get('instruction', '')}\0{sample.get('input', '')}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def reconstruct_held_out_set(args: argparse.Namespace):
    """Recreate the union of the per-client split(seed=42) test subsets."""
    fds = FederatedDataset(
        dataset=args.dataset,
        partitioners={"train": IidPartitioner(num_partitions=args.num_partitions)},
    )
    held_out_partitions = []
    partition_sizes = []
    for partition_id in range(args.num_partitions):
        partition = fds.load_partition(partition_id, "train")
        split = partition.train_test_split(
            test_size=args.eval_split_ratio, seed=args.split_seed
        )
        held_out_partitions.append(split["test"])
        partition_sizes.append(
            {
                "partition_id": partition_id,
                "total": len(partition),
                "train": len(split["train"]),
                "held_out": len(split["test"]),
            }
        )

    held_out = concatenate_datasets(held_out_partitions)
    ordered_indices = sorted(
        range(len(held_out)), key=lambda index: stable_sample_key(held_out[index])
    )
    selected_count = min(args.max_samples, len(ordered_indices))
    selected = held_out.select(ordered_indices[:selected_count])
    return selected, len(held_out), partition_sizes


def prompt_and_reference(sample: dict) -> tuple[str, str, str]:
    """Use the exact dataset text/template seen by SFT training."""
    full_text = sample["text"]
    reference = sample["output"]
    if reference and full_text.endswith(reference):
        answer_start = len(full_text) - len(reference)
        return full_text[:answer_start], reference, full_text

    marker_index = full_text.find(RESPONSE_MARKER)
    if marker_index < 0:
        raise ValueError("Sample text has no Alpaca response marker")
    answer_start = marker_index + len(RESPONSE_MARKER)
    while answer_start < len(full_text) and full_text[answer_start].isspace():
        answer_start += 1
    return full_text[:answer_start], full_text[answer_start:], full_text


def load_model(
    base_model: str,
    adapter_path: str | None,
    full_state_path: str | None,
    local_files_only: bool,
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            local_files_only=local_files_only,
        )
    elif full_state_path is not None:
        print(f"Loading full state dict: {full_state_path}", flush=True)
        state_dict = torch.load(
            full_state_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state_dict, dict):
            raise TypeError("Full checkpoint must contain a state_dict mapping")
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


def assistant_only_nll(
    model,
    tokenizer,
    samples,
    max_length: int,
) -> dict:
    """Token-weighted NLL over reference/output tokens only."""
    total_nll = 0.0
    total_tokens = 0
    evaluated_samples = 0
    skipped_samples = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for sample in samples:
            prompt, _, full_text = prompt_and_reference(sample)
            encoded = tokenizer(
                full_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            offsets = encoded.pop("offset_mapping")[0]
            input_ids = encoded["input_ids"]
            labels = input_ids.clone()
            answer_start = len(prompt)
            for token_index, (start, end) in enumerate(offsets.tolist()):
                if start < answer_start or start == end:
                    labels[0, token_index] = -100

            shifted_labels = labels[:, 1:]
            answer_tokens = int((shifted_labels != -100).sum().item())
            if answer_tokens == 0:
                skipped_samples += 1
                continue

            device_batch = {
                key: value.to(model.device) for key, value in encoded.items()
            }
            logits = model(**device_batch, use_cache=False).logits[:, :-1, :]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                shifted_labels.to(logits.device).reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += answer_tokens
            evaluated_samples += 1

    loss = total_nll / total_tokens if total_tokens else math.inf
    return {
        "loss": loss,
        "ppl": math.exp(loss) if loss < 50 else math.inf,
        "answer_tokens": total_tokens,
        "evaluated_samples": evaluated_samples,
        "skipped_samples": skipped_samples,
        "elapsed_seconds": time.perf_counter() - started,
    }


def generate_predictions(
    model,
    tokenizer,
    samples,
    max_length: int,
    max_new_tokens: int,
) -> tuple[list[dict], float]:
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            prompt, reference, _ = prompt_and_reference(sample)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            new_ids = output_ids[0, encoded["input_ids"].shape[1] :]
            prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            records.append(
                {
                    "sample_index": index,
                    "sample_hash": stable_sample_key(sample),
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "reference": reference,
                    "prediction": prediction,
                    "generated_tokens": int(new_ids.shape[0]),
                }
            )
            print(
                f"generated {index + 1}/{len(samples)} "
                f"({int(new_ids.shape[0])} tokens)",
                flush=True,
            )
    return records, time.perf_counter() - started


def rouge_l(records: list[dict]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(record["reference"], record["prediction"])["rougeL"]
        for record in records
    ]
    return {
        "precision": sum(score.precision for score in scores) / len(scores),
        "recall": sum(score.recall for score in scores) / len(scores),
        "f1": sum(score.fmeasure for score in scores) / len(scores),
    }


def generation_quality(records: list[dict]) -> dict:
    """Return deterministic, dependency-free generation diagnostics."""
    empty_predictions = sum(not record["prediction"].strip() for record in records)
    exact_matches = sum(
        record["prediction"].strip() == record["reference"].strip()
        for record in records
    )
    generated_tokens = [record["generated_tokens"] for record in records]
    count = len(records)
    return {
        "exact_match": exact_matches / count if count else 0.0,
        "empty_predictions": empty_predictions,
        "empty_prediction_rate": empty_predictions / count if count else 0.0,
        "average_generated_tokens": (
            sum(generated_tokens) / count if count else 0.0
        ),
    }


def add_bertscore(
    evaluations: list[dict],
    model_type: str,
    num_layers: int,
    batch_size: int,
) -> None:
    from bert_score import score as bert_score

    candidates = []
    references = []
    spans = []
    for evaluation in evaluations:
        start = len(candidates)
        candidates.extend(record["prediction"] for record in evaluation["predictions"])
        references.extend(record["reference"] for record in evaluation["predictions"])
        spans.append((start, len(candidates)))

    precision, recall, f1 = bert_score(
        candidates,
        references,
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        device="cuda",
        idf=False,
        rescale_with_baseline=False,
        verbose=True,
    )
    for evaluation, (start, end) in zip(evaluations, spans):
        evaluation["metrics"]["bertscore"] = {
            "precision": float(precision[start:end].mean().item()),
            "recall": float(recall[start:end].mean().item()),
            "f1": float(f1[start:end].mean().item()),
            "model_type": model_type,
            "num_layers": num_layers,
            "idf": False,
            "rescale_with_baseline": False,
        }


def evaluate_one(
    label: str,
    adapter_path: str | None,
    full_state_path: str | None,
    args: argparse.Namespace,
    tokenizer,
    samples,
) -> dict:
    print(f"\nLoading {label}", flush=True)
    model = load_model(
        args.base_model,
        adapter_path,
        full_state_path,
        args.local_files_only,
    )
    teacher_forced = assistant_only_nll(
        model, tokenizer, samples, args.max_length
    )
    predictions, generation_seconds = generate_predictions(
        model,
        tokenizer,
        samples,
        args.max_length,
        args.max_new_tokens,
    )
    metrics = {
        "assistant_only": teacher_forced,
        "rouge_l": rouge_l(predictions),
        "generation_quality": generation_quality(predictions),
        "generation_seconds": generation_seconds,
        "generation_samples": len(predictions),
        "generation_tokens": sum(item["generated_tokens"] for item in predictions),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "label": label,
        "adapter_path": adapter_path,
        "full_state_path": full_state_path,
        "metrics": metrics,
        "predictions": predictions,
    }


def save_intermediate(output_dir: Path, evaluations: list[dict]) -> None:
    """Persist completed inference before the separately fallible BERTScore step."""
    partial_results = []
    for evaluation in evaluations:
        partial_results.append(
            {
                "label": evaluation["label"],
                "adapter_path": evaluation["adapter_path"],
                "full_state_path": evaluation["full_state_path"],
                "metrics": evaluation["metrics"],
            }
        )
        prediction_path = output_dir / f"predictions_{evaluation['label']}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for record in evaluation["predictions"]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "partial_results.json").open("w", encoding="utf-8") as handle:
        json.dump({"results": partial_results}, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation job requires a CUDA GPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, held_out_size, partition_sizes = reconstruct_held_out_set(args)
    print(
        f"Held-out union: {held_out_size}; selected: {len(samples)}; "
        f"partition details: {partition_sizes}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        padding_side="right",
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    evaluations = []
    if not args.skip_base:
        evaluations.append(
            evaluate_one("base", None, None, args, tokenizer, samples)
        )
        save_intermediate(output_dir, evaluations)
    label = args.label or (
        "federated_peft_50" if args.adapter_path else "full_checkpoint"
    )
    evaluations.append(
        evaluate_one(
            label,
            args.adapter_path,
            args.full_state_path,
            args,
            tokenizer,
            samples,
        )
    )
    save_intermediate(output_dir, evaluations)

    add_bertscore(
        evaluations,
        args.bertscore_model,
        args.bertscore_layers,
        args.bertscore_batch_size,
    )

    metadata = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "full_state_path": args.full_state_path,
        "dataset": args.dataset,
        "partition_then_split": True,
        "num_partitions": args.num_partitions,
        "eval_split_ratio": args.eval_split_ratio,
        "split_seed": args.split_seed,
        "held_out_union_samples": held_out_size,
        "selected_samples": len(samples),
        "selection": "ascending sha256(instruction + NUL + input)",
        "partition_sizes": partition_sizes,
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
        "results": [
            {
                "label": evaluation["label"],
                "adapter_path": evaluation["adapter_path"],
                "full_state_path": evaluation["full_state_path"],
                "metrics": evaluation["metrics"],
            }
            for evaluation in evaluations
        ],
    }
    if len(evaluations) == 2:
        base = evaluations[0]["metrics"]
        tuned = evaluations[1]["metrics"]
        summary["delta_federated_minus_base"] = {
            "loss": tuned["assistant_only"]["loss"]
            - base["assistant_only"]["loss"],
            "ppl": tuned["assistant_only"]["ppl"]
            - base["assistant_only"]["ppl"],
            "rouge_l_f1": tuned["rouge_l"]["f1"] - base["rouge_l"]["f1"],
            "bertscore_f1": tuned["bertscore"]["f1"]
            - base["bertscore"]["f1"],
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    for evaluation in evaluations:
        prediction_path = output_dir / f"predictions_{evaluation['label']}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for record in evaluation["predictions"]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
