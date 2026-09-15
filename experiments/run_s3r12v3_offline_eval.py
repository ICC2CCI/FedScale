"""Standalone offline evaluation for S3R12v3 full-model checkpoints.

Downloads the final global model from MinIO, loads it on a single GPU,
and computes:
  - Validation metrics (teacher-forced loss, perplexity)
  - Generation metrics (ROUGE-L, BERTScore, token-overlap accuracy, Macro-F1)
  - Resource usage during evaluation (GPU memory, utilization)

Usage (on a GPU node, e.g. ICC1)::

    cd ~/fedscale-eval
    source ~/miniconda3/bin/activate flwr-ft
    export PYTHONPATH=experiments:flowertune-llm:$PYTHONPATH
    python experiments/run_s3r12v3_offline_eval.py \
        --model-path model/Qwen/Qwen2.5-0.5B \
        --eval-data data/medical_flashcards_eval.json \
        --minio-endpoint http://192.168.235.42:9000 \
        --minio-access-key fedscale \
        --minio-secret-key fedscale-minio-2026 \
        --minio-bucket fedscale-bucket \
        --round 5 \
        --output-dir results/offline_eval \
        --max-samples 50
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "flowertune-llm"))

from shared.minio_client import MinIOClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_dataset(eval_path: str) -> list[dict]:
    """Load evaluation dataset from JSON file."""
    path = Path(eval_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d eval samples from %s", len(data), path)
    return data


def format_sample(sample: dict, tokenizer) -> str:
    """Format an Alpaca-style sample into model text."""
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")

    if inp:
        prompt = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    return prompt + output


# ---------------------------------------------------------------------------
# Validation metrics (Path A: teacher-forced)
# ---------------------------------------------------------------------------

RESPONSE_MARKERS = ("### Response:", "<|im_start|>assistant\n")


def _split_prompt_reference(text: str) -> tuple[str, str]:
    for marker in RESPONSE_MARKERS:
        if marker in text:
            idx = text.index(marker) + len(marker)
            while idx < len(text) and text[idx].isspace():
                idx += 1
            return text[:idx], text[idx:]
    return text, ""


def compute_validation_metrics(
    model, tokenizer, evalset: list[str], device: str,
    max_samples: int = 50, max_length: int = 512,
) -> dict:
    """Compute assistant-only validation loss and perplexity."""
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    evaluated = 0
    skipped = 0

    with torch.inference_mode():
        for i in range(min(len(evalset), max_samples)):
            text = evalset[i]
            prompt, reference = _split_prompt_reference(text)
            if not reference:
                skipped += 1
                continue

            encodings = tokenizer(
                text, return_tensors="pt", return_offsets_mapping=True,
                truncation=True, max_length=max_length, add_special_tokens=True,
            )
            offsets = encodings.pop("offset_mapping")[0]
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            if input_ids.shape[1] < 2:
                skipped += 1
                continue

            labels = input_ids.clone()
            answer_start = len(prompt)
            for tok_idx, (start, end) in enumerate(offsets.tolist()):
                if start < answer_start or start == end:
                    labels[0, tok_idx] = -100

            shifted_labels = labels[:, 1:]
            answer_tokens = int((shifted_labels != -100).sum().item())
            if answer_tokens == 0:
                skipped += 1
                continue

            device_batch = {k: v.to(device) for k, v in encodings.items()}
            logits = model(**device_batch, use_cache=False).logits[:, :-1, :]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shifted_labels.to(logits.device).reshape(-1),
                ignore_index=-100, reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += answer_tokens
            evaluated += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        "loss_scope": "assistant_response_only",
        "evaluated_samples": evaluated,
        "skipped_samples": skipped,
        "answer_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Generation metrics (Path B: free generation)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def generate_predictions(
    model, tokenizer, evalset: list[str], device: str,
    max_samples: int = 50, max_length: int = 512, max_new_tokens: int = 128,
) -> tuple[list[dict], float]:
    model.eval()
    records: list[dict] = []
    started = time.perf_counter()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        with torch.inference_mode():
            for i in range(min(len(evalset), max_samples)):
                text = evalset[i]
                prompt, reference = _split_prompt_reference(text)
                if not reference:
                    continue

                encodings = tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=max_length, add_special_tokens=True,
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)

                output_ids = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id, use_cache=True,
                )
                new_ids = output_ids[0, input_ids.shape[1]:]
                prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                records.append({
                    "sample_index": i,
                    "reference": reference,
                    "prediction": prediction,
                    "generated_tokens": int(new_ids.shape[0]),
                })
    finally:
        tokenizer.padding_side = original_padding_side
    return records, time.perf_counter() - started


def _rouge_l_score(records: list[dict]) -> dict:
    if not records:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r["reference"], r["prediction"])["rougeL"] for r in records]
        return {
            "precision": round(sum(s.precision for s in scores) / len(scores), 4),
            "recall": round(sum(s.recall for s in scores) / len(scores), 4),
            "f1": round(sum(s.fmeasure for s in scores) / len(scores), 4),
        }
    except ImportError:
        # Fallback: LCS-based ROUGE-L
        def _lcs(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    dp[i][j] = dp[i-1][j-1]+1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        f1s = []
        for r in records:
            g, ref = _tokenize(r["prediction"]), _tokenize(r["reference"])
            if not g or not ref:
                f1s.append(0.0); continue
            l = _lcs(g, ref)
            p, rec = l/len(g), l/len(ref)
            f1s.append(2*p*rec/(p+rec) if (p+rec) > 0 else 0.0)
        avg = round(sum(f1s)/len(f1s), 4)
        return {"precision": avg, "recall": avg, "f1": avg}


def _bertscore_f1(records: list[dict]) -> dict:
    if not records:
        return {"f1": None, "precision": None, "recall": None}
    candidates = [r["prediction"] for r in records]
    references = [r["reference"] for r in records]
    try:
        from bert_score import score as bertscore_fn
        P, R, F1 = bertscore_fn(candidates, references, lang="en",
                                device="cuda" if torch.cuda.is_available() else "cpu",
                                verbose=False)
        return {
            "f1": round(float(F1.mean().item()), 4),
            "precision": round(float(P.mean().item()), 4),
            "recall": round(float(R.mean().item()), 4),
        }
    except ImportError:
        return {"f1": None, "precision": None, "recall": None,
                "error": "bert_score library not installed"}


def compute_generation_metrics(
    model, tokenizer, evalset: list[str], device: str,
    max_samples: int = 50, max_length: int = 512, max_new_tokens: int = 128,
) -> dict:
    records, gen_seconds = generate_predictions(
        model, tokenizer, evalset, device,
        max_samples=max_samples, max_length=max_length, max_new_tokens=max_new_tokens,
    )
    if not records:
        return {
            "rouge_l": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "bertscore": {"f1": None, "precision": None, "recall": None},
            "token_overlap_accuracy": 0.0, "macro_f1": 0.0, "exact_match": 0.0,
            "num_eval_samples": 0, "generation_seconds": round(gen_seconds, 4),
        }

    rouge = _rouge_l_score(records)
    bertscore = _bertscore_f1(records)

    # Token overlap accuracy
    correct = 0
    for r in records:
        ref_t = set(_tokenize(r["reference"]))
        gen_t = set(_tokenize(r["prediction"]))
        if ref_t and len(ref_t & gen_t) / len(ref_t) >= 0.3:
            correct += 1
    accuracy = round(correct / len(records), 4)

    # Macro F1
    f1_scores = []
    for r in records:
        g, ref = _tokenize(r["prediction"]), _tokenize(r["reference"])
        if not g or not ref:
            f1_scores.append(0.0); continue
        gc2, rc2 = Counter(g), Counter(ref)
        common = gc2 & rc2
        num_same = sum(common.values())
        if num_same == 0:
            f1_scores.append(0.0); continue
        p, rec = num_same / len(g), num_same / len(ref)
        f1_scores.append(2*p*rec/(p+rec))
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 4)

    # Exact match
    exact_match = round(sum(1 for r in records if r["prediction"].strip() == r["reference"].strip()) / len(records), 4)

    # Generation quality
    empty = sum(1 for r in records if not r["prediction"].strip())
    avg_tokens = sum(r["generated_tokens"] for r in records) / len(records)

    return {
        "rouge_l": rouge,
        "bertscore": bertscore,
        "token_overlap_accuracy": accuracy,
        "macro_f1": macro_f1,
        "exact_match": exact_match,
        "num_eval_samples": len(records),
        "generation_seconds": round(gen_seconds, 4),
        "generation_quality": {
            "exact_match": exact_match,
            "empty_predictions": empty,
            "empty_prediction_rate": round(empty / len(records), 4),
            "average_generated_tokens": round(avg_tokens, 2),
        },
        "predictions": records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="S3R12v3 offline generative evaluation")
    parser.add_argument("--model-path", required=True, help="Base model path (local)")
    parser.add_argument("--eval-data", required=True, help="Eval data JSON file")
    parser.add_argument("--minio-endpoint", required=True)
    parser.add_argument("--minio-access-key", required=True)
    parser.add_argument("--minio-secret-key", required=True)
    parser.add_argument("--minio-bucket", default="fedscale-bucket")
    parser.add_argument("--round", type=int, required=True, help="Round number of the global model to evaluate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Offline evaluation requires CUDA GPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # --- Download global model from MinIO ---
    mc = MinIOClient(
        args.minio_endpoint, args.minio_access_key,
        args.minio_secret_key, args.minio_bucket,
    )
    state_key = f"global_state/round-{args.round}/state.pt"
    local_state = output_dir / f"global_state_round_{args.round}.pt"
    if not local_state.exists():
        logger.info("Downloading %s from MinIO...", state_key)
        state_bytes = mc.get_bytes(state_key)
        local_state.write_bytes(state_bytes)
        logger.info("Downloaded %d bytes", len(state_bytes))
    else:
        logger.info("Using cached %s", local_state)

    # --- Load model ---
    logger.info("Loading base model from %s", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map={"": device}, low_cpu_mem_usage=True,
        trust_remote_code=False, attn_implementation="eager",
    )

    logger.info("Loading state dict from %s", local_state)
    state_dict = torch.load(local_state, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    model.eval()
    model.config.use_cache = True

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left",
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load eval data ---
    raw_data = load_eval_dataset(args.eval_data)
    evalset = [format_sample(s, tokenizer) for s in raw_data]

    # --- Track GPU memory before eval ---
    torch.cuda.reset_peak_memory_stats()

    # --- Run validation metrics (PPL, val_loss) ---
    logger.info("Computing validation metrics (PPL, val_loss)...")
    val_started = time.perf_counter()
    val_metrics = compute_validation_metrics(
        model, tokenizer, evalset, device,
        max_samples=args.max_samples, max_length=args.max_length,
    )
    val_seconds = time.perf_counter() - val_started
    logger.info("Validation: val_loss=%.4f, ppl=%.4f, samples=%d",
                val_metrics["val_loss"], val_metrics["perplexity"],
                val_metrics["evaluated_samples"])

    # --- Run generation metrics (ROUGE-L, BERTScore, etc.) ---
    logger.info("Computing generation metrics (ROUGE-L, BERTScore)...")
    gen_started = time.perf_counter()
    gen_metrics = compute_generation_metrics(
        model, tokenizer, evalset, device,
        max_samples=args.max_samples, max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    gen_seconds = time.perf_counter() - gen_started
    logger.info("Generation: rouge_l_f1=%s, bertscore_f1=%s, samples=%d",
                gen_metrics["rouge_l"]["f1"],
                gen_metrics["bertscore"]["f1"],
                gen_metrics["num_eval_samples"])

    # --- Resource usage during eval ---
    gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    gpu_util_pct = None
    try:
        gpu_util_pct = round(torch.cuda.utilization(), 2)
    except Exception:
        pass
    cpu_mem_mb = 0.0
    try:
        import resource as _r
        cpu_mem_mb = round(_r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        pass

    # --- Save predictions ---
    predictions = gen_metrics.pop("predictions", [])
    pred_path = output_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d predictions to %s", len(predictions), pred_path)

    # --- Build summary ---
    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "round": args.round,
        "model_path": args.model_path,
        "eval_data": args.eval_data,
        "max_samples": args.max_samples,
        "device": torch.cuda.get_device_name(0),
        "validation": val_metrics,
        "validation_seconds": round(val_seconds, 4),
        "generation": gen_metrics,
        "generation_seconds": round(gen_seconds, 4),
        "resource_usage": {
            "gpu_memory_peak_mb": round(gpu_mem_peak_mb, 2),
            "gpu_utilization_pct": gpu_util_pct,
            "cpu_memory_mb": cpu_mem_mb,
        },
    }

    summary_path = output_dir / "offline_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation complete!")
    logger.info("  val_loss=%.4f, perplexity=%.4f", val_metrics["val_loss"], val_metrics["perplexity"])
    logger.info("  rouge_l_f1=%s, bertscore_f1=%s", gen_metrics["rouge_l"]["f1"], gen_metrics["bertscore"]["f1"])
    logger.info("  Report saved to %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
