"""Free-generation evaluation metrics: ROUGE-L, BERTScore, Accuracy, F1, EM (Path B).

Implements the evaluation principles document §7-9:

    "只输入 prompt；模型自由生成 prediction；将 prediction 与 reference 比较。"

Key design decisions vs the legacy ``evaluator.py``:

- **Left padding** for decoder-only model generation (P1 fix): right padding
  can bleed padding tokens into the generated sequence on some tokenizers.
- **BERTScore returns ``None``** on ImportError (P0 fix): returning 0.0
  silently drags down federated averages and misleads the dashboard.
- **Accurate prompt extraction** using the same offset-mapping technique as
  validation_metrics.py.
- **Token-level metrics** use a lightweight regex tokenizer instead of
  ``str.split()`` to better handle punctuation.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

import torch

from evaluation.validation_metrics import _split_prompt_reference


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_predictions(
    model,
    tokenizer,
    evalset,
    device: str,
    max_samples: int = 50,
    max_length: int = 512,
    max_new_tokens: int = 128,
) -> tuple[list[dict], float]:
    """Generate responses for Alpaca-style prompts.

    Uses deterministic decoding (``do_sample=False, num_beams=1``) per the
    evaluation principles document §13.

    **Left padding** (P1 fix): the tokenizer's ``padding_side`` is set to
    ``"left"`` for the duration of generation so that padding tokens do not
    appear between the prompt and the generated continuation.

    Args:
        model: Unwrapped model in eval mode.
        tokenizer: HuggingFace tokenizer.
        evalset: Dataset with a ``text`` column.
        device: CUDA device string.
        max_samples: Maximum samples to evaluate.
        max_length: Maximum prompt tokenization length.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        ``(records, generation_seconds)``
    """
    model.eval()
    records: list[dict] = []
    started = time.perf_counter()

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        with torch.inference_mode():
            for i in range(min(len(evalset), max_samples)):
                sample = evalset[i]
                text = sample["text"]
                prompt, reference = _split_prompt_reference(text)

                if not reference:
                    continue

                encodings = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)

                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

                new_ids = output_ids[0, input_ids.shape[1]:]
                prediction = tokenizer.decode(
                    new_ids, skip_special_tokens=True
                ).strip()

                records.append(
                    {
                        "sample_index": i,
                        "instruction": sample.get("instruction", ""),
                        "input": sample.get("input", ""),
                        "reference": reference,
                        "prediction": prediction,
                        "generated_tokens": int(new_ids.shape[0]),
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side

    return records, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lightweight word tokenizer that strips punctuation."""
    return re.findall(r"\b\w+\b", text.lower())


def _exact_match_score(records: list[dict]) -> float:
    """Fraction of predictions that exactly match references after stripping."""
    if not records:
        return 0.0
    matches = sum(
        1 for r in records
        if r["prediction"].strip() == r["reference"].strip()
    )
    return matches / len(records)


def _accuracy_score(records: list[dict]) -> float:
    """Token-overlap accuracy: ≥30% of reference tokens appear in prediction.

    This is a loose accuracy proxy, not strict QA accuracy.  The evaluation
    principles document §18 notes that ROUGE-L/BERTScore should not be
    conflated with accuracy; this metric is reported separately and labelled
    as ``token_overlap_accuracy`` to avoid confusion.
    """
    if not records:
        return 0.0
    correct = 0
    for r in records:
        ref_tokens = set(_tokenize(r["reference"]))
        gen_tokens = set(_tokenize(r["prediction"]))
        if ref_tokens and len(ref_tokens & gen_tokens) / len(ref_tokens) >= 0.3:
            correct += 1
    return correct / len(records)


def _macro_f1_score(records: list[dict]) -> float:
    """Token-level macro F1 across all samples using regex tokenization."""
    if not records:
        return 0.0
    f1_scores: list[float] = []
    for r in records:
        gen_tokens = _tokenize(r["prediction"])
        ref_tokens = _tokenize(r["reference"])
        if not gen_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        gen_counter = Counter(gen_tokens)
        ref_counter = Counter(ref_tokens)
        common = gen_counter & ref_counter
        num_same = sum(common.values())
        if num_same == 0:
            f1_scores.append(0.0)
            continue
        precision = num_same / len(gen_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def _rouge_l_score(records: list[dict]) -> dict:
    """ROUGE-L F1 using the ``rouge_score`` library with LCS fallback."""
    if not records:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(r["reference"], r["prediction"])["rougeL"]
            for r in records
        ]
        return {
            "precision": sum(s.precision for s in scores) / len(scores),
            "recall": sum(s.recall for s in scores) / len(scores),
            "f1": sum(s.fmeasure for s in scores) / len(scores),
        }
    except ImportError:
        return _rouge_l_fallback(records)


def _rouge_l_fallback(records: list[dict]) -> dict:
    """Dependency-free ROUGE-L using longest common subsequence."""
    def _lcs_length(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    precisions, recalls, f1s = [], [], []
    for r in records:
        g_tokens = _tokenize(r["prediction"])
        ref_tokens = _tokenize(r["reference"])
        if not g_tokens or not ref_tokens:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
        lcs_len = _lcs_length(g_tokens, ref_tokens)
        p = lcs_len / len(g_tokens)
        rec = lcs_len / len(ref_tokens)
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        precisions.append(p)
        recalls.append(rec)
        f1s.append(f1)

    n = len(records)
    return {
        "precision": sum(precisions) / n,
        "recall": sum(recalls) / n,
        "f1": sum(f1s) / n,
    }


def _bertscore_f1(records: list[dict], model_type: str | None = None) -> dict:
    """BERTScore F1 using ``bert_score`` library.

    **P0 fix**: Returns ``{"f1": None, ...}`` on ImportError instead of 0.0.
    The caller should skip ``None`` values when aggregating across clients.
    """
    if not records:
        return {"f1": None, "precision": None, "recall": None}

    candidates = [r["prediction"] for r in records]
    references = [r["reference"] for r in records]

    try:
        from bert_score import score as bertscore_fn
        P, R, F1 = bertscore_fn(
            candidates,
            references,
            model_type=model_type,
            lang="en" if model_type is None else None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )
        return {
            "f1": float(F1.mean().item()),
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "model_type": model_type,
        }
    except ImportError:
        return {
            "f1": None,
            "precision": None,
            "recall": None,
            "error": "bert_score library not installed",
        }


def _generation_quality(records: list[dict]) -> dict:
    """Deterministic, dependency-free generation diagnostics."""
    if not records:
        return {
            "exact_match": 0.0,
            "empty_predictions": 0,
            "empty_prediction_rate": 0.0,
            "average_generated_tokens": 0.0,
        }
    empty = sum(1 for r in records if not r["prediction"].strip())
    tokens = [r["generated_tokens"] for r in records]
    n = len(records)
    return {
        "exact_match": _exact_match_score(records),
        "empty_predictions": empty,
        "empty_prediction_rate": empty / n,
        "average_generated_tokens": sum(tokens) / n,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_generation_metrics(
    model,
    tokenizer,
    evalset,
    device: str,
    max_samples: int = 50,
    max_length: int = 512,
    max_new_tokens: int = 128,
    bertscore_model: str | None = None,
    return_predictions: bool = False,
) -> dict:
    """Compute all generation-based downstream metrics (Path B).

    Generates responses for Alpaca-style instructions and compares with
    reference responses using ROUGE-L, BERTScore, token-overlap accuracy,
    Macro-F1, and Exact Match.

    Args:
        model: Unwrapped model in eval mode.
        tokenizer: HuggingFace tokenizer.
        evalset: Dataset with ``text`` column.
        device: CUDA device string.
        max_samples: Maximum samples to evaluate.
        max_length: Maximum prompt tokenization length.
        max_new_tokens: Maximum tokens to generate.
        bertscore_model: Optional BERTScore model type (e.g. ``"roberta-large"``).
        return_predictions: If True, include per-sample predictions in output.

    Returns:
        Dict with all metrics and optional predictions.
    """
    records, gen_seconds = generate_predictions(
        model, tokenizer, evalset, device,
        max_samples=max_samples,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    if not records:
        return {
            "rouge_l": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "bertscore": {"f1": None, "precision": None, "recall": None},
            "generation_quality": {
                "exact_match": 0.0,
                "empty_predictions": 0,
                "empty_prediction_rate": 0.0,
                "average_generated_tokens": 0.0,
            },
            "token_overlap_accuracy": 0.0,
            "macro_f1": 0.0,
            "exact_match": 0.0,
            "num_eval_samples": 0,
            "generation_seconds": round(gen_seconds, 4),
        }

    rouge = _rouge_l_score(records)
    bertscore = _bertscore_f1(records, model_type=bertscore_model)
    gen_quality = _generation_quality(records)
    accuracy = _accuracy_score(records)
    macro_f1 = _macro_f1_score(records)
    exact_match = _exact_match_score(records)

    result = {
        "rouge_l": rouge,
        "bertscore": bertscore,
        "generation_quality": gen_quality,
        "token_overlap_accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match": round(exact_match, 4),
        "num_eval_samples": len(records),
        "generation_seconds": round(gen_seconds, 4),
        "generation_tokens": sum(r["generated_tokens"] for r in records),
    }
    if return_predictions:
        result["predictions"] = records
    return result
