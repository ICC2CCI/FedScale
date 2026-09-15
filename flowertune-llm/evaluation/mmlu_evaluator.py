"""MMLU / MMLU-Pro multiple-choice accuracy evaluation.

Implements the evaluation principles document §19:

    "外部能力测试：
        MMLU / MMLU-Pro Accuracy
        可选 IFEval instruction-following accuracy"

This module evaluates the model's objective knowledge/reasoning accuracy on
multiple-choice questions, which is complementary to the Alpaca-GPT4
generation-similarity metrics (ROUGE-L / BERTScore).  The evaluation
principles document §18 explicitly notes that ROUGE-L/BERTScore should not
be conflated with factual accuracy.

The implementation uses a likelihood-based approach: for each question, the
model computes the conditional log-likelihood of each answer choice given
the question prompt, and the highest-likelihood choice is selected.  This
is the standard MMLU evaluation protocol used by HuggingFace's
``lm-evaluation-harness``.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F


# Default MMLU subject list (57 subjects).  The loader can also handle
# MMLU-Pro which has a different category structure.
_MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions",
]


def _format_mmlu_question(subject: str, question: str, choices: list[str]) -> str:
    """Format an MMLU question in the standard prompt template."""
    letters = ["A", "B", "C", "D"]
    formatted_choices = "\n".join(
        f"{letters[i]}. {choices[i]}" for i in range(len(choices))
    )
    return (
        f"The following are multiple choice questions (with answers) about "
        f"{subject.replace('_', ' ')}.\n\n"
        f"{question}\n{formatted_choices}\nAnswer:"
    )


def _format_mmlu_pro_question(question: str, choices: list[str]) -> str:
    """Format an MMLU-Pro question (up to 10 options)."""
    letters = "ABCDEFGHIJ"
    formatted_choices = "\n".join(
        f"{letters[i]}. {choices[i]}" for i in range(len(choices))
    )
    return (
        f"The following are multiple choice questions (with answers) about "
        f"professional knowledge.\n\n"
        f"{question}\n{formatted_choices}\nAnswer:"
    )


def _choice_loglikelihood(
    model,
    tokenizer,
    prompt: str,
    choice_text: str,
    device: str,
    max_length: int = 2048,
) -> float:
    """Compute the conditional log-likelihood of *choice_text* given *prompt*.

    Tokenizes ``prompt + " " + choice_text`` and sums the log-probabilities
    of the choice tokens conditioned on the prompt.
    """
    full_text = prompt + " " + choice_text
    prompt_encodings = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
        add_special_tokens=True,
    )
    full_encodings = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length,
        add_special_tokens=True,
    )

    prompt_len = prompt_encodings["input_ids"].shape[1]
    full_ids = full_encodings["input_ids"].to(device)

    if full_ids.shape[1] <= prompt_len:
        return float("-inf")

    with torch.inference_mode():
        logits = model(input_ids=full_ids, use_cache=False).logits[:, :-1, :]
        # Log-probabilities of tokens after the prompt prefix.
        choice_logits = logits[:, prompt_len - 1:, :]
        choice_labels = full_ids[:, prompt_len:]

        log_probs = F.log_softmax(choice_logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, choice_labels.unsqueeze(-1)
        ).squeeze(-1)
        return float(token_log_probs.sum().item())


def evaluate_mmlu(
    model,
    tokenizer,
    device: str,
    dataset_name: str = "cais/mmlu",
    subset: str = "all",
    max_samples: int = 200,
    max_length: int = 2048,
    is_mmlu_pro: bool = False,
) -> dict:
    """Evaluate MMLU or MMLU-Pro multiple-choice accuracy.

    Uses likelihood-based scoring: for each question, the model computes the
    conditional log-likelihood of each answer choice, and the highest-
    likelihood choice is selected as the prediction.

    Args:
        model: Unwrapped model in eval mode.
        tokenizer: HuggingFace tokenizer.
        device: CUDA device string.
        dataset_name: HuggingFace dataset name (``cais/mmlu`` or ``TIGER-Lab/MMLU-Pro``).
        subset: MMLU config name (``"all"`` for the full test split).
        max_samples: Maximum number of questions to evaluate.
        max_length: Maximum tokenization length.
        is_mmlu_pro: If True, use MMLU-Pro formatting (up to 10 options).

    Returns:
        Dict with ``accuracy``, ``num_questions``, ``per_subject`` breakdown,
        and ``evaluated_samples``.
    """
    from datasets import load_dataset

    model.eval()
    started = time.perf_counter()

    if is_mmlu_pro:
        ds = load_dataset(dataset_name, "default", split="test")
    else:
        ds = load_dataset(dataset_name, subset, split="test")

    # Select a deterministic subset.
    n = min(len(ds), max_samples)
    step = max(1, len(ds) // n)
    indices = list(range(0, len(ds), step))[:n]

    correct = 0
    total = 0
    per_subject: dict[str, list[bool]] = {}

    for idx in indices:
        sample = ds[idx]

        if is_mmlu_pro:
            question = sample["question"]
            choices = sample["options"]
            answer_idx = int(sample["answer_index"])
            subject = sample.get("category", "unknown")
            prompt = _format_mmlu_pro_question(question, choices)
        else:
            question = sample["question"]
            choices = sample["choices"]
            answer_idx = int(sample["answer"])
            subject = sample.get("subject", "unknown")
            prompt = _format_mmlu_question(subject, question, choices)

        # Compute log-likelihood for each choice.
        best_idx = -1
        best_ll = float("-inf")
        for ci, choice in enumerate(choices):
            ll = _choice_loglikelihood(
                model, tokenizer, prompt, choice, device, max_length
            )
            if ll > best_ll:
                best_ll = ll
                best_idx = ci

        is_correct = best_idx == answer_idx
        if is_correct:
            correct += 1
        total += 1

        per_subject.setdefault(subject, []).append(is_correct)

    accuracy = correct / total if total > 0 else 0.0
    elapsed = time.perf_counter() - started

    # Per-subject breakdown.
    subject_accuracy = {}
    for subj, results in per_subject.items():
        subject_accuracy[subj] = {
            "accuracy": sum(results) / len(results),
            "num_questions": len(results),
        }

    return {
        "accuracy": round(accuracy, 4),
        "num_questions": total,
        "correct": correct,
        "per_subject": subject_accuracy,
        "dataset": dataset_name,
        "is_mmlu_pro": is_mmlu_pro,
        "evaluation_seconds": round(elapsed, 4),
    }
