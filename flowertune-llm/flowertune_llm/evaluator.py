"""Model quality evaluation: validation metrics and downstream task metrics."""

import math
import torch
import torch.nn.functional as F
from collections import Counter


def create_eval_split(trainset, split_ratio=0.1):
    """Split training set into train and evaluation subsets.

    Args:
        trainset: HuggingFace Dataset with 'text' column.
        split_ratio: Fraction of data to use for evaluation.

    Returns:
        (trainset_split, evalset)
    """
    split = trainset.train_test_split(test_size=split_ratio, seed=42)
    return split["train"], split["test"]


def compute_validation_metrics(model, tokenizer, evalset, device, max_samples=50):
    """Compute response-only validation loss and perplexity.

    Args:
        model: Unwrapped PEFT model used for evaluation.
        tokenizer: HuggingFace tokenizer.
        evalset: Dataset with 'text' column.
        device: cuda device.
        max_samples: Maximum number of samples to evaluate.

    Returns:
        {"val_loss": float, "perplexity": float}
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_samples = min(len(evalset), max_samples)

    with torch.no_grad():
        for i in range(num_samples):
            text = evalset[i]["text"]
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].clone().contiguous()

            # Match SFT's completion-only training objective: prompt/system/user
            # tokens are context, not prediction targets. This makes validation
            # loss comparable in scope to train_loss while keeping the data held out.
            response_marker = None
            for marker in ("\n### Response:", "<|im_start|>assistant\n"):
                if marker in text:
                    response_marker = marker
                    break
            if response_marker is not None:
                prompt = text.split(response_marker, 1)[0] + response_marker
                prompt_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )["input_ids"]
                # labels[t] predicts input_ids[t + 1]. Mask predictions whose
                # target still belongs to the prompt prefix.
                prompt_targets = min(
                    max(prompt_ids.shape[1] - 1, 0), labels.shape[1]
                )
                labels[:, :prompt_targets] = -100

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )
            total_loss += loss.item()
            total_tokens += int((labels != -100).sum().item())

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        "loss_scope": "assistant_response_only",
    }


def compute_downstream_metrics(model, tokenizer, evalset, device, max_samples=50):
    """Compute downstream task metrics via generation comparison.

    Generates responses for Alpaca-style instructions and compares with
    reference responses using Accuracy, Macro-F1, Exact Match, ROUGE-L,
    and BERTScore.

    Args:
        model: Unwrapped PEFT model used for evaluation.
        tokenizer: HuggingFace tokenizer.
        evalset: Dataset with 'text' column containing Alpaca prompts.
        device: cuda device.
        max_samples: Maximum number of samples to evaluate.

    Returns:
        {"accuracy": float, "macro_f1": float, "exact_match": float,
         "rouge_l": float, "bertscore_f1": float}
    """
    model.eval()
    num_samples = min(len(evalset), max_samples)

    generated_texts = []
    reference_texts = []

    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    with torch.no_grad():
        for i in range(num_samples):
            sample = evalset[i]
            text = sample["text"]

            if "### Response:" in text:
                prompt_part = text.split("### Response:")[0] + "### Response:"
                reference = text.split("### Response:")[1].strip()
            else:
                prompt_part = text
                reference = ""

            if not reference:
                continue

            encodings = tokenizer(
                prompt_part,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            generated_texts.append(generated)
            reference_texts.append(reference)

    if not generated_texts:
        return {
            "accuracy": 0.0, "macro_f1": 0.0, "exact_match": 0.0,
            "rouge_l": 0.0, "bertscore_f1": 0.0,
            "num_eval_samples": 0,
        }

    exact_match = _exact_match_score(generated_texts, reference_texts)
    accuracy = _accuracy_score(generated_texts, reference_texts)
    macro_f1 = _macro_f1_score(generated_texts, reference_texts)
    rouge_l = _rouge_l_score(generated_texts, reference_texts)
    bertscore_f1 = _bertscore_f1_score(generated_texts, reference_texts)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match": round(exact_match, 4),
        "rouge_l": round(rouge_l, 4),
        "bertscore_f1": round(bertscore_f1, 4),
        "num_eval_samples": len(generated_texts),
    }


def _exact_match_score(generated, references):
    """Fraction of generated texts that exactly match references."""
    matches = sum(1 for g, r in zip(generated, references) if g.strip() == r.strip())
    return matches / len(generated) if generated else 0.0


def _accuracy_score(generated, references):
    """Loose accuracy: generated text contains key tokens from reference."""
    correct = 0
    for g, r in zip(generated, references):
        ref_tokens = set(r.lower().split())
        gen_tokens = set(g.lower().split())
        if ref_tokens and len(ref_tokens & gen_tokens) / len(ref_tokens) >= 0.3:
            correct += 1
    return correct / len(generated) if generated else 0.0


def _macro_f1_score(generated, references):
    """Token-level macro F1 across all samples."""
    f1_scores = []
    for g, r in zip(generated, references):
        gen_tokens = g.lower().split()
        ref_tokens = r.lower().split()
        gen_counter = Counter(gen_tokens)
        ref_counter = Counter(ref_tokens)

        common = gen_counter & ref_counter
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        precision = num_same / len(gen_tokens) if gen_tokens else 0.0
        recall = num_same / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def _rouge_l_score(generated, references):
    """ROUGE-L F1 score using longest common subsequence."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for g, r in zip(generated, references):
            score = scorer.score(r, g)
            scores.append(score["rougeL"].fmeasure)
        return sum(scores) / len(scores) if scores else 0.0
    except ImportError:
        return _rouge_l_fallback(generated, references)


def _rouge_l_fallback(generated, references):
    """Fallback ROUGE-L using simple LCS."""
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    scores = []
    for g, r in zip(generated, references):
        g_tokens = g.lower().split()
        r_tokens = r.lower().split()
        if not g_tokens or not r_tokens:
            scores.append(0.0)
            continue
        lcs_len = lcs_length(g_tokens, r_tokens)
        precision = lcs_len / len(g_tokens)
        recall = lcs_len / len(r_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def _bertscore_f1_score(generated, references):
    """BERTScore F1 using bert-score library."""
    try:
        from bert_score import score as bertscore_fn
        P, R, F1 = bertscore_fn(
            generated, references,
            lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )
        return F1.mean().item()
    except ImportError:
        return 0.0
