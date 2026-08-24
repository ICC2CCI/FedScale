"""Generate sample answers from one or more PEFT checkpoints."""

import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

try:
    from fastchat.conversation import get_conv_template
except ImportError:
    get_conv_template = None


DEFAULT_QUESTIONS = [
    "Explain federated learning in one paragraph.",
    "What are three practical risks when training large language models across cloud clusters?",
    "Write a short Chinese summary of why checkpointing is useful during model training.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--peft-path",
        action="append",
        required=True,
        help="PEFT checkpoint directory. Repeat this option to compare checkpoints.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=None,
        help="Question to ask. Repeat this option for multiple questions.",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Optional text file with one question per line.",
    )
    parser.add_argument("--template", type=str, default="vicuna_v1.1")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-jsonl", type=str, default="checkpoint_eval.jsonl")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def load_questions(args):
    questions = []

    if args.question:
        questions.extend(args.question)

    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions.extend(line.strip() for line in f if line.strip())

    return questions or DEFAULT_QUESTIONS


def build_prompt(question, template):
    if get_conv_template is None:
        prompt = f"USER: {question}\nASSISTANT:"
        return None, prompt

    conv = get_conv_template(template)
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv, conv.get_prompt()


def clean_output(output_ids, input_len, model, tokenizer, conv):
    output_ids = output_ids[0] if model.config.is_encoder_decoder else output_ids[0][input_len:]

    if conv is None:
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    if conv.stop_token_ids:
        stop_positions = [
            idx for idx, token_id in enumerate(output_ids) if token_id in conv.stop_token_ids
        ]
        if stop_positions:
            output_ids = output_ids[: stop_positions[0]]

    output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)

    if conv.stop_str and output.find(conv.stop_str) > 0:
        output = output[: output.find(conv.stop_str)]

    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for token in special_token:
                output = output.replace(token, "")
        else:
            output = output.replace(special_token, "")

    return output.strip()


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(peft_path, local_files_only, device):
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    base_model = model.peft_config["default"].base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=local_files_only,
    )
    return model, tokenizer


def main():
    args = parse_args()
    questions = load_questions(args)
    output_path = Path(args.output_jsonl)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    with output_path.open("w", encoding="utf-8") as out:
        for peft_path in args.peft_path:
            print(f"\n=== Loading checkpoint: {peft_path} ===")
            model, tokenizer = load_model(peft_path, args.local_files_only, device)

            for question in questions:
                conv, prompt = build_prompt(question, args.template)
                input_ids = tokenizer([prompt]).input_ids
                input_tensor = torch.as_tensor(input_ids).to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_tensor,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature if args.temperature > 0 else None,
                        max_new_tokens=args.max_new_tokens,
                    )

                answer = clean_output(
                    output_ids,
                    len(input_ids[0]),
                    model,
                    tokenizer,
                    conv,
                )

                record = {
                    "checkpoint": peft_path,
                    "question": question,
                    "answer": answer,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                print(f"\n[checkpoint] {peft_path}")
                print(f"[question] {question}")
                print(f"[answer] {answer}")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\nSaved JSONL results to: {output_path}")


if __name__ == "__main__":
    main()
