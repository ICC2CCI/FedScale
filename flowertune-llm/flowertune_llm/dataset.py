from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

FDS = None  # Cache FederatedDataset

ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"


def _select_ultrachat_sft(dataset_dict):
    """Expose UltraChat's train_sft split as the federated train split."""
    from datasets import DatasetDict

    if "train_sft" not in dataset_dict:
        raise ValueError(
            "UltraChat 200k cache is missing the required train_sft split; "
            f"available splits: {list(dataset_dict)}"
        )
    return DatasetDict({"train": dataset_dict["train_sft"]})


def _alpaca_user_content(instruction, user_input):
    """Map an Alpaca record to the content of one ChatML user turn."""
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if user_input:
        return f"{instruction}\n\n{user_input}"
    return instruction


def _chatml_formatter(tokenizer):
    """Return an SFT formatter that converts Alpaca records to Qwen ChatML."""
    def formatting_prompts_func(example):
        output_texts = []
        inputs = example.get("input", [""] * len(example["instruction"]))
        for instruction, user_input, response in zip(
            example["instruction"], inputs, example["response"]
        ):
            messages = [
                {"role": "user", "content": _alpaca_user_content(instruction, user_input)},
                {"role": "assistant", "content": response},
            ]
            output_texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        return output_texts

    return formatting_prompts_func


def _alpaca_completion_formatter(example):
    """Format non-chat base models such as OpenLLaMA for completion-only SFT."""
    output_texts = []
    inputs = example.get("input", [""] * len(example["instruction"]))
    prefix = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    for instruction, user_input, response in zip(
        example["instruction"], inputs, example["response"]
    ):
        user_content = _alpaca_user_content(instruction, user_input)
        output_texts.append(
            f"{prefix}\n### Instruction:\n{user_content}\n### Response: {response}"
        )
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    """Load tokenizer with detailed logging and optimized parameters.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        tokenizer, data_collator, formatting_prompts_func
    """
    import os
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"Loading tokenizer for: {model_name}")
    print(f"{'='*60}")

    # Print HuggingFace environment configuration
    print(f"\nHuggingFace Environment:")
    print(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'not set')}")
    print(f"  HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', 'not set')}")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'not set')}")
    print(f"  HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'not set')}")
    print(f"  HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', 'not set')}")

    # Check if model_name is a local path or HuggingFace repo
    is_local_path = Path(model_name).exists()
    print(f"\nModel source: {'Local path' if is_local_path else 'HuggingFace Hub'}")

    # If it's a HuggingFace repo, check local cache first
    if not is_local_path:
        cache_dir = os.environ.get('HF_HOME', '/app/.cache/huggingface')
        model_cache_path = Path(cache_dir) / 'hub' / f"models--{model_name.replace('/', '--')}"

        print(f"\nChecking local cache:")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Expected model cache: {model_cache_path}")

        if model_cache_path.exists():
            # Check snapshots directory
            snapshots_path = model_cache_path / 'snapshots'
            if snapshots_path.exists():
                snapshot_dirs = list(snapshots_path.iterdir())
                if snapshot_dirs:
                    latest_snapshot = snapshot_dirs[0]
                    files = list(latest_snapshot.iterdir())
                    print(f"  ✓ Cache found with {len(files)} files")
                    print(f"  Snapshot: {latest_snapshot.name}")

                    # List key files
                    key_files = ['tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'config.json']
                    for kf in key_files:
                        kf_path = latest_snapshot / kf
                        if kf_path.exists():
                            size = kf_path.stat().st_size
                            print(f"    ✓ {kf}: {size/1024:.1f}KB")
                        else:
                            print(f"    ✗ {kf}: not found")
                else:
                    print(f"  ⚠ Cache directory exists but no snapshots found")
            else:
                print(f"  ⚠ Snapshots directory not found")
        else:
            print(f"  ✗ Cache not found, will attempt download")
    else:
        print(f"  Using local path: {model_name}")

    # Load tokenizer with optimized parameters
    print(f"\nLoading tokenizer...")
    try:
        # Use legacy=False to avoid SentencePiece/Tiktoken conversion issues
        # Add trust_remote_code=True if needed for custom tokenizers
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="right",
            legacy=False,  # Use new tokenizer behavior to avoid conversion errors
            # Don't set local_files_only - let transformers auto-detect
            # Don't set trust_remote_code unless needed (security risk)
        )
        print(f"✓ Tokenizer loaded successfully")

        # Print tokenizer info
        print(f"\nTokenizer info:")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Model max length: {tokenizer.model_max_length}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  BOS token: {tokenizer.bos_token}")

    except Exception as e:
        print(f"✗ Failed to load tokenizer: {type(e).__name__}: {str(e)[:300]}")
        raise

    tokenizer.pad_token = tokenizer.eos_token
    is_qwen_chat_model = model_name.lower().startswith("qwen/")
    if is_qwen_chat_model:
        response_template_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        formatting_func = _chatml_formatter(tokenizer)
    else:
        response_template_ids = tokenizer.encode(
            "\n### Response:", add_special_tokens=False
        )[2:]
        formatting_func = _alpaca_completion_formatter
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    print(f"{'='*60}\n")

    return tokenizer, data_collator, formatting_func


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        kwargs = {"dataset": dataset_name, "partitioners": {"train": partitioner}}
        if str(dataset_name).strip().lower() == ULTRACHAT_DATASET.lower():
            kwargs["preprocessor"] = _select_ultrachat_sft
        FDS = FederatedDataset(**kwargs)
    client_trainset = FDS.load_partition(partition_id, "train")
    if "output" in client_trainset.column_names and "response" not in client_trainset.column_names:
        client_trainset = client_trainset.rename_column("output", "response")

    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
