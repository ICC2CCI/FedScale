"""Three-way train/validation/test split with stable hash-based selection.

The split is deterministic across runs and across different training methods
(DDP, FSDP, FedAvg, FedScale), satisfying the fairness requirement from the
evaluation principles document §10:

    "DDP、FSDP、Plain FedAvg、FedScale
     必须使用相同的 train / validation / test 划分。"
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def stable_sample_key(sample: dict) -> str:
    """Return a deterministic SHA-256 key for an Alpaca-style sample.

    Uses ``instruction`` and ``input`` so the same logical question always
    maps to the same split regardless of formatting or response changes.
    """
    source = f"{sample.get('instruction', '')}\0{sample.get('input', '')}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DataSplit:
    """Result of a three-way dataset split."""

    train: "Any"  # datasets.Dataset
    validation: "Any"  # datasets.Dataset
    test: "Any"  # datasets.Dataset
    split_info: dict


def create_train_val_test_split(
    dataset: "Any",  # datasets.Dataset
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> DataSplit:
    """Split a dataset into train / validation / test using stable hashing.

    The partition is deterministic: samples are sorted by
    ``sha256(instruction + NUL + input)`` and then sliced by ratio.  This
    guarantees that every training method evaluates on the *same* held-out
    data, enabling fair comparison.

    Args:
        dataset: HuggingFace Dataset with ``instruction`` and ``input`` columns,
                 or a ``text`` column (Alpaca-formatted).
        val_ratio: Fraction of data for validation (default 5%).
        test_ratio: Fraction of data for test (default 5%).
        seed: Not used for the hash itself, but kept for API compatibility
              and future stratified-split extensions.

    Returns:
        ``DataSplit`` with ``train``, ``validation``, ``test`` datasets and
        ``split_info`` metadata.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot split an empty dataset")

    from datasets import Dataset  # noqa: F811 — typed hint only

    # Sort indices by stable hash for deterministic, method-agnostic splitting.
    indices = sorted(range(n), key=lambda i: stable_sample_key(dataset[i]))

    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError(
            f"Dataset too small ({n} samples) for val_ratio={val_ratio}, "
            f"test_ratio={test_ratio}"
        )

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    train_ds = dataset.select(train_indices)
    val_ds = dataset.select(val_indices)
    test_ds = dataset.select(test_indices)

    split_info = {
        "total_samples": n,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "test_samples": len(test_ds),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "selection": "ascending sha256(instruction + NUL + input)",
    }

    return DataSplit(
        train=train_ds,
        validation=val_ds,
        test=test_ds,
        split_info=split_info,
    )


def reconstruct_held_out_test_set(
    dataset_name: str,
    num_partitions: int = 2,
    test_ratio: float = 0.05,
    seed: int = 42,
    max_samples: int = 100,
) -> tuple["Any", dict]:  # (datasets.Dataset, info)
    """Reconstruct the union of per-partition held-out test subsets.

    Mirrors the federated training partition order (IID first), then applies
    the same stable-hash split inside every partition.  The union of all
    per-partition test subsets is the federated test set.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. ``vicgalle/alpaca-gpt4``).
        num_partitions: Number of IID federated partitions.
        test_ratio: Fraction of each partition reserved for testing.
        seed: Split seed for compatibility.
        max_samples: Maximum number of test samples to select.

    Returns:
        ``(selected_test_set, partition_info)``
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import IidPartitioner
    from datasets import concatenate_datasets

    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": IidPartitioner(num_partitions=num_partitions)},
    )

    test_partitions: list[Dataset] = []
    partition_sizes: list[dict] = []

    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id, "train")
        split = create_train_val_test_split(
            partition,
            val_ratio=0.0,
            test_ratio=test_ratio,
            seed=seed,
        )
        test_partitions.append(split.test)
        partition_sizes.append(
            {
                "partition_id": partition_id,
                "total": len(partition),
                "test": len(split.test),
            }
        )

    held_out = concatenate_datasets(test_partitions)

    # Select a deterministic subset of the held-out union.
    ordered_indices = sorted(
        range(len(held_out)), key=lambda i: stable_sample_key(held_out[i])
    )
    selected_count = min(max_samples, len(ordered_indices))
    selected = held_out.select(ordered_indices[:selected_count])

    info = {
        "dataset": dataset_name,
        "num_partitions": num_partitions,
        "test_ratio": test_ratio,
        "held_out_union_samples": len(held_out),
        "selected_samples": selected_count,
        "partition_sizes": partition_sizes,
    }

    return selected, info
