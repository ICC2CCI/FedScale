"""Canonical encoding for mask_hash / layout_hash computation.

Per the "FedScale Public Block Mask v1" spec section 4.3:
  - Fixed UTF-8 domain-separation strings
  - job_id / key_name: length-prefixed bytes (uint32 big-endian length + raw bytes)
  - integers: fixed-width unsigned 64-bit big-endian
  - hashes: fixed 32-byte raw digest

This module provides deterministic, cross-implementation-reproducible
encoding so that server and all ICCs compute identical mask_hash / layout_hash.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

# A canonical block is (key_name, start, end) — the same tuple used in GroupBlocks
CanonicalBlock = Tuple[str, int, int]


def _encode_str(s: str) -> bytes:
    """Length-prefixed UTF-8 string: uint32 BE length + raw bytes."""
    raw = s.encode("utf-8")
    return len(raw).to_bytes(4, "big") + raw


def _encode_u64(n: int) -> bytes:
    """Fixed-width unsigned 64-bit big-endian."""
    return int(n).to_bytes(8, "big", signed=False)


def encode_block(block: CanonicalBlock) -> bytes:
    """Canonical encode of a single block (key_name, start, end)."""
    key_name, start, end = block
    return _encode_str(key_name) + _encode_u64(start) + _encode_u64(end)


def encode_block_list(blocks: Sequence[CanonicalBlock]) -> bytes:
    """Canonical encode of an ordered list of blocks.

    The encoding is:
      uint64(count) || encode_block(block_0) || encode_block(block_1) || ...
    """
    out = bytearray()
    out += _encode_u64(len(blocks))
    for b in blocks:
        out += encode_block(b)
    return bytes(out)


def encode_selected_blocks(selected: "dict[str, list[tuple[int, int]]] | list") -> bytes:
    """Canonical encode of a selected_blocks structure for mask_hash.

    Accepts either:
      - SelectedByKey dict: {key_name: [(start, end), ...]}
      - ordered list of (key_name, start, end)

    For cross-implementation consistency, we sort by (key_name, start, end)
    then encode as a flat block list. This ensures mask_hash is independent
    of dict insertion order.
    """
    if isinstance(selected, dict):
        flat: List[CanonicalBlock] = []
        for key_name, slices in selected.items():
            for s, e in slices:
                flat.append((key_name, int(s), int(e)))
        flat.sort()
        return encode_block_list(flat)
    else:
        flat2 = [(str(kn), int(s), int(e)) for kn, s, e in selected]
        flat2.sort()
        return encode_block_list(flat2)


def encode_layout(
    group_blocks: "dict[str, list[tuple[str, int, int]]]",
    always_on_keys: "set[str] | None" = None,
    coverage_h: int = 0,
    block_size: int = 0,
) -> bytes:
    """Canonical encode of the full block layout for layout_hash.

    Encoding:
      uint64(coverage_h) || uint64(block_size) ||
      uint64(num_groups) ||
      for each group (sorted by gid):
        encode_str(gid) || uint64(num_blocks_in_group) ||
        for each block (in canonical order):
          encode_block(block) || uint8(is_always_on)

    The always_on flag (0 or 1) is included per-block so that layout_hash
    changes if always_on designation changes.
    """
    always_on_keys = always_on_keys or set()
    out = bytearray()
    out += _encode_u64(coverage_h)
    out += _encode_u64(block_size)
    out += _encode_u64(len(group_blocks))
    for gid in sorted(group_blocks):
        out += _encode_str(gid)
        blocks = group_blocks[gid]
        out += _encode_u64(len(blocks))
        for block in blocks:
            out += encode_block(block)
            # always_on flag: block is always_on if its key_name is in always_on set
            is_on = 1 if block[0] in always_on_keys else 0
            out += bytes([is_on])
    return bytes(out)


def compute_mask_hash(selected: "dict[str, list[tuple[int, int]]] | list") -> str:
    """Compute SHA256 hex digest of canonical-encoded selected blocks."""
    return hashlib.sha256(encode_selected_blocks(selected)).hexdigest()


def compute_layout_hash(
    group_blocks: "dict[str, list[tuple[str, int, int]]]",
    always_on_keys: "set[str] | None" = None,
    coverage_h: int = 0,
    block_size: int = 0,
) -> str:
    """Compute SHA256 hex digest of canonical-encoded layout."""
    return hashlib.sha256(
        encode_layout(group_blocks, always_on_keys, coverage_h, block_size)
    ).hexdigest()


# Mask policy identifier (spec section 9: mask_policy_id)
MASK_POLICY_ID = "hierarchical-permute-rotate-v1"
