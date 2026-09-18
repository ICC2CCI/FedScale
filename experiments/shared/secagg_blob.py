"""Pack many SecAgg masked windows into one self-describing blob.

Layout (little-endian), independent of model / window count:

    magic      : 4 bytes  b"SAW1"
    n_windows  : uint32
    repeated n_windows times:
        window_id  : uint32
        vector_len : uint32   # element count for unpack_zq
        nbytes     : uint32
        payload    : nbytes   # packed Z_q bytes (INT16 etc.)
"""
from __future__ import annotations

import struct
from typing import Iterable, List, Sequence, Tuple

MAGIC = b"SAW1"
_HEADER = struct.Struct("<I")
_REC = struct.Struct("<III")

MaskedWindowPart = Tuple[int, int, bytes]  # window_id, vector_len, payload


def pack_masked_windows_blob(parts: Sequence[MaskedWindowPart]) -> bytes:
    chunks: List[bytes] = [MAGIC, _HEADER.pack(len(parts))]
    for window_id, vector_len, payload in parts:
        payload = bytes(payload)
        chunks.append(_REC.pack(int(window_id), int(vector_len), len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def unpack_masked_windows_blob(data: bytes) -> List[MaskedWindowPart]:
    if not data or data[:4] != MAGIC:
        got = data[:4] if data else b""
        raise ValueError(f"SecAgg blob magic mismatch: {got!r}")
    (n,) = _HEADER.unpack_from(data, 4)
    off = 8
    out: List[MaskedWindowPart] = []
    for _ in range(int(n)):
        if off + _REC.size > len(data):
            raise ValueError("SecAgg blob truncated in record header")
        window_id, vector_len, nbytes = _REC.unpack_from(data, off)
        off += _REC.size
        if off + nbytes > len(data):
            raise ValueError(f"SecAgg blob truncated in window {window_id} payload")
        payload = data[off : off + nbytes]
        off += nbytes
        out.append((int(window_id), int(vector_len), payload))
    return out


def blob_window_ids(parts: Iterable[MaskedWindowPart]) -> List[int]:
    return [int(wid) for wid, _vlen, _payload in parts]
