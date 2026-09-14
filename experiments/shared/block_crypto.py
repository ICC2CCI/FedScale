"""SEC-1/2/3：上传隐私防护——block_id 化 + 加密 + 末 block 填充。

威胁模型：截获 client→server 上传包的人不应从 payload 推断出
对应的模型参数名 / 位置 / 层类型。

三层组合（均不影响训练精度）：
- SEC-1：上传 payload 不含 key_name，用 global block id (gidx) 替代
- SEC-2：gidx 经 per-round 对称密钥（HKDF-SHA256 + XOR 流）加密
- SEC-3：末 block 填充到统一 BLOCK_SIZE，截获者无法从大小区分

纯标准库实现（hashlib + hmac），不依赖 cryptography / pycryptodome。
模型无关：gidx 由 server 从 reference_state 的 state_dict 自动编号，
适用于任意基于 state_dict 的神经网络。
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Dict, List, Optional, Tuple

import torch

from .protocol import DEFAULT_BLOCK_SIZE

# ---------------------------------------------------------------------------
# HKDF-SHA256 (RFC 5869) —— 纯标准库实现
# ---------------------------------------------------------------------------


def hkdf_sha256(
    ikm: bytes,
    salt: bytes = b"",
    info: bytes = b"",
    length: int = 32,
) -> bytes:
    """HKDF-SHA256 密钥派生。length <= 255*32。"""
    if not salt:
        salt = b"\x00" * 32
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    t = b""
    i = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]


# ---------------------------------------------------------------------------
# SEC-2：per-round 对称密钥派生 + gidx 加解密
# ---------------------------------------------------------------------------


def derive_round_key(epoch_seed: bytes, round_idx: int) -> bytes:
    """从 epoch_seed + round_idx 派生 per-round 密钥（4 字节够盖 gidx）。

    server 与 client 用相同 (epoch_seed, round_idx) 即可独立派生相同密钥，
    无需密钥分发。epoch_seed 由 server 在每个 Mask Epoch 开始时生成并广播
    （当前实现复用 protocol.build_permutations 的 seed 机制）。
    """
    info = b"FedScale-SecAgg-v1|round-" + str(round_idx).encode("ascii")
    return hkdf_sha256(epoch_seed, info=info, length=4)


def encrypt_gidx(gidx: int, round_key: bytes) -> bytes:
    """加密单个 block id（4 字节大端 + XOR 流）。返回 4 字节密文。"""
    gidx_bytes = int(gidx).to_bytes(4, "big")
    return bytes(a ^ b for a, b in zip(gidx_bytes, round_key))


def decrypt_gidx(enc: bytes, round_key: bytes) -> int:
    """解密 block id。"""
    dec = bytes(a ^ b for a, b in zip(enc, round_key))
    return int.from_bytes(dec, "big")


# ---------------------------------------------------------------------------
# SEC-3：末 block 填充到统一 BLOCK_SIZE
# ---------------------------------------------------------------------------


def pad_slice(slice_data: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    """若 slice 长度 < block_size，用 0 填充到 block_size。

    返回的 tensor 长度恒等于 block_size（除非 slice 本身就等于 block_size）。
    填充值用 0：delta 的 0 填充不影响聚合（加权后仍为 0）。
    """
    n = int(slice_data.numel())
    if n >= block_size:
        return slice_data
    pad_len = block_size - n
    pad = torch.zeros(pad_len, dtype=slice_data.dtype, device=slice_data.device)
    return torch.cat([slice_data, pad])


def unpad_slice(padded: torch.Tensor, real_len: int) -> torch.Tensor:
    """截取前 real_len 个元素（server 端按 start:end 还原）。"""
    if int(padded.numel()) <= real_len:
        return padded
    return padded[:real_len]


# ---------------------------------------------------------------------------
# SEC-1：block_id 化的 payload 编解码
# ---------------------------------------------------------------------------

# SEC-1 payload 格式（pipeline per-block 上传）:
#   {
#     "v": 1,                              # 版本号
#     "enc_gidx": bytes(4),                # SEC-2 加密后的 gidx
#     "slice": torch.Tensor[BLOCK_SIZE],   # SEC-3 填充后的 slice
#     "real_len": int,                     # SEC-3 真实长度（unpad 用）
#     "is_int8": bool,                     # 是否 int8 量化
#     "scale": Optional[float],            # int8 量化 scale
#     "num_examples": int,
#     "train_loss": float,
#     "eval_loss": Optional[float],
#   }
#
# 旧格式（v=0 / 无 v）仍兼容：{"block_delta": {key_name: [(s,e,slice),...]}, ...}

SEC_PAYLOAD_VERSION = 1


def encode_sec_block_payload(
    gidx: int,
    slice_data: torch.Tensor,
    round_key: bytes,
    *,
    real_len: Optional[int] = None,
    is_int8: bool = False,
    scale: Optional[float] = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_examples: int = 1,
    train_loss: float = 0.0,
    eval_loss: Optional[float] = None,
) -> Dict:
    """SEC-1/2/3：编码单个 block 的上传 payload。

    - gidx 经 SEC-2 加密
    - slice 经 SEC-3 填充到 block_size
    - 不含 key_name（SEC-1）
    """
    if real_len is None:
        real_len = int(slice_data.numel())
    padded = pad_slice(slice_data, block_size=block_size)
    enc_gidx = encrypt_gidx(gidx, round_key)
    payload: Dict = {
        "v": SEC_PAYLOAD_VERSION,
        "enc_gidx": enc_gidx,
        "slice": padded,
        "real_len": int(real_len),
        "is_int8": bool(is_int8),
        "scale": float(scale) if scale is not None else None,
        "num_examples": int(num_examples),
        "train_loss": float(train_loss),
        "eval_loss": float(eval_loss) if eval_loss is not None else None,
    }
    return payload


def decode_sec_block_payload(
    payload: Dict,
    round_key: bytes,
) -> Tuple[int, torch.Tensor, int, bool, Optional[float]]:
    """SEC-1/2/3：解码单个 block 的上传 payload。

    返回 (gidx, slice_data, real_len, is_int8, scale)。
    slice_data 已 unpad 到 real_len 长度。
    """
    enc_gidx = payload["enc_gidx"]
    gidx = decrypt_gidx(enc_gidx, round_key)
    real_len = int(payload.get("real_len", 0))
    padded = payload["slice"]
    slice_data = unpad_slice(padded, real_len)
    is_int8 = bool(payload.get("is_int8", False))
    scale = payload.get("scale")
    return gidx, slice_data, real_len, is_int8, (float(scale) if scale is not None else None)


def is_sec_payload(payload: Any) -> bool:
    """判断 payload 是否为 SEC-1/2/3 格式。"""
    return isinstance(payload, dict) and payload.get("v") == SEC_PAYLOAD_VERSION
