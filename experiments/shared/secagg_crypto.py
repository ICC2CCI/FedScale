"""SecAgg 密码学原语：X25519 DH、SHAKE-256 PRG、HKDF、mask 生成。

依赖：
    cryptography 库（X25519 DH）—— 三端已安装
    hashlib（SHAKE-256 PRG）—— 标准库

设计原则：
    - 所有函数纯函数式，无全局状态
    - mask 生成跨平台确定性（SHAKE-256 + HKDF-SHA256）
    - DH 用 X25519（RFC 7748），公钥 32 bytes raw
    - 每层域分离通过 HKDF info 实现（spec section 8）
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# cryptography 库（X25519 DH）
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


# ---------------------------------------------------------------------------
# X25519 DH 密钥协商
# ---------------------------------------------------------------------------


def generate_keypair() -> Tuple[bytes, bytes]:
    """生成 X25519 密钥对。

    返回 (private_key_raw, public_key_raw)，均为 32 bytes。
    private_key_raw 仅 client 自己持有，绝不发给 server。
    public_key_raw 通过 server relay 发给其他 client。
    """
    sk = X25519PrivateKey.generate()
    pk = sk.public_key()
    sk_bytes = sk.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pk_bytes = pk.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return sk_bytes, pk_bytes


def dh_shared_secret(private_key_raw: bytes, peer_public_key_raw: bytes) -> bytes:
    """从己方私钥 + 对方公钥计算 DH 共享秘密（32 bytes）。

    双方独立调用得到相同的 shared_secret。
    Server 永远看不到这个值。
    """
    sk = X25519PrivateKey.from_private_bytes(private_key_raw)
    pk = X25519PublicKey.from_public_bytes(peer_public_key_raw)
    return sk.exchange(pk)


def public_key_from_raw(raw: bytes) -> X25519PublicKey:
    """从 raw bytes 还原公钥对象。"""
    return X25519PublicKey.from_public_bytes(raw)


# ---------------------------------------------------------------------------
# HKDF-SHA256 密钥派生（RFC 5869）
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


def derive_pair_seed(
    shared_secret: bytes,
    domain_info: bytes,
) -> bytes:
    """从 DH 共享秘密派生 pairwise mask 的种子（32 bytes）。

    domain_info 包含 session_id, attempt_id, window_id, window_layout_hash 等
    （spec section 6.1）。
    """
    return hkdf_sha256(
        ikm=shared_secret,
        info=b"FedScale-SecAgg-PairMask-v2|" + domain_info,
        length=32,
    )


def derive_self_seed(
    self_master: bytes,
    domain_info: bytes,
) -> bytes:
    """从 self_master 派生 self mask 的种子（32 bytes）。

    domain_info 包含 session_id, attempt_id, client_id, window_id 等
    （spec section 7.2）。
    """
    return hkdf_sha256(
        ikm=self_master,
        info=b"FedScale-SecAgg-SelfMask-v2|" + domain_info,
        length=32,
    )


def generate_self_master() -> bytes:
    """生成 self_master（32 bytes 随机数）。client 端生成，server 不知道。"""
    return os.urandom(32)


# ---------------------------------------------------------------------------
# SHAKE-256 PRG：确定性伪随机流
# ---------------------------------------------------------------------------


def prg_stream(seed: bytes, length: int) -> bytes:
    """SHAKE-256 伪随机流。确定性、跨平台一致。

    seed: 任意长度种子
    length: 输出字节数
    """
    return hashlib.shake_256(seed).digest(length)


def derive_mask_vector(
    seed: bytes,
    vector_len: int,
    q: int,
) -> torch.Tensor:
    """从种子生成 Z_q 上的随机向量。

    返回 int64 tensor，值在 [0, q)。

    实现细节：
    - 每个 Z_q 元素需要 ceil(modulus_bits / 8) bytes
    - 用 SHAKE-256 生成足够的随机字节
    - 解析为整数后 mod q
    """
    modulus_bits = q.bit_length()
    n_bytes_per_elem = (modulus_bits + 7) // 8

    # 生成随机字节流
    stream = prg_stream(seed, vector_len * n_bytes_per_elem)

    # 解析为整数
    arr = np.frombuffer(stream, dtype=np.uint8)
    if n_bytes_per_elem == 1:
        ints = arr.astype(np.int64)
    elif n_bytes_per_elem == 2:
        ints = arr.reshape(-1, 2).copy().view(np.uint16).astype(np.int64)
    elif n_bytes_per_elem == 3:
        ints = (
            arr[0::3].astype(np.uint32)
            | (arr[1::3].astype(np.uint32) << 8)
            | (arr[2::3].astype(np.uint32) << 16)
        ).astype(np.int64)
    elif n_bytes_per_elem == 4:
        ints = arr.reshape(-1, 4).copy().view(np.uint32).astype(np.int64)
    else:
        # 通用路径：逐元素读取
        ints = np.zeros(vector_len, dtype=np.int64)
        for i in range(vector_len):
            val = 0
            for j in range(n_bytes_per_elem):
                val |= int(arr[i * n_bytes_per_elem + j]) << (8 * j)
            ints[i] = val

    # mod q → [0, q)
    ints = ints % q
    return torch.from_numpy(ints)


# ---------------------------------------------------------------------------
# 域分离 info 构建（spec section 8）
# ---------------------------------------------------------------------------


def build_domain_info(
    session_id: str,
    attempt_id: int,
    window_id: int,
    window_layout_hash: str,
    extra: bytes = b"",
) -> bytes:
    """构建 HKDF info（域分离）。

    包含: session_id, attempt_id, window_id, window_layout_hash
    确保 mask 不跨 session/attempt/window 复用。
    """
    return (
        session_id.encode("utf-8")
        + b"|" + attempt_id.to_bytes(4, "big")
        + b"|" + window_id.to_bytes(4, "big")
        + b"|" + window_layout_hash.encode("utf-8")
        + b"|" + extra
    )


def build_pair_domain_info(
    session_id: str,
    attempt_id: int,
    cohort_hash: str,
    window_id: int,
    window_layout_hash: str,
    client_a_id: int,
    client_b_id: int,
) -> bytes:
    """构建 pairwise mask 的域分离 info（含 peer pair 排序）。

    spec section 6.1: peer 身份按 canonical ICC id 排序。
    """
    lo, hi = min(client_a_id, client_b_id), max(client_a_id, client_b_id)
    return (
        session_id.encode("utf-8")
        + b"|" + attempt_id.to_bytes(4, "big")
        + b"|" + cohort_hash.encode("utf-8")
        + b"|" + window_id.to_bytes(4, "big")
        + b"|" + window_layout_hash.encode("utf-8")
        + b"|" + lo.to_bytes(4, "big")
        + b"|" + hi.to_bytes(4, "big")
    )


def build_self_domain_info(
    session_id: str,
    attempt_id: int,
    client_id: int,
    window_id: int,
    window_layout_hash: str,
) -> bytes:
    """构建 self mask 的域分离 info。"""
    return (
        session_id.encode("utf-8")
        + b"|" + attempt_id.to_bytes(4, "big")
        + b"|" + client_id.to_bytes(4, "big")
        + b"|" + window_id.to_bytes(4, "big")
        + b"|" + window_layout_hash.encode("utf-8")
    )


# ---------------------------------------------------------------------------
# Mask 生成 + 加掩码
# ---------------------------------------------------------------------------


def compute_pairwise_mask(
    pair_seed: bytes,
    session_id: str,
    attempt_id: int,
    cohort_hash: str,
    window_id: int,
    window_layout_hash: str,
    client_a_id: int,
    client_b_id: int,
    vector_len: int,
    q: int,
) -> torch.Tensor:
    """计算 pairwise mask 向量 R_{ab,w}。

    返回 Z_q 上的随机向量。双方独立调用得到相同结果。
    """
    domain = build_pair_domain_info(
        session_id, attempt_id, cohort_hash,
        window_id, window_layout_hash, client_a_id, client_b_id,
    )
    seed = derive_pair_seed(pair_seed, domain)
    return derive_mask_vector(seed, vector_len, q)


def compute_self_mask(
    self_master: bytes,
    session_id: str,
    attempt_id: int,
    client_id: int,
    window_id: int,
    window_layout_hash: str,
    vector_len: int,
    q: int,
) -> torch.Tensor:
    """计算 self mask 向量 B_{k,w}。

    返回 Z_q 上的随机向量。client 和 server（拿到 self_master 后）都能计算。
    """
    domain = build_self_domain_info(
        session_id, attempt_id, client_id, window_id, window_layout_hash,
    )
    seed = derive_self_seed(self_master, domain)
    return derive_mask_vector(seed, vector_len, q)


def apply_mask(
    q_zq: torch.Tensor,
    pairwise_masks: List[Tuple[int, torch.Tensor]],  # [(peer_id, R_kl), ...]
    self_mask: torch.Tensor,
    client_id: int,
    q: int,
) -> torch.Tensor:
    """对量化值加 pairwise mask + self mask。

    符号规则（spec section 6.3）：
        - client_id 小的一方加 R_kl
        - client_id 大的一方减 R_kl
        - self mask 总是加

    z_k = q_k + Σ_{l>k} R_kl - Σ_{l<k} R_lk + B_k  (mod q)

    pairwise_masks: [(peer_id, mask_vector), ...]
    """
    z = q_zq.to(dtype=torch.int64).clone()
    # pairwise masks
    for peer_id, mask in pairwise_masks:
        if client_id < peer_id:
            z = (z + mask.to(dtype=torch.int64)) % q
        else:
            z = (z - mask.to(dtype=torch.int64)) % q
    # self mask
    z = (z + self_mask.to(dtype=torch.int64)) % q
    # 确保非负
    z = (z + q) % q
    return z


def compute_session_id(
    job_id: str,
    round_idx: int,
    successful_round_index: int,
    base_model_version: str,
    cohort_hash: str,
    layout_hash: str,
    mask_hash: str,
) -> str:
    """计算 SecAgg session_id（spec section 5.1）。

    secagg_session_id = H(CanonicalEncode(
        "FedScale-SecAgg-v2", job_id, round_id, successful_round_index,
        base_model_version, cohort_hash, layout_hash, mask_hash))
    """
    import hashlib

    data = (
        b"FedScale-SecAgg-v2|"
        + job_id.encode("utf-8") + b"|"
        + str(round_idx).encode("utf-8") + b"|"
        + str(successful_round_index).encode("utf-8") + b"|"
        + base_model_version.encode("utf-8") + b"|"
        + cohort_hash.encode("utf-8") + b"|"
        + layout_hash.encode("utf-8") + b"|"
        + mask_hash.encode("utf-8")
    )
    return hashlib.sha256(data).hexdigest()


def compute_cohort_hash(client_ids: List[int]) -> str:
    """计算 cohort_hash（sorted client_ids 的 SHA256）。"""
    import hashlib

    data = b"cohort|" + b"|".join(str(c).encode() for c in sorted(client_ids))
    return hashlib.sha256(data).hexdigest()
