"""Shamir 门限秘密分享（seed share）+ 2-client 退化。

原方案：client 把 seed s_i 通过 Shamir (t, n) 分享给 n 个辅助节点，
辅助节点在 seed 空间求和后返回聚合份额，server 收集 ≥t 份重构 s_sum。

2-client 退化（§5.2）：
- 两个 client 互为辅助节点
- 每个 client 把自己的 seed_vec 直接发给对方（t=1 退化，无需 Shamir 多份）
- 对方在 seed 空间求和后返回
- Server 收齐两份聚合份额重构 s_sum = s_0 + s_1

注意：Shamir 秘密分享要求模数为素数。这里用 SHAMIR_PRIME = 65521（< 2^16 的最大素数）。
SHPRG 的模数 2^16 与此不同（LWR 风格用 2 的幂），seed_vec 在两者间通过 mod 转换。
"""
from __future__ import annotations

import hashlib
import random
from typing import List, Tuple

import torch

# Shamir 要求素数模数；65521 是 < 2^16 的最大素数
SHAMIR_PRIME = 65521


def _eval_poly(coeffs: List[int], x: int, modulus: int) -> int:
    """在 Z_modulus 上求值多项式 coeffs[0] + coeffs[1]*x + ... """
    result = 0
    x_pow = 1
    for c in coeffs:
        result = (result + c * x_pow) % modulus
        x_pow = (x_pow * x) % modulus
    return result


def shamir_split(
    secret: int,
    n: int,
    t: int,
    modulus: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Shamir (t, n) 分享单个整数 secret。

    Returns:
        shares: [(x_i, y_i), ...] 共 n 份
    """
    if t > n:
        raise ValueError("t > n")
    if t < 1:
        raise ValueError("t < 1")
    coeffs = [secret % modulus] + [rng.randrange(modulus) for _ in range(t - 1)]
    shares: List[Tuple[int, int]] = []
    for i in range(1, n + 1):
        x = i
        y = _eval_poly(coeffs, x, modulus)
        shares.append((x, y))
    return shares


def shamir_reconstruct(
    shares: List[Tuple[int, int]],
    modulus: int,
) -> int:
    """Lagrange 插值重构 secret（常数项）。"""
    secret = 0
    for j, (xj, yj) in enumerate(shares):
        num = 1
        den = 1
        for m, (xm, _) in enumerate(shares):
            if m == j:
                continue
            num = (num * (-xm)) % modulus
            den = (den * (xj - xm)) % modulus
        # 模逆
        den_inv = pow(den, -1, modulus)
        lagrange = (num * den_inv) % modulus
        secret = (secret + yj * lagrange) % modulus
    return secret % modulus


# ---- 向量版（seed_vec 多维） ----

def shamir_split_vec(
    secret_vec: List[int],
    n: int,
    t: int,
    modulus: int,
    rng: random.Random,
) -> List[Tuple[int, List[int]]]:
    """对 seed_vec 每个分量做 Shamir 分享。

    Returns:
        shares: [(x_i, [y_i_0, y_i_1, ...]), ...] 共 n 份
    """
    per_dim: List[List[Tuple[int, int]]] = []
    for s in secret_vec:
        per_dim.append(shamir_split(s, n, t, modulus, rng))

    shares: List[Tuple[int, List[int]]] = []
    for i in range(n):
        x = per_dim[0][i][0]
        ys = [per_dim[d][i][1] for d in range(len(secret_vec))]
        shares.append((x, ys))
    return shares


def shamir_reconstruct_vec(
    shares: List[Tuple[int, List[int]]],
    modulus: int,
) -> List[int]:
    """向量版重构。"""
    dim = len(shares[0][1])
    out: List[int] = []
    for d in range(dim):
        scalar_shares = [(x, ys[d]) for (x, ys) in shares]
        out.append(shamir_reconstruct(scalar_shares, modulus))
    return out


def aggregate_shares_vec(
    shares_from_clients: List[List[int]],
    modulus: int,
) -> List[int]:
    """辅助节点：在 seed 空间求和各 client 的 share。

    Args:
        shares_from_clients: 每个 client 发来的 share（同 x），每项是 seed_vec

    Returns:
        聚合 share（seed_vec 各分量 mod q 求和）
    """
    dim = len(shares_from_clients[0])
    out = [0] * dim
    for sv in shares_from_clients:
        for d in range(dim):
            out[d] = (out[d] + sv[d]) % modulus
    return out


# ---- 2-client 退化封装 ----

class TwoClientSeedExchange:
    """2-client 退化：client 互为辅助节点，t=1。

    流程：
    1. client_0 把 seed_vec_0 发给 client_1；client_1 把 seed_vec_1 发给 client_0
    2. client_0 作为辅助节点：agg_share_0 = seed_vec_0 + seed_vec_1（收到的）
       client_1 作为辅助节点：agg_share_1 = seed_vec_0（收到的）+ seed_vec_1
       —— 两者相同
    3. server 收任一份即可重构 s_sum（t=1）

    实际上 2-client t=1 退化为「双方直接交换 seed 并求和」。
    """

    def __init__(self, modulus: int):
        self.modulus = modulus

    def client_pack_share(self, own_seed_vec: List[int]) -> List[int]:
        """client 把自己的 seed_vec 打包发给对方（t=1 无需 Shamir）。"""
        return [s % self.modulus for s in own_seed_vec]

    def auxiliary_aggregate(
        self,
        own_seed_vec: List[int],
        received_seed_vec: List[int],
    ) -> List[int]:
        """辅助节点：聚合自己 + 收到的 seed（求和 mod q）。"""
        return aggregate_shares_vec([own_seed_vec, received_seed_vec], self.modulus)

    def server_reconstruct(self, agg_shares: List[List[int]]) -> List[int]:
        """server 收 ≥1 份聚合 share 即可（t=1）。"""
        if not agg_shares:
            raise ValueError("no agg shares")
        # t=1：直接返回第一份
        return agg_shares[0]
