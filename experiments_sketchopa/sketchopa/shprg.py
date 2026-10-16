"""Seed-Homomorphic PRG (SHPRG)。

基于 LWR（Learning with Rounding）风格实现，满足同态性：
    SHPRG(s1) + SHPRG(s2) = SHPRG(s1 + s2)  (mod q)

原理：用 seed 作为 AES-CTR 密钥流的种子，但为保证同态性，
采用「seed 在整数环上可加，PRG 输出在模 q 下可加」的设计：
    SHPRG(s) = PRG_stream(s) mod q
    SHPRG(s1+s2) = PRG_stream(s1+s2) mod q

由于标准 AES-CTR 不具备 SHPRG(s1)+SHPRG(s2)=SHPRG(s1+s2)，
这里采用基于 LWR 的构造：把 seed 直接作为确定性 PRF 输入，
输出向量在 Z_q 上，且 seed 空间定义为 Z_q_seed（小整数环），
seed 相加 = 输出相加（mod q）。

实现采用「seed -> 长伪随机向量 (mod q)」的确定性扩展，
通过让 seed 直接参与线性生成来保证同态：
    SHPRG(s)[i] = (A[i] * s + b[i]) mod q
其中 A, b 由公开 ctx 派生（与 seed 无关）。
这样 SHPRG(s1) + SHPRG(s2) = A*(s1+s2) + 2b != SHPRG(s1+s2) = A*(s1+s2)+b
因此 b 必须为 0：SHPRG(s)[i] = (A[i] * s) mod q。

为增强随机性，seed 扩展为多元素种子向量 s=(s_0,...,s_{m-1})，
SHPRG(s) = A @ s mod q，A 由 ctx 派生。

本实现提供实用版本：seed 是单个整数，输出长度 L 的向量在 Z_q 上。
"""
from __future__ import annotations

import hashlib
from typing import List

import torch


class SHPRG:
    """Seed-Homomorphic PRG（LWR 风格）。

    SHPRG(s) = (A @ s_vec) mod q，A 由 ctx 派生。
    同态性：SHPRG(s1) + SHPRG(s2) = A@(s1+s2) = SHPRG(s1+s2) (mod q)。

    seed 空间：Z_q（模 q 整数）。
    输出：长度 L 的 Z_q 向量。
    """

    def __init__(
        self,
        length: int,
        ctx_hash: str = "",
        modulus_q: int = 2**16,       # q
        seed_dim: int = 4,            # seed 向量维度
    ):
        self.length = length
        self.ctx_hash = ctx_hash
        self.modulus_q = modulus_q
        self.seed_dim = seed_dim
        # 派生矩阵 A (length × seed_dim)，元素 ∈ Z_q
        self._A = self._derive_matrix_A()

    def _derive_matrix_A(self) -> torch.Tensor:
        """从 ctx_hash 派生 A 矩阵 (length × seed_dim)，元素 ∈ [0, q)。

        向量化实现：用 ctx_hash 作为 torch 随机种子，一次性生成整个矩阵。
        """
        # 用 ctx_hash 派生 torch 种子
        seed = int(hashlib.sha256(f"shprg_A|{self.ctx_hash}".encode()).hexdigest()[:16], 16)
        g = torch.Generator()
        g.manual_seed(seed)
        # 一次性生成 (length × seed_dim) 的 Z_q 矩阵
        A = torch.randint(0, self.modulus_q, (self.length, self.seed_dim), generator=g, dtype=torch.int64)
        return A

    def expand(self, seed_vec: List[int]) -> torch.Tensor:
        """SHPRG(s) = (A @ s_vec) mod q，返回 Z_q 向量。

        Args:
            seed_vec: 长度 seed_dim 的整数向量

        Returns:
            (length,) int64 tensor，元素 ∈ [0, q)
        """
        s = torch.tensor(seed_vec, dtype=torch.int64)
        if s.numel() != self.seed_dim:
            raise ValueError(f"seed_vec dim {s.numel()} != seed_dim {self.seed_dim}")
        out = (self._A @ s) % self.modulus_q
        return out

    def expand_single(self, seed: int) -> torch.Tensor:
        """单整数 seed → seed_vec 扩展 → SHPRG。

        seed_vec 通过 hash(seed) 确定性派生。
        注意：单整数 seed 的同态性发生在「seed 整数相加后重新派生 seed_vec」时，
        为保证严格同态，应直接在 seed_vec 层面相加（见 add_seed_vectors）。
        """
        seed_vec = self._seed_to_vec(seed)
        return self.expand(seed_vec)

    def _seed_to_vec(self, seed: int) -> List[int]:
        """单整数 seed → seed_dim 维向量（确定性）。"""
        h = hashlib.sha256(f"shprg_seed|{int(seed)}".encode()).digest()
        vec = []
        for j in range(self.seed_dim):
            off = j * 2
            if off + 2 > len(h):
                h = hashlib.sha256(h + b"|more").digest()
                off = 0
            v = int.from_bytes(h[off:off+2], "little") % self.modulus_q
            vec.append(v)
        return vec

    @staticmethod
    def add_seed_vectors(
        s1: List[int],
        s2: List[int],
        modulus_q: int,
    ) -> List[int]:
        """在 seed 空间内相加（mod q）。"""
        if len(s1) != len(s2):
            raise ValueError("seed vector dim mismatch")
        return [(int(a) + int(b)) % modulus_q for a, b in zip(s1, s2)]

    def mask_sketch_zq(
        self,
        sketch_zq: torch.Tensor,
        seed_vec: List[int],
    ) -> torch.Tensor:
        """在 Z_q 上加掩码：ct = (sketch_zq + SHPRG(s)) mod q。

        sketch 需先量化到 Z_q（见 quant.py int16 模式，scale 把浮点映射到 [-q/2,q/2) 再 mod q）。
        """
        mask_q = self.expand(seed_vec)  # [0, q)
        ct = (sketch_zq.to(torch.int64) + mask_q) % self.modulus_q
        return ct

    def unmask_sketch_zq(
        self,
        agg_ct: torch.Tensor,
        agg_seed_vec: List[int],
    ) -> torch.Tensor:
        """在 Z_q 上减掩码：U = (agg_ct - SHPRG(s_sum)) mod q。"""
        mask_q = self.expand(agg_seed_vec)
        U = (agg_ct.to(torch.int64) - mask_q) % self.modulus_q
        return U
