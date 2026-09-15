"""Client 端 SecAgg 逻辑：密钥生成、DH 协商、mask、量化、上传。

职责：
    1. 生成 X25519 密钥对 + self_master
    2. 提交公钥给 server，获取其他 client 公钥
    3. DH 协商 pairwise shared secret
    4. 对每个 window: 量化 → 加 mask → 上传 z_k
    5. 提交 self_master 给 server（2-client 无掉线场景）
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from shared.fixed_point import quantize_to_zq
from shared.protocol import SecAggPlan, WindowDescriptor
from shared.secagg_crypto import (
    apply_mask,
    compute_pairwise_mask,
    compute_self_mask,
    dh_shared_secret,
    derive_pair_seed,
    generate_keypair,
    generate_self_master,
    build_pair_domain_info,
)

logger = logging.getLogger(__name__)


class SecAggClient:
    """Client 端 SecAgg 状态机（per-round, per-attempt）。"""

    def __init__(
        self,
        client_id: int,
        secagg_plan: SecAggPlan,
        windows: List[WindowDescriptor],
    ) -> None:
        self.client_id = client_id
        self.plan = secagg_plan
        self.windows = windows

        # 生成密钥
        self.private_key, self.public_key = generate_keypair()
        self.self_master = generate_self_master()

        # DH 共享秘密: peer_id → shared_secret (bytes)
        self.peer_secrets: Dict[int, bytes] = {}

        logger.info(
            "SecAggClient %s: initialized, %s windows, session=%s",
            client_id, len(windows), secagg_plan.secagg_session_id[:16] if secagg_plan.secagg_session_id else "?",
        )

    # ----------------------------------------------------------------------- #
    # Phase 1: 公钥提交 + DH 协商
    # ----------------------------------------------------------------------- #

    def get_public_key_hex(self) -> str:
        """返回自己的公钥（hex，用于 JSON 传输给 server）。"""
        return self.public_key.hex()

    def setup_dh(self, peer_public_keys: Dict[str, str]) -> None:
        """收到所有 peer 的公钥后，计算 DH 共享秘密。

        peer_public_keys: {client_id_str: pk_hex, ...}
        """
        for cid_str, pk_hex in peer_public_keys.items():
            peer_id = int(cid_str)
            if peer_id == self.client_id:
                continue
            peer_pk = bytes.fromhex(pk_hex)
            shared = dh_shared_secret(self.private_key, peer_pk)
            self.peer_secrets[peer_id] = shared
            logger.info(
                "SecAggClient %s: DH with peer %s established",
                self.client_id, peer_id,
            )

    def get_self_master_hex(self) -> str:
        """返回 self_master（hex，用于提交给 server）。"""
        return self.self_master.hex()

    # ----------------------------------------------------------------------- #
    # Phase 3: 量化 + mask + 上传
    # ----------------------------------------------------------------------- #

    def mask_window(
        self,
        window: WindowDescriptor,
        delta_slice: torch.Tensor,
    ) -> torch.Tensor:
        """对单个 window 的 delta 做量化 + mask，返回 z_k (Z_q 整数向量)。

        z_k = q_k + Σ_{l>k} R_kl - Σ_{l<k} R_lk + B_k  (mod q)
        """
        q = self.plan.modulus_q
        q_max = self.plan.q_max
        scale = self.plan.quantization_scale
        stochastic = self.plan.stochastic_rounding

        # 1. 定点量化
        q_k = quantize_to_zq(delta_slice, scale, q_max, q, stochastic=stochastic)

        # 2. 生成 pairwise masks
        pairwise_masks: List[Tuple[int, torch.Tensor]] = []
        for peer_id, shared_secret in self.peer_secrets.items():
            # 派生 pair_seed
            domain = build_pair_domain_info(
                session_id=self.plan.secagg_session_id,
                attempt_id=self.plan.attempt_id,
                cohort_hash=self.plan.cohort_hash,
                window_id=window.window_id,
                window_layout_hash=window.window_layout_hash,
                client_a_id=self.client_id,
                client_b_id=peer_id,
            )
            pair_seed = derive_pair_seed(shared_secret, domain)
            R = compute_pairwise_mask(
                pair_seed=pair_seed,
                session_id=self.plan.secagg_session_id,
                attempt_id=self.plan.attempt_id,
                cohort_hash=self.plan.cohort_hash,
                window_id=window.window_id,
                window_layout_hash=window.window_layout_hash,
                client_a_id=self.client_id,
                client_b_id=peer_id,
                vector_len=window.vector_length,
                q=q,
            )
            pairwise_masks.append((peer_id, R))

        # 3. 生成 self mask
        B_k = compute_self_mask(
            self_master=self.self_master,
            session_id=self.plan.secagg_session_id,
            attempt_id=self.plan.attempt_id,
            client_id=self.client_id,
            window_id=window.window_id,
            window_layout_hash=window.window_layout_hash,
            vector_len=window.vector_length,
            q=q,
        )

        # 4. 加 mask
        z_k = apply_mask(q_k, pairwise_masks, B_k, self.client_id, q)
        return z_k

    def mask_all_windows(
        self,
        delta_slices: Dict[int, torch.Tensor],  # window_id → delta_slice
    ) -> Dict[int, torch.Tensor]:
        """对所有 window 做 mask，返回 {window_id → z_k}。"""
        results: Dict[int, torch.Tensor] = {}
        for window in self.windows:
            delta_slice = delta_slices.get(window.window_id)
            if delta_slice is None:
                # 该 window 无 delta（可能未被选中或 delta=0）
                delta_slice = torch.zeros(window.vector_length, dtype=torch.float32)
            z_k = self.mask_window(window, delta_slice)
            results[window.window_id] = z_k
        return results
