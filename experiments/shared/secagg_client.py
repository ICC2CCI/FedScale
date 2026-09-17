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

from shared.fixed_point import quantize_to_zq, quantize_to_zq_with_feedback
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
        return_feedback: bool = False,
    ):
        """对单个 window 的 delta 做量化 + mask，返回 z_k (Z_q 整数向量)。

        z_k = q_k + Σ_{l>k} R_kl - Σ_{l<k} R_lk + B_k  (mod q)

        return_feedback=True 时返回 (z_k, residual, stats)，residual 用于 memory error-feedback。
        """
        q = self.plan.modulus_q
        q_max = self.plan.q_max
        scale = self.plan.get_window_scale(window.window_id)
        stochastic = self.plan.stochastic_rounding

        # 1. 定点量化（per-window scale）
        if return_feedback:
            q_k, residual, stats = quantize_to_zq_with_feedback(
                delta_slice, scale, q_max, q, stochastic=stochastic,
            )
        else:
            q_k = quantize_to_zq(delta_slice, scale, q_max, q, stochastic=stochastic)
            residual, stats = None, None

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
        if return_feedback:
            return z_k, residual, stats
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


def extract_window_slices(
    to_send: Dict[str, torch.Tensor],
    windows: List[WindowDescriptor],
) -> Dict[int, torch.Tensor]:
    """从 fp32/fp16 state 抽出每个 window 的 fp32 slice（量化必须用 fp32，不要先转 fp16）。"""
    slices: Dict[int, torch.Tensor] = {}
    for window in windows:
        tensor = to_send.get(window.key_name)
        if tensor is None or not tensor.is_floating_point():
            slices[window.window_id] = torch.zeros(window.vector_length, dtype=torch.float32)
            continue
        flat = tensor.contiguous().view(-1)
        end = min(int(window.end), int(flat.numel()))
        start = min(int(window.start), end)
        sl = flat[start:end].detach().to(dtype=torch.float32)
        if sl.numel() < window.vector_length:
            padded = torch.zeros(window.vector_length, dtype=torch.float32)
            if sl.numel() > 0:
                padded[: sl.numel()] = sl
            sl = padded
        slices[window.window_id] = sl.contiguous()
    return slices


def window_amax_payload(slices: Dict[int, torch.Tensor]) -> Dict[str, float]:
    """构造 key-announce 用的 per-window max|delta|（JSON 键必须是 str）。"""
    out: Dict[str, float] = {}
    for wid, tensor in slices.items():
        if tensor is None or tensor.numel() == 0:
            out[str(wid)] = 0.0
        else:
            out[str(wid)] = float(tensor.abs().max().item())
    return out
