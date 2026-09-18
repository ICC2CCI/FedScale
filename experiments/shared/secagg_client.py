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

from shared.block_selection import BlockDelta, _lookup_quant_residual_slice
from shared.fixed_point import quantize_to_zq, quantize_to_zq_with_feedback
from shared.fixed_point import (
    generate_rademacher_signs,
    hadamard_transform,
    inverse_hadamard_transform,
    hadamard_work_length,
    pad_to_length,
)
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
        quant_input: Optional[torch.Tensor] = None,
        signs: Optional[torch.Tensor] = None,
    ):
        """对单个 window 的 delta 做量化 + mask，返回 z_k (Z_q 整数向量)。

        z_k = q_k + Σ_{l>k} R_kl - Σ_{l<k} R_lk + B_k  (mod q)

        return_feedback=True 时返回 (z_k, residual, stats)，residual 用于 memory error-feedback。

        如果传入 quant_input（及可选 signs），跳过正向 Hadamard——amax 阶段已算过。
        residual 仍逆变换后裁回原始长度。
        """
        if not self.plan.secagg_session_id:
            raise RuntimeError(
                "SecAgg: secagg_session_id is empty; apply peer-keys session before mask_window"
            )

        q = self.plan.modulus_q
        q_max = self.plan.q_max
        scale = self.plan.get_window_scale(window.window_id)
        stochastic = self.plan.stochastic_rounding

        orig = delta_slice.to(dtype=torch.float32).reshape(-1)
        orig_len = int(orig.numel())
        if quant_input is not None:
            quant_input = quant_input.to(dtype=torch.float32).reshape(-1).contiguous()
            work_len = int(quant_input.numel())
            if signs is not None:
                signs = signs.to(dtype=torch.float32).reshape(-1)
        else:
            work_len = hadamard_work_length(orig_len, self.plan.hadamard_enabled)
            quant_input = pad_to_length(orig, work_len) if work_len != orig_len else orig.contiguous()
            signs = None
            if self.plan.hadamard_enabled and work_len > 0:
                seed = self.plan.hadamard_seed + window.window_id
                signs = generate_rademacher_signs(work_len, seed)
                quant_input = hadamard_transform(quant_input, signs)

        # 1. 定点量化（per-window / global scale）
        if return_feedback:
            q_k, residual_y, stats = quantize_to_zq_with_feedback(
                quant_input, scale, q_max, q, stochastic=stochastic,
            )
            if signs is not None:
                residual = inverse_hadamard_transform(residual_y, signs)
            else:
                residual = residual_y
            residual = residual.reshape(-1)[:orig_len].contiguous()
        else:
            q_k = quantize_to_zq(quant_input, scale, q_max, q, stochastic=stochastic)
            residual, stats = None, None

        # 2. 生成 pairwise masks（长度 = work_len，与量化向量一致）
        pairwise_masks: List[Tuple[int, torch.Tensor]] = []
        for peer_id, shared_secret in self.peer_secrets.items():
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
                vector_len=work_len,
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
            vector_len=work_len,
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


def _flat_fp32_slice(
    tensor: Optional[torch.Tensor],
    start: int,
    end: int,
    length: int,
) -> torch.Tensor:
    if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return torch.zeros(length, dtype=torch.float32)
    flat = tensor.detach().contiguous().view(-1).to(dtype=torch.float32)
    s = min(max(int(start), 0), int(flat.numel()))
    e = min(max(int(end), s), int(flat.numel()))
    sl = flat[s:e]
    if sl.numel() == length:
        return sl.contiguous()
    padded = torch.zeros(length, dtype=torch.float32)
    if sl.numel() > 0:
        padded[: sl.numel()] = sl
    return padded


def extract_window_slices(
    to_send: Dict[str, torch.Tensor],
    windows: List[WindowDescriptor],
) -> Dict[int, torch.Tensor]:
    """从 fp32/fp16 state 抽出每个 window 的 fp32 slice（量化必须用 fp32，不要先转 fp16）。"""
    slices: Dict[int, torch.Tensor] = {}
    for window in windows:
        slices[window.window_id] = _flat_fp32_slice(
            to_send.get(window.key_name),
            window.start,
            window.end,
            window.vector_length,
        )
    return slices


def extract_window_to_send_fp32(
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    block_memory: Dict[str, torch.Tensor],
    quant_residual: BlockDelta,
    windows: List[WindowDescriptor],
) -> Dict[int, torch.Tensor]:
    """按 window 在 FP32 中计算 x = delta + block_memory + quant_residual。

    不物化完整 FP32 state，也不在量化前 round 回 FP16。
    """
    slices: Dict[int, torch.Tensor] = {}
    for window in windows:
        length = int(window.vector_length)
        local_sl = _flat_fp32_slice(local_state.get(window.key_name), window.start, window.end, length)
        global_sl = _flat_fp32_slice(
            global_state.get(window.key_name) if global_state is not None else None,
            window.start,
            window.end,
            length,
        )
        mem_sl = _flat_fp32_slice(
            block_memory.get(window.key_name) if block_memory is not None else None,
            window.start,
            window.end,
            length,
        )
        x = local_sl - global_sl + mem_sl
        res = _lookup_quant_residual_slice(
            quant_residual or {}, window.key_name, window.start, window.end,
        )
        if res is not None:
            res_f = res.detach().to(dtype=torch.float32).reshape(-1)
            n = min(int(res_f.numel()), length)
            if n > 0:
                x[:n] = x[:n] + res_f[:n]
        slices[window.window_id] = x.contiguous()
    return slices


def window_quant_input_and_amax(
    delta_slice: torch.Tensor,
    window: WindowDescriptor,
    plan: SecAggPlan,
) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
    """返回量化域输入、amax，以及 Hadamard signs（未旋转时 signs=None）。"""
    quant_input = delta_slice.to(dtype=torch.float32).reshape(-1)
    signs = None
    if plan.hadamard_enabled:
        work_len = hadamard_work_length(int(quant_input.numel()), True)
        quant_input = pad_to_length(quant_input, work_len)
        if work_len > 0:
            seed = plan.hadamard_seed + window.window_id
            signs = generate_rademacher_signs(work_len, seed)
            quant_input = hadamard_transform(quant_input, signs)
    amax = float(quant_input.abs().max().item()) if quant_input.numel() else 0.0
    return quant_input, amax, signs


def window_amax_payload(
    slices: Dict[int, torch.Tensor],
    windows: Optional[List[WindowDescriptor]] = None,
    plan: Optional[SecAggPlan] = None,
) -> Dict[str, float]:
    """构造 key-announce 用的 per-window max|delta|（JSON 键必须是 str）。

    仅非 Hadamard 路径使用。Hadamard 路径请用 global_amax_from_slices，
    避免把 269 个 per-window L∞ 泄露给 server。
    """
    window_by_id = {w.window_id: w for w in (windows or [])}
    out: Dict[str, float] = {}
    for wid, tensor in slices.items():
        window = window_by_id.get(int(wid))
        if plan is not None and window is not None and plan.hadamard_enabled:
            _quant_input, amax, _signs = window_quant_input_and_amax(tensor, window, plan)
            out[str(wid)] = amax
            continue
        if tensor is None or tensor.numel() == 0:
            out[str(wid)] = 0.0
        else:
            out[str(wid)] = float(tensor.abs().max().item())
    return out


def prepare_window_quant_cache(
    slices: Dict[int, torch.Tensor],
    windows: List[WindowDescriptor],
    plan: SecAggPlan,
) -> Tuple[float, Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
    """对每个 window 做一次（可选）Hadamard，返回 global_amax 与可复用的 (y, signs)。

    量化必须等 server 下发 scale 之后；正向旋转与 amax 不依赖 scale，只做一遍。
    """
    cache: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    amax = 0.0
    for window in windows:
        tensor = slices.get(window.window_id)
        if tensor is None:
            tensor = torch.zeros(window.vector_length, dtype=torch.float32)
        quant_input, window_amax, signs = window_quant_input_and_amax(tensor, window, plan)
        cache[window.window_id] = (quant_input, signs)
        if window_amax > amax:
            amax = window_amax
    return float(amax), cache


def global_amax_from_slices(
    slices: Dict[int, torch.Tensor],
    windows: List[WindowDescriptor],
    plan: SecAggPlan,
) -> float:
    """量化域全局 amax：Hadamard 开启时为旋转后 max|y|，否则为原始 max|x|。

    只上报这一个标量，不暴露哪个 window 更大。
    """
    amax, _cache = prepare_window_quant_cache(slices, windows, plan)
    return float(amax)
