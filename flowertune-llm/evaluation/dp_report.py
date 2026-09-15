"""RDP (Rényi Differential Privacy) accounting and DP-utility curve support.

Implements the FedScale design document §5.4.3 and experiment E4 (§11.7):

    ρ_r(α) = α * S_r² / (2 * σ_agg,r²)

    ρ_tot(α) = Σ_r ρ_r(α)

    ε(δ) = min_{α>1} [ρ_tot(α) + ln(1/δ) / (α-1)]

Where:
    S_r ≤ 2C / q_r  (L2 sensitivity under ICC-level replacement adjacency)
    σ_agg,r = σ_client,r / √q_r  (aggregated noise standard deviation)

The DP-utility curve (E4) evaluates model quality (loss, PPL, ROUGE-L, etc.)
under different noise levels, enabling the privacy-utility trade-off analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RDPResult:
    """Result of RDP accounting for a federated training run."""

    epsilon: float
    delta: float
    alpha_optimal: float
    rdp_per_round: list[float]
    total_rounds: int
    clip_norm: float
    sigma_client: float
    min_cohort_size: int
    parameters: dict


def compute_rdp_epsilon(
    num_rounds: int,
    clip_norm: float,
    sigma_client: float,
    cohort_size: int,
    delta: float = 1e-5,
    alpha_values: Sequence[float] | None = None,
) -> RDPResult:
    """Compute the final (ε, δ)-DP guarantee for a federated training run.

    Assumes ICC-level / update-level distributed differential privacy where:
    - Each ICC clips its update to L2 norm ≤ C (clip_norm).
    - Each ICC adds Gaussian noise with std σ_client to its clipped update.
    - The server averages over q_r successful participants.
    - The aggregated noise std is σ_agg = σ_client / √q_r.

    The L2 sensitivity under replacement adjacency is S_r = 2C / q_r.

    Args:
        num_rounds: Number of completed federated rounds R.
        clip_norm: Clipping threshold C.
        sigma_client: Per-ICC noise standard deviation σ_client.
        cohort_size: Number of successful participants q_r (assumed constant).
        delta: Target δ for (ε, δ)-DP conversion.
        alpha_values: Rényi order values to search.  If None, uses a default
                      range from 1.5 to 256.

    Returns:
        ``RDPResult`` with the final ε, optimal α, and per-round RDP values.
    """
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if sigma_client <= 0:
        raise ValueError("sigma_client must be positive")
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive")

    if alpha_values is None:
        alpha_values = [1.5, 2, 3, 4, 8, 16, 32, 64, 128, 256]

    # Aggregated noise std: σ_agg = σ_client / √q
    sigma_agg = sigma_client / math.sqrt(cohort_size)

    # L2 sensitivity under replacement: S = 2C / q
    sensitivity = 2.0 * clip_norm / cohort_size

    rdp_per_round: list[float] = []
    best_epsilon = float("inf")
    best_alpha = 0.0

    for alpha in alpha_values:
        # Per-round RDP: ρ_r(α) = α * S² / (2 * σ_agg²)
        rdp_round = alpha * sensitivity ** 2 / (2.0 * sigma_agg ** 2)
        rdp_total = rdp_round * num_rounds

        # Convert to (ε, δ)-DP: ε = ρ_total + ln(1/δ) / (α - 1)
        epsilon = rdp_total + math.log(1.0 / delta) / (alpha - 1.0)

        rdp_per_round.append(round(rdp_round, 6))

        if epsilon < best_epsilon:
            best_epsilon = epsilon
            best_alpha = alpha

    return RDPResult(
        epsilon=round(best_epsilon, 4),
        delta=delta,
        alpha_optimal=best_alpha,
        rdp_per_round=rdp_per_round,
        total_rounds=num_rounds,
        clip_norm=clip_norm,
        sigma_client=sigma_client,
        min_cohort_size=cohort_size,
        parameters={
            "sigma_agg": round(sigma_agg, 6),
            "sensitivity": round(sensitivity, 6),
            "alpha_values": list(alpha_values),
            "adjacency": "ICC-level / update-level replacement",
        },
    )


@dataclass(frozen=True)
class DPUtilityPoint:
    """One point on the DP-utility curve."""

    sigma_client: float
    epsilon: float
    delta: float
    metrics: dict


def dp_utility_curve(
    sigma_values: Sequence[float],
    utility_fn,
    num_rounds: int,
    clip_norm: float,
    cohort_size: int,
    delta: float = 1e-5,
    alpha_values: Sequence[float] | None = None,
) -> list[DPUtilityPoint]:
    """Generate a DP-utility trade-off curve (Experiment E4).

    For each noise level σ in *sigma_values*:
    1. Compute the (ε, δ)-DP guarantee via RDP accounting.
    2. Call ``utility_fn(sigma)`` to obtain model quality metrics.
    3. Record the (σ, ε, metrics) point on the curve.

    The design document §11.7 recommends:
        σ ∈ {0, σ_low, σ_mid, σ_high}

    Args:
        sigma_values: List of σ_client values to evaluate.
        utility_fn: Callable that takes σ_client (float) and returns a dict
                    of model quality metrics (e.g. val_loss, rouge_l, etc.).
        num_rounds: Number of completed federated rounds.
        clip_norm: Clipping threshold C.
        cohort_size: Number of participants q.
        delta: Target δ for DP conversion.
        alpha_values: Rényi orders for RDP search.

    Returns:
        List of ``DPUtilityPoint``, one per σ value.
    """
    results: list[DPUtilityPoint] = []

    for sigma in sigma_values:
        if sigma <= 0:
            # No noise → no DP guarantee.
            metrics = utility_fn(0.0)
            results.append(DPUtilityPoint(
                sigma_client=0.0,
                epsilon=float("inf"),
                delta=delta,
                metrics=metrics,
            ))
            continue

        rdp = compute_rdp_epsilon(
            num_rounds=num_rounds,
            clip_norm=clip_norm,
            sigma_client=sigma,
            cohort_size=cohort_size,
            delta=delta,
            alpha_values=alpha_values,
        )

        metrics = utility_fn(sigma)
        results.append(DPUtilityPoint(
            sigma_client=sigma,
            epsilon=rdp.epsilon,
            delta=delta,
            metrics=metrics,
        ))

    return results


def format_dp_report(curve: list[DPUtilityPoint]) -> str:
    """Format a DP-utility curve as a readable report string."""
    lines = [
        "DP-Utility Trade-off Curve",
        "=" * 60,
        f"{'σ_client':>12} {'ε':>10} {'δ':>12}  Metrics",
        "-" * 60,
    ]
    for point in curve:
        eps_str = f"{point.epsilon:.4f}" if point.epsilon != float("inf") else "∞"
        metrics_str = ", ".join(
            f"{k}={v}" for k, v in point.metrics.items()
            if isinstance(v, (int, float))
        )
        lines.append(
            f"{point.sigma_client:>12.4f} {eps_str:>10} {point.delta:>12.0e}  {metrics_str}"
        )
    return "\n".join(lines)
