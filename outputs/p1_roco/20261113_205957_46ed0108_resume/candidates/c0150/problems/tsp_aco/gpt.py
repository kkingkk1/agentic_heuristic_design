import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric graph-structural priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (raw + raw.T)
    distances = np.where(valid, np.maximum(symmetric, 0.0), np.inf)
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[valid]
    positive = finite_values[finite_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=float)
    row_indices = np.arange(n)[:, None]
    rank_positions = np.arange(n, dtype=float)
    ranks[row_indices, order] = rank_positions

    rank_denominator = float(max(n - 1, 1))
    row_quality = np.zeros((n, n), dtype=float)
    row_quality[valid] = 1.0 - ranks[valid] / rank_denominator

    mutual_rank_quality = np.sqrt(
        np.maximum(row_quality * row_quality.T, 0.0)
    )

    rank_agreement = np.zeros((n, n), dtype=float)
    rank_agreement[valid] = 1.0 - np.abs(
        ranks[valid] - ranks.T[valid]
    ) / rank_denominator
    rank_agreement = np.clip(rank_agreement, 0.0, 1.0)

    two_hop_bottleneck = np.full((n, n), np.inf, dtype=float)

    for intermediate in range(n):
        via_intermediate = np.maximum(
            distances[:, intermediate, None],
            distances[None, intermediate, :],
        )
        two_hop_bottleneck = np.minimum(
            two_hop_bottleneck,
            via_intermediate,
        )

    bypass_margin = np.zeros((n, n), dtype=float)
    finite_bypass = valid & np.isfinite(two_hop_bottleneck)

    bypass_margin[finite_bypass] = (
        two_hop_bottleneck[finite_bypass]
        - distances[finite_bypass]
    ) / np.maximum(
        two_hop_bottleneck[finite_bypass],
        distances[finite_bypass],
    ) + eps

    bypass_margin[valid & ~np.isfinite(two_hop_bottleneck)] = 1.0
    bypass_margin = np.clip(bypass_margin, 0.0, 1.0)

    global_quality = np.zeros((n, n), dtype=float)
    global_quality[valid] = 0.5 * (
        row_quality[valid] + row_quality.T[valid]
    )

    score = (
        0.42 * global_quality
        + 0.38 * bypass_margin
        + 0.20 * rank_agreement
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
