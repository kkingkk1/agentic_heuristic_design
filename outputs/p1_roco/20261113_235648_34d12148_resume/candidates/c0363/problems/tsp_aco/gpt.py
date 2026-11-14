import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse and long-term-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(raw)
        & np.isfinite(raw.T)
        & (raw >= 0.0)
        & (raw.T >= 0.0)
        & ~diagonal
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    upper = float(np.quantile(observed, 0.90))
    penalty = max(upper, scale) + 8.0 * scale

    clean = np.where(valid, distances, penalty)
    np.fill_diagonal(clean, penalty)

    spread = float(np.median(np.abs(observed - scale)))
    if not np.isfinite(spread):
        spread = 0.0

    temperature = max(
        0.52 * scale + 0.35 * spread,
        eps,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(clean[valid] / temperature, 50.0)
    )

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_signal = np.exp(-ranks / rank_scale)
    rank_signal[~valid] = 0.0

    row_mass = affinity.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)

    probabilities = affinity / row_mass[:, None]
    concentration = np.sum(probabilities * probabilities, axis=1)
    flexibility = 1.0 - np.sqrt(np.clip(concentration, 0.0, 1.0))
    flexibility = np.clip(flexibility, 0.05, 1.0)

    remaining = np.maximum(
        row_mass[:, None] - affinity,
        0.0,
    )
    remaining_capacity = np.sqrt(
        remaining * remaining.T
    )
    capacity_scale = max(
        float(np.median(row_mass)),
        eps,
    )
    capacity_signal = np.clip(
        remaining_capacity / capacity_scale,
        0.0,
        1.5,
    )
    capacity_signal[~valid] = 0.0

    reserve_signal = np.sqrt(
        flexibility[:, None] * flexibility[None, :]
    )
    reserve_signal[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(
        1,
        min(n, 2_000_000 // max(n, 1)),
    )

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via = (
            clean[:, start:stop, None]
            + clean[None, start:stop, :]
        )
        bypass = np.minimum(
            bypass,
            np.min(via, axis=1),
        )

    np.fill_diagonal(bypass, penalty)

    replacement_gap = np.maximum(
        bypass - clean,
        0.0,
    )
    bridge_signal = 1.0 - np.exp(
        -replacement_gap / max(1.25 * scale, eps)
    )
    bridge_signal[~valid] = 0.0

    affinity_norm = affinity / max(
        float(affinity.max()),
        eps,
    )
    rank_pair = np.sqrt(
        np.maximum(
            rank_signal * rank_signal.T,
            0.0,
        )
    )

    base = (
        0.56 * affinity_norm
        + 0.14 * rank_pair
        + 0.18 * np.minimum(capacity_signal, 1.0)
        + 0.07 * reserve_signal
        + 0.05 * bridge_signal
    )

    base[~valid] = 0.0
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)

    maximum = float(base.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)
    base /= maximum

    balanced = base.copy()

    for _ in range(8):
        row_sum = balanced.sum(axis=1)
        row_sum = np.maximum(row_sum, eps)

        node_scale = 1.0 / np.sqrt(row_sum)
        correction = np.sqrt(
            node_scale[:, None] * node_scale[None, :]
        )

        candidate = balanced * correction
        candidate[~valid] = 0.0
        np.fill_diagonal(candidate, 0.0)

        balanced = (
            0.72 * balanced
            + 0.28 * candidate
        )

        peak = float(balanced.max())
        if np.isfinite(peak) and peak > 0.0:
            balanced /= peak

    result = (
        0.74 * base
        + 0.26 * balanced
    )

    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
