import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diverse, globally balanced edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & np.isfinite(raw.T)
        & (raw.T >= 0.0)
        & ~diagonal
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    upper = float(np.quantile(values, 0.95))
    penalty = max(upper, scale) + 10.0 * scale

    clean = np.where(valid, distances, penalty)
    np.fill_diagonal(clean, penalty)

    temperature = max(0.58 * scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(clean[valid] / temperature, 50.0)
    )

    row_mass = affinity.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)

    # Mild rank preference, kept separate from distance affinity.
    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_signal = np.exp(-ranks / rank_scale)
    rank_signal[~valid] = 0.0

    # Remaining high-quality choices after using edge (i, j).
    remaining_i = np.maximum(row_mass[:, None] - affinity, 0.0)
    future_capacity = np.sqrt(
        remaining_i * remaining_i.T
    )
    capacity_scale = max(float(np.quantile(row_mass, 0.5)), eps)
    future_capacity = np.clip(
        future_capacity / capacity_scale,
        0.0,
        2.0,
    )
    future_capacity[~valid] = 0.0

    # Two-step bypass: a small bridge bonus for edges with few good detours.
    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via = (
            clean[:, start:stop, None]
            + clean[None, start:stop, :]
        )
        bypass = np.minimum(bypass, np.min(via, axis=1))

    np.fill_diagonal(bypass, penalty)
    advantage = np.maximum(bypass - clean, 0.0)
    bridge_signal = 1.0 - np.exp(
        -advantage / max(1.15 * scale, eps)
    )
    bridge_signal[~valid] = 0.0

    affinity_norm = affinity / np.maximum(
        float(affinity.max()),
        eps,
    )
    rank_pair = np.sqrt(
        np.maximum(rank_signal * rank_signal.T, 0.0)
    )

    # Additive composition avoids repeated multiplicative amplification.
    score = (
        0.52 * affinity_norm
        + 0.14 * rank_pair
        + 0.24 * np.minimum(future_capacity, 1.0)
        + 0.10 * bridge_signal
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)
    score /= maximum

    # Symmetric balancing spreads probability mass across nodes and regions.
    balanced = score.copy()
    for _ in range(24):
        row_sum = balanced.sum(axis=1)
        scale_vector = 1.0 / np.sqrt(
            np.maximum(row_sum, eps)
        )
        balanced *= scale_vector[:, None] * scale_vector[None, :]
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

        peak = float(balanced.max())
        if np.isfinite(peak) and peak > 0.0:
            balanced /= peak

    result = 0.68 * score + 0.32 * balanced
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
