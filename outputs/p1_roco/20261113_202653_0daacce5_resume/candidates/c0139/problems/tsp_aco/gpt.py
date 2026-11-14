import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded stochastic edge-selection priors for TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & np.isfinite(raw.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.where(valid, np.maximum(raw, 0.0), 0.0)
    distances = 0.5 * (distances + distances.T)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    normalized = np.clip(distances / scale, 0.0, 12.0)

    # Smooth distance preference; clipping prevents extreme edges from
    # dominating numerical behavior.
    distance_affinity = np.exp(-normalized / 1.25)
    distance_affinity[~valid] = 0.0
    np.fill_diagonal(distance_affinity, 0.0)

    # Soft local rank. Ties receive the same rank, and no finite edge is
    # removed from consideration.
    rank = np.sum(
        (distances[:, :, None] > distances[:, None, :])
        & valid[:, :, None]
        & valid[:, None, :],
        axis=2,
    ).astype(float)

    rank_temperature = max(1.0, 0.35 * np.sqrt(float(n)))
    rank_affinity = np.exp(-rank / rank_temperature)
    rank_affinity[~valid] = 0.0
    np.fill_diagonal(rank_affinity, 0.0)

    # Long-term potential: favor endpoints that have several accessible
    # neighbors, without assigning a large reward to any single detour.
    row_mass = distance_affinity.sum(axis=1)
    row_mass /= max(float(np.max(row_mass)), np.finfo(float).eps)
    node_potential = np.sqrt(row_mass[:, None] * row_mass[None, :])
    node_potential[~valid] = 0.0

    # Small reproducible symmetric perturbation for stable tie breaking.
    indices = np.arange(n, dtype=float)
    phase = (
        0.754877666 * (indices[:, None] + indices[None, :])
        + 0.569840291 * (indices[:, None] * indices[None, :])
    )
    perturbation = 0.5 + 0.5 * np.sin(phase)
    perturbation = np.clip(perturbation, 0.0, 1.0)
    perturbation[~valid] = 0.0
    np.fill_diagonal(perturbation, 0.0)

    distance_affinity /= max(
        float(np.max(distance_affinity)), np.finfo(float).eps
    )
    rank_affinity /= max(
        float(np.max(rank_affinity)), np.finfo(float).eps
    )

    score = (
        0.58 * distance_affinity
        + 0.24 * rank_affinity
        + 0.13 * node_potential
        + 0.05 * perturbation
    )

    # Keep a nonzero floor for every finite edge.
    floor = 0.035
    score = floor + (1.0 - floor) * score
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    # Mild balancing reduces endpoint starvation while preserving the main
    # distance and rank signals.
    balanced = score.copy()
    for _ in range(8):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.82 * score + 0.18 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
