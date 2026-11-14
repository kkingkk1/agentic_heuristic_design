import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based and graph-diffusion priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    values = d[valid]
    positive_values = values[values > 0.0]

    if positive_values.size:
        distance_scale = float(np.median(positive_values))
    else:
        distance_scale = max(float(np.median(values)), 1.0)

    distance_scale = max(distance_scale, eps)

    normalized_distance = np.zeros_like(d)
    np.divide(
        d,
        distance_scale,
        out=normalized_distance,
        where=valid,
    )
    normalized_distance = np.minimum(normalized_distance, 1e12)

    direct_affinity = np.zeros_like(d)
    np.divide(
        1.0,
        1.0 + normalized_distance,
        out=direct_affinity,
        where=valid,
    )
    direct_affinity[~valid] = 0.0

    ranks = np.zeros((n, n), dtype=float)

    for row_index in range(n):
        order = np.argsort(
            np.where(valid[row_index], d[row_index], np.inf),
            kind="mergesort",
        )
        valid_order = order[valid[row_index, order]]
        ranks[row_index, valid_order] = np.arange(
            valid_order.size,
            dtype=float,
        )

    rank_affinity = np.zeros_like(d)
    np.divide(
        1.0,
        1.0 + ranks,
        out=rank_affinity,
        where=valid,
    )
    rank_affinity[~valid] = 0.0

    transition = np.zeros_like(d)
    row_mass = direct_affinity.sum(axis=1, keepdims=True)
    np.divide(
        direct_affinity,
        row_mass,
        out=transition,
        where=row_mass > 0.0,
    )

    diffusion = transition @ transition
    diffusion[~np.isfinite(diffusion)] = 0.0
    diffusion = np.maximum(diffusion, 0.0)

    diffusion_scale = diffusion.max(axis=1, keepdims=True)
    normalized_diffusion = np.zeros_like(diffusion)
    np.divide(
        diffusion,
        diffusion_scale,
        out=normalized_diffusion,
        where=diffusion_scale > 0.0,
    )

    score = (
        direct_affinity
        * (0.35 + 0.65 * rank_affinity)
        * (0.70 + 0.30 * normalized_diffusion)
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
