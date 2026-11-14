import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw) & (raw >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (raw + raw.T)
    d = np.where(valid, np.maximum(d, 0.0), np.inf)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    if positive.size:
        scale = max(float(np.median(positive)), eps)
    else:
        scale = 1.0

    finite_values = d[valid]
    largest = max(float(np.max(finite_values)), scale, eps)

    safe = np.where(valid, d, np.inf)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized = np.minimum(safe / scale, 1e6)

        short_edge = 1.0 / (1.0 + normalized)

        affinity = np.exp(-normalized)
        affinity[~valid] = 0.0
        np.fill_diagonal(affinity, 0.0)

        node_support = np.sum(affinity, axis=1)
        support_product = np.sqrt(
            np.maximum(node_support[:, None] * node_support[None, :], eps)
        )
        shared_support = (affinity @ affinity) / support_product
        shared_support = np.clip(shared_support, 0.0, 1.0)

        two_hop = np.min(
            safe[:, :, None] + safe[None, :, :],
            axis=1,
        )
        two_hop = np.where(
            np.isfinite(two_hop),
            two_hop,
            largest * 2.0,
        )

        direct_ratio = d / np.maximum(two_hop, eps)
        detour_gain = np.clip(1.0 - direct_ratio, 0.0, 1.0)

        local_mass = np.sum(affinity, axis=1)
        mass_ratio = np.minimum(
            local_mass[:, None],
            local_mass[None, :],
        ) / np.maximum(
            np.maximum(local_mass[:, None], local_mass[None, :]),
            eps,
        )

        score = (
            0.46 * short_edge
            + 0.30 * detour_gain
            + 0.16 * shared_support
            + 0.08 * mass_ratio
        )

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    for _ in range(4):
        row_mass = np.sum(score, axis=1)
        correction = np.sqrt(
            np.maximum(row_mass[:, None] * row_mass[None, :], eps)
        )
        score = np.divide(
            score,
            correction,
            out=np.zeros_like(score),
            where=correction > 0.0,
        )
        score = 0.5 * (score + score.T)
        score[~valid] = 0.0
        np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
