import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structural edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances)
    valid &= distances >= 0.0
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    d = np.where(valid, distances, np.inf)
    d = np.minimum(d, d.T)
    valid = np.isfinite(d) & ~np.eye(n, dtype=bool)

    positive = d[valid & (d > 0.0)]
    if positive.size:
        quantiles = np.quantile(positive, [0.25, 0.50, 0.75])
        scales = np.maximum(quantiles, np.finfo(float).eps)
    else:
        scales = np.ones(3, dtype=float)

    affinity = np.zeros((n, n), dtype=float)

    for scale, weight in zip(scales, (0.25, 0.35, 0.40)):
        kernel = np.exp(-np.minimum(d / scale, 60.0))
        kernel[~valid] = 0.0
        affinity += weight * kernel

    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)

    degree = affinity.sum(axis=1)
    safe_degree = np.maximum(degree, np.finfo(float).eps)

    normalized_affinity = affinity / np.sqrt(
        safe_degree[:, None] * safe_degree[None, :]
    )

    two_hop_support = normalized_affinity @ normalized_affinity
    two_hop_support[~valid] = 0.0
    two_hop_support = np.maximum(two_hop_support, 0.0)

    support_scale = np.quantile(
        two_hop_support[valid],
        0.75,
    ) if np.any(valid) else 1.0
    support_scale = max(float(support_scale), np.finfo(float).eps)
    support_term = np.tanh(two_hop_support / support_scale)

    endpoint_balance = np.sqrt(
        np.minimum(
            affinity.sum(axis=1)[:, None],
            affinity.sum(axis=1)[None, :],
        )
        / np.maximum(
            np.maximum(
                affinity.sum(axis=1)[:, None],
                affinity.sum(axis=1)[None, :],
            ),
            np.finfo(float).eps,
        )
    )

    score = affinity * (
        0.55
        + 0.30 * support_term
        + 0.15 * endpoint_balance
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
