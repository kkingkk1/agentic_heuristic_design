import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    valid = finite_nonnegative & finite_nonnegative.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.zeros((n, n), dtype=float)
    symmetric[valid] = 0.5 * (
        distances[valid] + distances.T[valid]
    )
    np.fill_diagonal(symmetric, np.inf)

    positive = symmetric[valid & (symmetric > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    safe = np.where(valid, symmetric, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + eps)

    row_peak = np.max(inverse, axis=1, keepdims=True)
    row_peak = np.maximum(row_peak, eps)
    row_preference = inverse / row_peak

    exponential = np.zeros((n, n), dtype=float)
    exponential[valid] = np.exp(
        -safe[valid] / max(scale, eps)
    )

    row_mass = exponential.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)
    local_preference = exponential / row_mass[:, None]

    mutual_preference = np.sqrt(
        np.maximum(row_preference * row_preference.T, 0.0)
    )
    mutual_locality = np.sqrt(
        np.maximum(local_preference * local_preference.T, 0.0)
    )

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), n, dtype=float)
    rank_values = np.arange(n, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = rank_values

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])

    mutual_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )

    k = min(3, n - 1)
    top_values = np.partition(
        np.where(valid, inverse, 0.0),
        kth=n - k,
        axis=1,
    )[:, n - k:]

    node_potential = top_values.mean(axis=1)
    node_potential /= max(float(np.max(node_potential)), eps)

    endpoint_potential = np.sqrt(
        np.maximum(
            node_potential[:, None] * node_potential[None, :],
            0.0,
        )
    )

    score = (
        0.40 * mutual_preference
        + 0.25 * mutual_locality
        + 0.20 * mutual_rank
        + 0.15 * endpoint_potential
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
