import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    fallback = float(np.median(finite_positive)) if finite_positive.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    clean = np.where(np.isfinite(d) & (d > 0), d, fallback)
    np.fill_diagonal(clean, np.inf)

    order = np.argsort(clean, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    nearest = np.take_along_axis(clean, order[:, : min(3, n - 1)], axis=1)
    local_scale = np.median(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]

    distance_score = np.exp(
        -0.5 * (clean / row_scale + clean / col_scale)
    )

    rank_score = np.exp(
        -0.5 * (ranks / max(1.0, np.log2(n + 1)))
    )

    mutual_score = np.sqrt(
        np.exp(-ranks / max(1.0, np.log2(n + 1)))
        * np.exp(-ranks.T / max(1.0, np.log2(n + 1)))
    )

    second_best = np.partition(clean, 1, axis=1)[:, 1]
    urgency = np.clip(
        (second_best - local_scale) / np.maximum(second_best, fallback),
        0.0,
        1.0,
    )

    endpoint_pressure = np.sqrt(
        (1.0 + urgency[:, None]) * (1.0 + urgency[None, :])
    )

    directed_score = (
        0.50 * distance_score
        + 0.25 * rank_score
        + 0.15 * mutual_score
        + 0.10 * endpoint_pressure * distance_score
    )

    score = 0.5 * (directed_score + directed_score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if not np.isfinite(maximum) or maximum <= 0:
        return np.zeros((n, n), dtype=float)

    return score / maximum
