import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(finite_positive)) if finite_positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    x = np.array(d, dtype=float, copy=True)
    x[~np.isfinite(x)] = scale * 1e6
    x = np.maximum(x, 0.0)
    x = 0.5 * (x + x.T)
    np.fill_diagonal(x, np.inf)

    # Adaptive closeness: robust to distance matrices with different scales.
    closeness = np.exp(-x / scale)
    np.fill_diagonal(closeness, 0.0)

    # Mutual preference: an edge is favored when both endpoints rank it highly.
    row_sum = closeness.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, np.finfo(float).eps)
    preference = closeness / row_sum
    mutual_preference = np.sqrt(preference * preference.T)

    # Exclusivity: emphasize edges that are much better than local alternatives.
    row_best = np.max(closeness, axis=1, keepdims=True)
    row_best = np.maximum(row_best, np.finfo(float).eps)
    exclusivity = np.sqrt(
        (closeness / row_best) * (closeness.T / row_best.T)
    )

    # Two-hop coherence measures whether the endpoints share short compatible routes.
    two_hop = closeness @ closeness
    if np.max(two_hop) > 0:
        two_hop = two_hop / np.max(two_hop)

    score = (
        0.45 * closeness
        + 0.35 * mutual_preference
        + 0.15 * exclusivity
        + 0.05 * two_hop
    )

    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    # Row normalization keeps the output usable as sampling probabilities.
    row_sum = score.sum(axis=1, keepdims=True)
    valid = row_sum[:, 0] > 0
    score[valid] /= row_sum[valid]

    if not np.all(valid):
        score[~valid] = 1.0 / max(n - 1, 1)
        np.fill_diagonal(score, 0.0)

    return score
