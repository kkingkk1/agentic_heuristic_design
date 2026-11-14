import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    valid_distances = d[positive]
    fallback = float(np.max(valid_distances))
    safe_d = np.where(positive, d, fallback)

    # Robust inverse-distance desirability.
    scale = float(np.median(valid_distances))
    scale = max(scale, np.finfo(float).eps)
    inverse = 1.0 / (safe_d + scale * 1e-6)

    # Rank-based neighborhood desirability, less sensitive to distance scale.
    order = np.argsort(safe_d, axis=1)
    rank = np.empty_like(order, dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-rank / max(1.0, np.sqrt(n)))

    # Require both endpoints to consider an edge promising.
    mutual_score = np.sqrt(rank_score * rank_score.T)

    # Favor short edges relative to the local scale of both endpoints.
    row_scale = np.median(
        np.where(np.isfinite(safe_d), safe_d, fallback), axis=1
    )
    local_scale = np.sqrt(
        np.maximum(row_scale[:, None], np.finfo(float).eps)
        * np.maximum(row_scale[None, :], np.finfo(float).eps)
    )
    local_shortness = np.exp(-safe_d / np.maximum(local_scale, np.finfo(float).eps))

    score = (
        0.55 * inverse / np.max(inverse)
        + 0.25 * mutual_score
        + 0.20 * local_shortness
    )

    # Enforce undirected-edge consistency and remove invalid/self edges.
    score = 0.5 * (score + score.T)
    score[~finite] = 0.0
    score[d <= 0] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
