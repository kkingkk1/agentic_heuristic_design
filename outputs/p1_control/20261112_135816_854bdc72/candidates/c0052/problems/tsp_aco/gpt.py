import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.array(distances, dtype=float, copy=True)
    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid & (d > 0.0)]
    if positive.size:
        scale = np.median(positive)
    else:
        scale = 1.0
    scale = max(float(scale), np.finfo(float).eps)

    finite_values = d[valid]
    low = np.percentile(finite_values, 10.0)
    high = np.percentile(finite_values, 90.0)
    spread = max(float(high - low), scale * 0.05, np.finfo(float).eps)

    clipped = np.clip(d, low, high)
    normalized = (clipped - low) / spread
    inverse = 1.0 / (0.08 + normalized)
    inverse[~valid] = 0.0

    order = np.argsort(d, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(d, order, axis=1)
    finite_sorted = np.isfinite(sorted_distances)

    ranks = np.full((n, n), n, dtype=float)
    rank_values = np.broadcast_to(
        np.arange(n, dtype=float),
        (n, n),
    )
    ranks[np.arange(n)[:, None], order] = np.where(
        finite_sorted,
        rank_values,
        float(n),
    )
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))
    rank_score[~valid] = 0.0

    nearest_k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    nearest[rows, order[:, :nearest_k]] = finite_sorted[:, :nearest_k]
    np.fill_diagonal(nearest, 0.0)

    mutual = nearest * nearest.T
    one_sided = nearest + nearest.T

    first = np.take_along_axis(d, order[:, :2], axis=1)
    best = first[:, 0]
    second = first[:, 1]
    best = np.where(np.isfinite(best), best, scale)
    second = np.where(np.isfinite(second), second, best + scale)

    regret = np.maximum(second - best, 0.0) / (second + 0.05 * scale)
    regret_bonus = 1.0 + 0.35 * (
        regret[:, None] * nearest + regret[None, :] * nearest.T
    )

    structural = 1.0 + 0.85 * mutual + 0.20 * one_sided
    score = inverse * (0.72 + 0.28 * rank_score) * structural * regret_bonus
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    mean_score = score[valid].mean()
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score.astype(float, copy=False)
