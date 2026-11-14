import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    safe = np.where(valid, d, np.inf)
    eps = np.finfo(float).eps

    positive = safe[valid & (safe > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    sorted_rows = np.sort(safe, axis=1)
    k = max(1, min(n - 1, int(np.sqrt(n))))

    local_neighbors = sorted_rows[:, :k]
    finite_neighbors = np.where(
        np.isfinite(local_neighbors),
        local_neighbors,
        global_scale,
    )
    local_scale = np.mean(finite_neighbors, axis=1)
    local_scale = np.maximum(local_scale, global_scale * 1e-3)

    nearest = sorted_rows[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    if n > 2:
        second = sorted_rows[:, 1]
        second = np.where(np.isfinite(second), second, nearest)
    else:
        second = nearest

    margin = np.maximum(second - nearest, 0.0)
    margin_scale = np.maximum(margin, 0.1 * local_scale)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(safe, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = rank_values
    ranks[~valid] = np.inf

    local_scale_i = local_scale[:, None]
    local_scale_j = local_scale[None, :]

    local_attraction = np.exp(
        -np.minimum(
            safe / (local_scale_i + eps),
            50.0,
        )
    )

    global_attraction = np.exp(
        -np.minimum(safe / global_scale, 50.0)
    )

    rank_support = 1.0 / np.sqrt(
        np.maximum(ranks, 1.0) * np.maximum(ranks.T, 1.0)
    )

    endpoint_margin_i = np.exp(
        -np.minimum(
            np.maximum(safe - nearest[:, None], 0.0)
            / (margin_scale[:, None] + eps),
            50.0,
        )
    )
    endpoint_margin_j = np.exp(
        -np.minimum(
            np.maximum(safe - nearest[None, :], 0.0)
            / (margin_scale[None, :] + eps),
            50.0,
        )
    )
    mutual_margin = np.sqrt(endpoint_margin_i * endpoint_margin_j)

    mutual_local = np.sqrt(local_attraction * local_attraction.T)

    score = (
        mutual_local
        * (0.55 + 0.45 * rank_support)
        * (0.70 + 0.30 * mutual_margin)
        * (0.75 + 0.25 * global_attraction)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
