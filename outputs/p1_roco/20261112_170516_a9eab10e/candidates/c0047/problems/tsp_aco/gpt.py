import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite_positive = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(finite_positive, False)

    valid = finite_positive & finite_positive.T
    if not np.any(valid):
        return scores

    symmetric_distances = 0.5 * (distances + distances.T)
    safe_distances = np.where(valid, symmetric_distances, np.inf)

    positive_values = safe_distances[valid]
    global_scale = float(np.median(positive_values))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = float(np.mean(positive_values))

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    ordered = np.sort(safe_distances, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else nearest

    nearest = np.where(np.isfinite(nearest), nearest, global_scale)
    second = np.where(np.isfinite(second), second, nearest)

    local_scale = np.where(
        np.isfinite(ordered).any(axis=1),
        np.nanmedian(
            np.where(np.isfinite(safe_distances), safe_distances, np.nan),
            axis=1,
        ),
        global_scale,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    relative_distance = safe_distances / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    visibility = 1.0 / (1.0 + relative_distance)
    neighborhood = np.exp(-np.clip(relative_distance - 1.0, 0.0, 60.0))

    ranks = np.full((n, n), n + 1.0, dtype=float)
    for i in range(n):
        valid_indices = np.flatnonzero(valid[i])
        if valid_indices.size:
            order = valid_indices[
                np.argsort(safe_distances[i, valid_indices], kind="stable")
            ]
            ranks[i, order] = np.arange(1, order.size + 1, dtype=float)

    rank_strength = np.exp(
        -(ranks - 1.0) / max(1.0, np.sqrt(float(n)))
    )
    rank_strength[~valid] = 0.0

    nearest_gap = np.maximum(second - nearest, 0.0)
    node_regret = nearest_gap / np.maximum(second, eps)
    node_regret = np.clip(node_regret, 0.0, 1.0)

    endpoint_regret = 1.0 + 0.35 * (
        node_regret[:, None] + node_regret[None, :]
    )

    mutual_rank = np.sqrt(rank_strength * rank_strength.T)
    mutual_bonus = 0.75 + 0.5 * mutual_rank

    score = (
        visibility
        * neighborhood
        * endpoint_regret
        * mutual_bonus
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
