import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized stochastic edge-priority weights for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    ordered = np.sort(work, axis=1)
    nearest = ordered[:, 0]

    if n > 2:
        second = ordered[:, 1]
    else:
        second = nearest * 2.0

    finite_nearest = np.isfinite(nearest)
    nearest_safe = np.where(finite_nearest, nearest, 1.0)
    second_safe = np.where(np.isfinite(second), second, nearest_safe * 2.0)

    local_scale = 0.7 * nearest_safe + 0.3 * second_safe
    local_scale = np.maximum(local_scale, eps)

    regret = np.maximum(second_safe - nearest_safe, 0.0)
    urgency = regret / np.maximum(second_safe, eps)
    urgency = np.clip(urgency, 0.0, 1.0)
    urgency *= finite_nearest.astype(float)

    row_ratio = d / local_scale[:, None]
    col_ratio = d / local_scale[None, :]

    row_quality = 1.0 / (1.0 + np.maximum(row_ratio, 0.0))
    col_quality = 1.0 / (1.0 + np.maximum(col_ratio, 0.0))

    mutual_quality = np.sqrt(row_quality * col_quality)

    row_rank = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    col_rank = np.argsort(np.argsort(work, axis=0), axis=0).astype(float)

    rank_scale = max(n - 2, 1)
    rank_quality = np.exp(
        -0.9 * (row_rank + col_rank) / rank_scale
    )

    distance_scale = np.sqrt(
        np.maximum(local_scale[:, None], eps)
        * np.maximum(local_scale[None, :], eps)
    )
    relative_distance = np.divide(
        d,
        distance_scale,
        out=np.full_like(d, np.inf),
        where=distance_scale > 0,
    )

    distance_quality = 1.0 / (1.0 + np.maximum(relative_distance, 0.0))

    endpoint_boost = (
        1.0
        + 0.8 * (urgency[:, None] + urgency[None, :])
        + 0.5 * urgency[:, None] * urgency[None, :]
    )

    weak_endpoint_penalty = np.power(
        np.maximum(mutual_quality, eps),
        0.75,
    )

    score = (
        distance_quality
        * rank_quality
        * weak_endpoint_penalty
        * endpoint_boost
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_totals = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0.0,
    )

    return score
