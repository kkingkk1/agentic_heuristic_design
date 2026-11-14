import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic edge-priority indicators for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)

    ordered = np.sort(work, axis=1)
    nearest = ordered[:, 0]

    if n > 2:
        second = ordered[:, 1]
    else:
        second = nearest * 1.5

    finite_nearest = np.isfinite(nearest)
    nearest_safe = np.where(finite_nearest, nearest, 1.0)
    second_safe = np.where(
        np.isfinite(second),
        second,
        nearest_safe * 1.5,
    )

    eps = np.finfo(float).eps
    local_scale = np.maximum(
        0.5 * nearest_safe + 0.5 * second_safe,
        eps,
    )

    regret = np.maximum(second_safe - nearest_safe, 0.0)
    regret_ratio = regret / np.maximum(second_safe, eps)
    regret_ratio = np.clip(regret_ratio, 0.0, 1.0)

    relative_row = d / local_scale[:, None]
    relative_col = d / local_scale[None, :]

    row_quality = np.exp(-np.clip(relative_row, 0.0, 40.0))
    col_quality = np.exp(-np.clip(relative_col, 0.0, 40.0))

    mutual_quality = np.sqrt(row_quality * col_quality)

    endpoint_regret = (
        1.0
        + 0.75 * regret_ratio[:, None]
        + 0.75 * regret_ratio[None, :]
    )

    row_rank = np.divide(
        nearest_safe[:, None],
        np.maximum(d, eps),
    )
    col_rank = np.divide(
        nearest_safe[None, :],
        np.maximum(d, eps),
    )

    rank_bonus = np.sqrt(
        np.clip(row_rank, 0.0, 1.0) * np.clip(col_rank, 0.0, 1.0)
    )

    distance_scale = np.sqrt(
        np.maximum(local_scale[:, None], eps)
        * np.maximum(local_scale[None, :], eps)
    )

    normalized_distance = np.divide(
        d,
        distance_scale,
        out=np.full_like(d, np.inf),
        where=distance_scale > 0,
    )

    distance_term = np.exp(-np.clip(normalized_distance, 0.0, 40.0))

    score = (
        distance_term
        * mutual_quality
        * (0.55 + 0.45 * rank_bonus)
        * endpoint_regret
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0,
    )
