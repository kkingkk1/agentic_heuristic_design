import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n < 2:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    finite_values = d[valid]
    fallback = float(np.median(finite_values)) if finite_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    clean = np.where(valid, d, fallback)
    np.fill_diagonal(clean, fallback)

    neighborhood_size = max(
        1,
        min(n - 1, int(np.ceil(np.sqrt(n))) + 1),
    )

    ordered = np.sort(
        np.where(valid, clean, np.inf),
        axis=1,
    )
    local_scale = np.median(
        ordered[:, :neighborhood_size],
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )

    scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    scale = np.maximum(scale, np.finfo(float).eps)

    distance_score = np.exp(-clean / scale)

    ranks = np.full((n, n), n, dtype=float)
    rank_temperature = max(1.0, 0.85 * np.sqrt(n))

    for i in range(n):
        row_order = np.argsort(clean[i], kind="stable")
        row_order = row_order[row_order != i]
        ranks[i, row_order] = np.arange(n - 1, dtype=float)

    rank_score = np.exp(-ranks / rank_temperature)

    nearest = np.argmin(
        np.where(valid, clean, np.inf),
        axis=1,
    )

    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and valid[i, j] and nearest[j] == i:
            mutual_bonus[i, j] = 1.0
            mutual_bonus[j, i] = 1.0

    score = (
        0.50 * distance_score
        + 0.32 * rank_score
        + 0.18 * mutual_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
