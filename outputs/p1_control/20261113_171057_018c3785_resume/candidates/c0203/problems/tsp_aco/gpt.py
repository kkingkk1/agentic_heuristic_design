import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, locally informed priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    tiny = np.finfo(float).eps
    scale = max(scale, tiny)

    safe = np.where(valid, d, np.inf)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_second = np.partition(safe, kth=1, axis=1)[:, 1]
    col_second = np.partition(safe, kth=1, axis=0)[1, :]

    row_gap = np.where(
        np.isfinite(row_second),
        np.maximum(row_second - row_min, 0.0),
        0.0,
    )
    col_gap = np.where(
        np.isfinite(col_second),
        np.maximum(col_second - col_min, 0.0),
        0.0,
    )

    row_gap_values = row_gap[row_gap > 0.0]
    col_gap_values = col_gap[col_gap > 0.0]

    row_gap_scale = (
        float(np.median(row_gap_values)) if row_gap_values.size else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values)) if col_gap_values.size else scale
    )
    row_gap_scale = max(row_gap_scale, tiny)
    col_gap_scale = max(col_gap_scale, tiny)

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)

    temperature = max(0.28 * scale, tiny)
    row_gibbs = np.exp(-np.clip(row_delta / temperature, 0.0, 50.0))
    col_gibbs = np.exp(-np.clip(col_delta / temperature, 0.0, 50.0))

    row_regret = np.exp(-np.clip(row_delta / row_gap_scale, 0.0, 40.0))
    col_regret = np.exp(-np.clip(col_delta / col_gap_scale, 0.0, 40.0))
    regret = np.sqrt(row_regret * col_regret)

    row_nearest = valid & (d <= row_min[:, None] + tiny)
    col_nearest = valid & (d <= col_min[None, :] + tiny)
    mutual_nearest = row_nearest & col_nearest

    local_affinity = np.sqrt(row_gibbs * col_gibbs)
    local_affinity *= np.power(regret + tiny, 0.55)
    local_affinity *= np.where(row_nearest, 1.08, 1.0)
    local_affinity *= np.where(mutual_nearest, 1.22, 1.0)
    local_affinity[~valid] = 0.0
    local_affinity[~np.isfinite(local_affinity)] = 0.0

    balanced = local_affinity.copy()
    for _ in range(16):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

    balanced[~valid] = 0.0

    row_mass = balanced.sum(axis=1)
    row_entropy = np.zeros(n, dtype=float)
    for i in range(n):
        if row_mass[i] <= 0.0:
            continue
        probabilities = balanced[i] / row_mass[i]
        probabilities = probabilities[probabilities > 0.0]
        row_entropy[i] = -float(np.sum(probabilities * np.log(probabilities)))

    degrees = valid.sum(axis=1).astype(float)
    normalized_entropy = np.divide(
        row_entropy,
        np.log(np.maximum(degrees, 2.0)),
        out=np.zeros_like(row_entropy),
        where=degrees > 0.0,
    )
    concentration = 1.0 - float(np.mean(np.clip(normalized_entropy, 0.0, 1.0)))
    floor_weight = float(np.clip(0.035 + 0.18 * concentration, 0.035, 0.215))

    uniform = np.divide(
        valid.astype(float),
        degrees[:, None],
        out=np.zeros_like(balanced),
        where=degrees[:, None] > 0.0,
    )
    score = (1.0 - floor_weight) * balanced + floor_weight * uniform
    score[~valid] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
