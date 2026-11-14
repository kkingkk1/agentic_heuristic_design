import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d > 0)
    np.fill_diagonal(valid, False)

    safe_d = np.where(valid, d, np.nan)

    local_scale = np.nanmedian(safe_d, axis=1)
    global_scale = np.nanmedian(safe_d)

    if not np.isfinite(global_scale) or global_scale <= 0:
        return np.zeros_like(d, dtype=float)

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]

    normalized_distance = d / np.sqrt(row_scale * col_scale)
    inverse_distance = np.zeros_like(d, dtype=float)
    inverse_distance[valid] = 1.0 / (1.0 + normalized_distance[valid])

    nearest = np.full(n, np.inf, dtype=float)
    nearest[valid.any(axis=1)] = np.nanmin(safe_d, axis=1)[valid.any(axis=1)]

    second_nearest = nearest.copy()
    for i in range(n):
        values = d[i, valid[i]]
        if values.size >= 2:
            ordered = np.partition(values, 1)
            second_nearest[i] = ordered[1]

    mutual_affinity = np.zeros_like(d, dtype=float)
    for i in range(n):
        for j in range(n):
            if valid[i, j]:
                ni = nearest[i]
                nj = nearest[j]
                mutual_affinity[i, j] = np.exp(
                    -0.5
                    * (
                        max(d[i, j] - ni, 0.0) / max(ni, global_scale)
                        + max(d[i, j] - nj, 0.0) / max(nj, global_scale)
                    )
                )

    regret = np.zeros_like(d, dtype=float)
    for i in range(n):
        if np.isfinite(second_nearest[i]) and second_nearest[i] > 0:
            regret[i, valid[i]] = np.clip(
                (second_nearest[i] - d[i, valid[i]])
                / second_nearest[i],
                0.0,
                1.0,
            )

    regret_pair = 0.5 * (regret + regret.T)

    score = (
        0.55 * inverse_distance
        + 0.30 * mutual_affinity
        + 0.15 * regret_pair
    )

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return score
