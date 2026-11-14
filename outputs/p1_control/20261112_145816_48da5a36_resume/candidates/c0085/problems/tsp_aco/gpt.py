import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    valid = positive & positive.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    global_scale = max(
        float(np.median(positive_values)),
        np.finfo(float).eps,
    )

    fallback = max(float(np.max(positive_values)), global_scale)
    d_work = np.where(finite, d, fallback * 10.0)
    d_work = np.maximum(d_work, np.finfo(float).eps)

    order = np.argsort(d_work, axis=1)
    sorted_distances = np.take_along_axis(d_work, order, axis=1)

    nearest_index = 0
    local_scale = sorted_distances[:, nearest_index]
    if n >= 4:
        local_scale = 0.5 * (
            sorted_distances[:, 0] + sorted_distances[:, 1]
        )

    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative_length = d_work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    inverse_signal = 1.0 / (1.0 + relative_length)
    global_signal = 1.0 / (1.0 + d_work / global_scale)

    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    rank_signal = np.exp(
        -2.5 * (ranks + ranks.T) / max(float(n - 1), 1.0)
    )

    nearest = order[:, 0]
    mutual_nearest = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    local_density = global_scale / (global_scale + local_scale)
    density_gap = np.abs(
        np.log((local_scale[:, None] + np.finfo(float).eps)
               / (local_scale[None, :] + np.finfo(float).eps))
    )
    bridge_floor = 0.35 * np.exp(-density_gap)

    score = (
        0.42 * inverse_signal
        + 0.24 * rank_signal
        + 0.18 * global_signal
        + 0.10 * mutual_nearest
        + 0.06 * bridge_floor
    )

    density_balance = 1.0 / (
        1.0
        + np.abs(local_density[:, None] - local_density[None, :])
    )
    score *= 0.85 + 0.15 * density_balance

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
