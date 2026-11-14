import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = np.isfinite(d) & (d > 0.0)
    valid_edges = positive & positive.T

    if not np.any(valid_edges):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[positive]

    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(positive, d, np.inf)

    sorted_edges = np.sort(safe, axis=1)
    second_index = min(1, n - 2)
    local_scale = sorted_edges[:, second_index]

    finite_local = np.isfinite(local_scale)
    if np.any(~finite_local):
        for i in np.flatnonzero(~finite_local):
            values = d[i, positive[i]]
            local_scale[i] = (
                float(np.median(values))
                if values.size
                else fallback_distance
            )

    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    relative_length = safe_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= max(n - 1, 1)

    mutual_rank = rank + rank.T
    nearest_rank = np.minimum(rank, rank.T)

    distance_signal = 0.24 + 0.76 * np.exp(-0.62 * relative_length)
    rank_signal = 0.28 + 0.72 * np.exp(-2.15 * mutual_rank)

    global_signal = 0.36 + 0.64 * np.exp(
        -0.20 * safe_distance / global_scale
    )

    candidate_width = max(2.0, min(8.0, np.sqrt(float(n))))
    retention_signal = 0.68 + 0.32 * np.exp(
        -2.0 * nearest_rank / candidate_width
    )

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None:]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    balance = 1.0 / (
        1.0 + 0.55 * scale_difference / (scale_sum + eps)
    )
    density_signal = 0.90 + 0.10 * balance

    score = (
        np.power(distance_signal, 0.38)
        * np.power(rank_signal, 0.34)
        * np.power(global_signal, 0.12)
        * np.power(retention_signal, 0.10)
        * density_signal
    )

    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
