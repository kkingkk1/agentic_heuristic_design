import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        d,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
        copy=True,
    )
    d += d.T
    d *= 0.5
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    local_input = np.where(valid, d, np.inf)
    local_scale = np.partition(
        local_input,
        kth=min(1, n - 2),
        axis=1,
    )[:, min(1, n - 2)]
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    invalid_distance = fallback_distance * 10.0
    safe_distance = np.where(valid, d, invalid_distance)
    safe_distance = np.maximum(safe_distance, eps)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scale_product = np.sqrt(
            local_scale[:, None] * local_scale[None, :]
        )

        score = np.exp(-1.10 * safe_distance / scale_product)

        order = np.argsort(
            safe_distance,
            axis=1,
            kind="stable",
        )
        rank = np.empty((n, n), dtype=float)
        np.put_along_axis(
            rank,
            order,
            np.arange(n, dtype=float)[None, :],
            axis=1,
        )
        rank /= max(n - 1, 1)

        rank += rank.T
        score *= np.exp(-3.20 * rank)

        score *= 0.62 + 0.38 * np.exp(
            -0.26 * safe_distance / global_scale
        )

        local_i = local_scale[:, None]
        local_j = local_scale[None, :]
        scale_sum = local_i + local_j
        scale_difference = np.abs(local_i - local_j)

        balance = 1.0 / (
            1.0 + 0.85 * scale_difference / (scale_sum + eps)
        )
        score *= 0.78 + 0.22 * balance

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
