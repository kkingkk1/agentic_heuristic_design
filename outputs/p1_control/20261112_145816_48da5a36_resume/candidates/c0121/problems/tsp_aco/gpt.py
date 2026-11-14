import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros_like(d, dtype=float)

    if n <= 1:
        return scores

    finite_offdiag = np.isfinite(d)
    np.fill_diagonal(finite_offdiag, False)

    positive = d[finite_offdiag & (d > 0)]
    if positive.size == 0:
        return scores

    fallback = float(np.max(positive) * 10.0)
    safe_d = np.where(
        finite_offdiag & (d > 0),
        d,
        fallback,
    )
    np.fill_diagonal(safe_d, fallback)

    k = max(1, min(n - 1, int(np.sqrt(n))))
    sorted_rows = np.sort(safe_d, axis=1)
    local_scale = np.mean(sorted_rows[:, 1:k + 1], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    global_scale = float(np.median(positive))
    global_scale = max(global_scale, np.finfo(float).eps)

    endpoint_scale = 0.5 * (
        local_scale[:, None] + local_scale[None, :]
    )

    relative_closeness = np.exp(
        -1.30 * safe_d / (endpoint_scale + np.finfo(float).eps)
    )

    mutual_preference = 1.0 / (
        1.0 + np.exp(
            np.clip(
                1.40 * (safe_d - endpoint_scale)
                / (0.20 * endpoint_scale + 1e-12),
                -50.0,
                50.0,
            )
        )
    )

    isolation = np.sqrt(
        np.outer(local_scale, local_scale)
    ) / (global_scale + np.finfo(float).eps)

    isolation_bonus = 1.0 + 0.22 * np.clip(isolation, 0.0, 3.0)

    scores = (
        relative_closeness
        * (0.30 + 0.70 * mutual_preference)
        * isolation_bonus
    )

    valid = finite_offdiag & (d > 0)
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
