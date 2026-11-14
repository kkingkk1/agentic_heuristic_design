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

    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    fallback = float(np.max(positive_values))
    fallback = max(fallback, np.finfo(float).eps)

    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, np.finfo(float).eps)

    # Use partial selection instead of sorting every row.
    kth = min(2, n - 2)
    nearest = np.partition(safe_d, kth=kth, axis=1)[:, kth]
    nearest = np.maximum(nearest, np.finfo(float).eps)

    endpoint_scale = np.sqrt(nearest[:, None] * nearest[None, :])
    relative_length = safe_d / endpoint_scale

    # Short edges receive high scores while retaining useful contrast.
    length_preference = 1.0 / (1.0 + relative_length) ** 2

    # Favor edges whose endpoints have comparable local distance scales.
    scale_balance = 1.0 - (
        np.abs(nearest[:, None] - nearest[None, :])
        / (nearest[:, None] + nearest[None, :] + np.finfo(float).eps)
    )
    scale_balance = np.clip(scale_balance, 0.0, 1.0)

    score = length_preference * scale_balance
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
