import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    x = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf).copy()
    x = np.maximum(x, 0.0)
    np.fill_diagonal(x, np.inf)

    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    positive_values = x[finite]
    scale = np.median(positive_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(positive_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    normalized = x / scale

    row_sorted = np.sort(normalized, axis=1)
    nearest = row_sorted[:, 0]
    local_scale = np.median(row_sorted[:, :max(1, min(5, n - 1))], axis=1)
    local_scale = np.maximum(local_scale, 1e-12)

    distance_signal = np.exp(-normalized)

    row_rank = np.empty_like(normalized)
    for i in range(n):
        order = np.argsort(normalized[i], kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        row_rank[i] = ranks / max(1.0, n - 2.0)

    mutual_rank = 1.0 - 0.5 * (row_rank + row_rank.T)
    mutual_rank = np.clip(mutual_rank, 0.0, 1.0)

    endpoint_scarcity = np.exp(
        -0.5 * (
            normalized / local_scale[:, None]
            + normalized / local_scale[None, :]
        )
    )

    two_step_best = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        candidate = normalized[:, k, None] + normalized[None, k, :]
        two_step_best = np.minimum(two_step_best, candidate)

    triangle_gain = two_step_best - normalized
    triangle_signal = 1.0 / (1.0 + np.exp(-np.clip(triangle_gain, -30.0, 30.0)))

    score = (
        0.50 * distance_signal
        + 0.25 * mutual_rank
        + 0.15 * endpoint_scarcity
        + 0.10 * triangle_signal
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float)
