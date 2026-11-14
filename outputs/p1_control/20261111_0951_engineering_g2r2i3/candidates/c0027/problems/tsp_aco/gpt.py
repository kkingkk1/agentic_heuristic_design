import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    eps = np.finfo(float).eps
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    safe_d = np.where(valid, np.maximum(d, 0.0), 0.0)
    scaled = safe_d / global_scale

    utility = np.exp(-0.5 * np.minimum(scaled * scaled, 80.0))
    utility[~valid] = 0.0

    row_mass = np.sum(utility, axis=1)
    col_mass = np.sum(utility, axis=0)

    row_mass = np.maximum(row_mass, eps)
    col_mass = np.maximum(col_mass, eps)

    continuation = np.sqrt(
        row_mass[:, None] * col_mass[None, :]
    )
    continuation /= max(float(np.max(continuation)), eps)

    kernel = utility * (0.75 + 0.25 * continuation)
    kernel[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    for _ in range(12):
        row_totals = kernel @ col_scale
        row_scale = np.divide(
            1.0,
            row_totals,
            out=np.zeros_like(row_totals),
            where=row_totals > 0.0,
        )

        col_totals = kernel.T @ row_scale
        col_scale = np.divide(
            1.0,
            col_totals,
            out=np.zeros_like(col_totals),
            where=col_totals > 0.0,
        )

    score = kernel * row_scale[:, None] * col_scale[None, :]
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
