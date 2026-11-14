import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    safe_d = np.where(valid, d, np.inf)
    row_sorted = np.sort(safe_d, axis=1)

    nearest = np.where(np.isfinite(row_sorted[:, 0]), row_sorted[:, 0], 0.0)
    second = np.where(np.isfinite(row_sorted[:, 1]), row_sorted[:, 1], nearest)
    local_scale = np.maximum(second - nearest, np.finfo(float).eps)

    positive = safe_d[valid]
    global_scale = np.median(positive) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    row_scale = np.maximum(0.5 * (nearest + second), global_scale * 0.05)
    row_scale = row_scale[:, None]

    relative = safe_d / row_scale
    shortness = np.exp(-np.minimum(relative, 50.0))

    row_rank = np.argsort(np.argsort(safe_d, axis=1), axis=1)
    rank_weight = 1.0 / (1.0 + row_rank.astype(float))

    contrast = np.exp(
        -np.minimum(
            np.maximum(safe_d - nearest[:, None], 0.0)
            / (local_scale[:, None] + global_scale * 1e-6),
            50.0,
        )
    )

    score = (
        np.sqrt(shortness * shortness.T)
        * np.sqrt(rank_weight * rank_weight.T)
        * np.sqrt(contrast * contrast.T)
    )

    score *= valid
    np.fill_diagonal(score, 0.0)

    maximum = score.max()
    if maximum > 0:
        score /= maximum

    return score.astype(float)
