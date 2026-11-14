import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(d) & (d >= 0)

    if np.any(valid):
        valid_values = d[valid]
        fallback = float(np.max(valid_values))
        positive_values = valid_values[valid_values > 0]
        scale = float(np.median(positive_values)) if positive_values.size else 1.0
    else:
        fallback = 1.0
        scale = 1.0

    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    clean = np.where(valid, d, fallback)
    np.fill_diagonal(clean, np.inf)

    finite_offdiag = np.isfinite(clean)
    positive_distances = clean[finite_offdiag & (clean > 0)]

    if positive_distances.size:
        local_scale = np.median(positive_distances, axis=0)
        if not np.isfinite(local_scale) or local_scale <= 0:
            local_scale = scale
    else:
        local_scale = scale

    inverse = np.zeros_like(clean, dtype=float)
    positive = np.isfinite(clean) & (clean > 0)
    inverse[positive] = 1.0 / clean[positive]
    zero_edges = np.isfinite(clean) & (clean == 0)
    inverse[zero_edges] = 1.0 / np.finfo(float).eps

    global_inverse = inverse * scale
    row_max = np.max(global_inverse, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)
    local_inverse = global_inverse / row_max

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_positions = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_positions[None, :]
    ranks[~np.isfinite(clean)] = float(n)

    rank_decay = max(1.0, np.sqrt(float(max(n - 1, 1))))
    rank_score = np.exp(-ranks / rank_decay)
    rank_score[~np.isfinite(clean)] = 0.0

    score = (
        0.60 * local_inverse
        + 0.25 * np.minimum(global_inverse, 1.0)
        + 0.15 * rank_score
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0

    offdiag = ~np.eye(n, dtype=bool)
    if np.any(offdiag):
        upper = np.percentile(score[offdiag], 99)
        if np.isfinite(upper) and upper > 0:
            score = np.clip(score / upper, 0.0, 1.0)

    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
