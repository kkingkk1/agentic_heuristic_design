import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    valid &= ~np.eye(n, dtype=bool)

    finite_values = d[valid]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(finite_values))
    safe_d = np.where(valid, d, fallback)

    positive = safe_d[(safe_d > 0) & valid]
    eps = max(float(np.min(positive)) * 1e-9 if positive.size else 1e-12, 1e-12)

    row_scales = np.full(n, fallback, dtype=float)
    for i in range(n):
        row_values = safe_d[i, valid[i]]
        positive_values = row_values[row_values > 0]
        if positive_values.size:
            row_scales[i] = np.median(positive_values)

    row_scales = np.maximum(row_scales, eps)

    normalized = safe_d / row_scales[:, None]
    local_affinity = np.exp(-normalized)

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        order = np.argsort(safe_d[i])
        rank = 0
        for j in order:
            if valid[i, j]:
                ranks[i, j] = rank
                rank += 1

    rank_affinity = 1.0 / (1.0 + ranks)

    mutual_affinity = np.sqrt(
        np.exp(-safe_d / row_scales[:, None])
        * np.exp(-safe_d / row_scales[None, :])
    )

    score = (
        0.45 * local_affinity
        + 0.30 * rank_affinity
        + 0.25 * mutual_affinity
    )

    score[~valid] = 0.0

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
