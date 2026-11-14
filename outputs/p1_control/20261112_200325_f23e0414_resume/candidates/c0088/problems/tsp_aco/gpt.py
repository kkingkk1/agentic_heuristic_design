import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    normalized = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    endpoint_closeness = np.exp(-normalized)
    multiplicative = np.sqrt(
        endpoint_closeness * endpoint_closeness.T
    )

    endpoint_ratio = d / (local_scale[:, None] + eps)
    minimum_compatibility = 1.0 / (
        1.0 + np.maximum(endpoint_ratio, endpoint_ratio.T)
    )

    candidate_count = max(1, min(n - 1, int(np.ceil(np.log2(n + 1)))))
    candidate_threshold = np.partition(
        work, candidate_count - 1, axis=1
    )[:, candidate_count - 1]

    sparse_candidate = (
        (d <= candidate_threshold[:, None])
        | (d <= candidate_threshold[None, :])
    ).astype(float)

    endpoint_margin = np.maximum(
        candidate_threshold[:, None] - d,
        0.0,
    ) / (local_scale[:, None] + eps)
    alternative_margin = np.minimum(
        endpoint_margin,
        endpoint_margin.T,
    )
    alternative_margin = 1.0 - np.exp(
        -np.minimum(alternative_margin, 2.0)
    )

    score = (
        0.56 * multiplicative
        + 0.30 * minimum_compatibility
        + 0.09 * sparse_candidate * multiplicative
        + 0.05 * alternative_margin
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]
    lo = np.min(values)
    hi = np.max(values)

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
