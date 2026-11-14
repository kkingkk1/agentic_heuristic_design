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

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    local_k = max(1, min(n - 1, int(np.ceil(1.5 * np.sqrt(n)))))
    nearest = np.partition(work, local_k - 1, axis=1)[:, :local_k]

    local_scale = np.maximum(np.mean(nearest, axis=1), eps)

    endpoint_relative = 0.5 * (
        d / local_scale[:, None]
        + d / local_scale[None, :]
    )
    score = np.exp(-endpoint_relative)

    candidate_count = max(
        1,
        min(n - 1, int(np.ceil(0.20 * (n - 1)))),
    )
    candidate_order = np.argpartition(
        work,
        candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    rows = np.arange(n)[:, None]
    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[rows, candidate_order] = True
    np.fill_diagonal(candidate_mask, False)

    mutual_candidate = candidate_mask & candidate_mask.T
    score *= 1.0 + 0.08 * mutual_candidate

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    low = float(np.min(values))
    high = float(np.max(values))

    if high > low:
        score = (score - low) / (high - low)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
