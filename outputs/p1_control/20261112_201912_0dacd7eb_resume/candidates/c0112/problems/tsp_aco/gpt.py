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
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    scale = np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps

    relative_length = d / scale
    closeness = np.exp(-relative_length)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)
    base_score = closeness * mutual_rank

    if n > 2:
        replacement_margin = np.full((n, n), np.inf, dtype=float)

        chunk_size = max(1, min(n, int(2_000_000 // max(1, n))))
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            intermediates = np.arange(start, stop)

            detour = (
                d[:, intermediates, None]
                + d[None, intermediates, :]
                - d[None, None, :, :]
            )

            valid = (
                (intermediates[None, :, None] != np.arange(n)[None, None, :])
                & (intermediates[None, :, None] != np.arange(n)[None, :, None])
            )
            detour = np.where(valid, detour, np.inf)
            replacement_margin = np.minimum(
                replacement_margin,
                np.min(detour, axis=1),
            )

        replacement_margin = np.where(
            np.isfinite(replacement_margin),
            np.maximum(replacement_margin, 0.0),
            0.0,
        )
    else:
        replacement_margin = np.zeros((n, n), dtype=float)

    normalized_margin = replacement_margin / scale
    off_diagonal = ~np.eye(n, dtype=bool)

    local_order = np.argsort(
        -normalized_margin,
        axis=1,
        kind="mergesort",
    )
    local_rank = np.empty_like(local_order, dtype=float)
    local_rank[rows, local_order] = np.arange(n, dtype=float)[None, :]

    candidate_count = max(1, int(np.ceil(0.20 * (n - 1))))
    local_high = (local_rank < candidate_count) & off_diagonal
    mutual_high = local_high & local_high.T

    indispensable = 1.0 - np.exp(-normalized_margin)
    bonus = np.where(
        mutual_high,
        0.10 * indispensable,
        0.0,
    )

    score = base_score * (1.0 + bonus)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
