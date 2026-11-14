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
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    endpoint_compatibility = np.sqrt(
        np.exp(-d / (local_scale[:, None] + eps))
        * np.exp(-d / (local_scale[None, :] + eps))
    )

    relative_length = d / (pair_scale + eps)
    short_edge = np.exp(-relative_length)

    ranks = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    alternatives = (
        work[:, None, :]
        + work[None, :, :]
        - d[:, :, None]
    )
    alternatives[:, :, np.arange(n)] = np.inf
    replacement_margin = np.min(alternatives, axis=2)
    replacement_margin = np.maximum(replacement_margin, 0.0)

    margin_scale = pair_scale + eps
    normalized_margin = replacement_margin / margin_scale

    finite_margin = normalized_margin[np.isfinite(normalized_margin)]
    if finite_margin.size:
        clip_value = float(np.quantile(finite_margin, 0.90))
        clip_value = max(clip_value, eps)
        clipped_margin = np.minimum(normalized_margin, clip_value) / clip_value
    else:
        clipped_margin = np.zeros_like(normalized_margin)

    indispensable = 1.0 - np.exp(-clipped_margin)

    score = (
        0.48 * endpoint_compatibility
        + 0.32 * short_edge
        + 0.15 * mutual_rank
        + 0.05 * indispensable
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size == 0:
        return np.zeros_like(score, dtype=float)

    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
