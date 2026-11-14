import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    finite_counts = valid.sum(axis=1)
    fallback = np.where(
        finite_counts > 0,
        np.min(work, axis=1, initial=np.inf),
        1.0,
    )
    fallback = np.where(np.isfinite(fallback), fallback, 1.0)

    local_scale = np.where(
        finite_counts > 1,
        np.partition(work, 1, axis=1)[:, 1],
        fallback,
    )
    local_scale = np.where(
        np.isfinite(local_scale),
        np.maximum(local_scale, eps),
        np.maximum(fallback, eps),
    )

    best = np.minimum(fallback, local_scale)
    regret = np.clip((local_scale - best) / np.maximum(local_scale, eps), 0.0, 1.0)

    row_ratio = d / np.maximum(local_scale[:, None], eps)
    col_ratio = d / np.maximum(local_scale[None, :], eps)

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 50.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 50.0))

    two_sided_quality = np.sqrt(row_quality * col_quality)

    endpoint_regret = 1.0 + 0.65 * (
        regret[:, None] + regret[None, :]
    )

    poor_for_both = np.sqrt(
        np.clip((1.0 - row_quality) * (1.0 - col_quality), 0.0, 1.0)
    )
    exclusivity = 1.0 - 0.35 * poor_for_both

    candidate_k = max(2, min(n - 1, int(np.ceil(np.sqrt(n)))))
    candidate_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        row = work[i].copy()
        row[i] = np.inf
        candidates = np.argpartition(row, candidate_k - 1)[:candidate_k]
        candidate_mask[i, candidates] = True

    mutual_or_candidate = candidate_mask | candidate_mask.T

    score = (
        two_sided_quality
        * endpoint_regret
        * exclusivity
        * mutual_or_candidate
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_totals = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0,
    )

    return score
