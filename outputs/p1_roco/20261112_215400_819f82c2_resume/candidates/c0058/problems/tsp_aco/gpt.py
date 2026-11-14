import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    positive_finite = np.isfinite(d) & (d > 0) & offdiag

    positive_values = d[positive_finite]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(positive_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    x = d.copy()
    invalid = (~np.isfinite(x)) | (x < 0)
    x[invalid] = fallback * 1e6
    np.fill_diagonal(x, np.inf)

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    k2 = max(k, min(n - 1, 2 * k))

    nearest = np.partition(x, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )

    broader = np.partition(x, kth=k2 - 1, axis=1)[:, :k2]
    reference_scale = np.median(broader, axis=1)
    reference_scale = np.where(
        np.isfinite(reference_scale) & (reference_scale > 0),
        reference_scale,
        fallback,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, fallback * 1e-12)

    safe_x = np.maximum(x, 0.0)

    relative_shortness = np.exp(
        -safe_x / pair_scale
    )

    order = np.argsort(x, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    columns = np.arange(n)
    ranks[order, columns] = np.arange(n, dtype=float)[:, None]
    ranks = np.minimum(ranks, n - 1)

    rank_scale = max(1.0, float(k))
    rank_score = np.exp(-ranks / rank_scale)
    mutual_rank = np.sqrt(
        rank_score * rank_score.T
    )

    endpoint_compatibility = np.exp(
        -np.abs(
            np.log(
                (safe_x + 1e-12)
                / (reference_scale[:, None] + 1e-12)
            )
        )
    )
    endpoint_compatibility *= np.exp(
        -np.abs(
            np.log(
                (safe_x + 1e-12)
                / (reference_scale[None, :] + 1e-12)
            )
        )
    )

    score = (
        np.power(np.maximum(relative_shortness, 1e-30), 0.50)
        * np.power(np.maximum(mutual_rank, 1e-30), 0.35)
        * np.power(np.maximum(endpoint_compatibility, 1e-30), 0.15)
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
