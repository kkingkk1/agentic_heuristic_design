import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(d)
    if not np.all(finite | diagonal):
        raise ValueError("distance_matrix contains invalid non-finite off-diagonal values")

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, fallback)

    k = max(1, min(n - 1, int(np.sqrt(n))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]

    local_scale = np.median(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    row_order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = rank_values
    ranks /= max(1.0, n - 1)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = work / (pair_scale + eps)

    locality = np.exp(-normalized_distance)

    mutual_competition = np.exp(
        -1.5 * (ranks + ranks.T)
    )

    endpoint_density = np.minimum(
        local_scale[:, None] / (local_scale[None, :] + eps),
        local_scale[None, :] / (local_scale[:, None] + eps),
    )
    isolation_bridge = 1.0 + 0.30 * (
        1.0 - endpoint_density
    ) * np.exp(-0.5 * normalized_distance)

    savings = (
        local_scale[:, None]
        + local_scale[None, :]
        - work
    ) / (local_scale[:, None] + local_scale[None, :] + eps)
    savings = 0.75 + 0.25 * np.clip(savings, 0.0, 1.0)

    score = locality
    score *= 0.55 + 0.45 * mutual_competition
    score *= isolation_bridge
    score *= savings

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
