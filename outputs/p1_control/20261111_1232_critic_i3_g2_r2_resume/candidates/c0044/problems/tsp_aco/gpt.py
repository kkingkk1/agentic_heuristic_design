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
        raise ValueError(
            "distance_matrix contains invalid non-finite off-diagonal values"
        )

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, fallback)

    k = min(n - 1, max(2, int(np.sqrt(n)) + 1))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]

    local_scale = np.median(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    threshold = np.max(nearest, axis=1)
    near = (work <= threshold[:, None]) | (work <= threshold[None, :])
    near &= finite
    np.fill_diagonal(near, False)

    row_order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    ranks /= max(1.0, n - 1)

    pair_scale = local_scale[:, None] + local_scale[None, :]
    normalized_distance = work / (pair_scale + eps)

    local_shortness = np.exp(-1.65 * normalized_distance)
    mutual_competition = np.exp(-1.80 * (ranks + ranks.T))

    smaller_scale = np.minimum(
        local_scale[:, None], local_scale[None, :]
    )
    larger_scale = np.maximum(
        local_scale[:, None], local_scale[None, :]
    )
    density_ratio = smaller_scale / (larger_scale + eps)
    scale_gap = np.maximum(0.0, 1.0 - density_ratio)

    sparse_endpoint_rank = np.where(
        local_scale[:, None] >= local_scale[None, :],
        ranks,
        ranks.T,
    )
    eligible_bridge = (
        (scale_gap > 0.20)
        & (sparse_endpoint_rank <= 0.35)
        & (normalized_distance <= 1.35)
        & near
    )

    bridge_gain = np.minimum(
        0.14,
        0.20
        * scale_gap
        * np.exp(-0.70 * normalized_distance),
    )
    isolation_bridge = 1.0 + np.where(eligible_bridge, bridge_gain, 0.0)

    score = np.full((n, n), 0.04, dtype=float)
    refined = (
        local_shortness
        * (0.50 + 0.50 * mutual_competition)
        * isolation_bridge
    )
    score[near] = refined[near]

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
