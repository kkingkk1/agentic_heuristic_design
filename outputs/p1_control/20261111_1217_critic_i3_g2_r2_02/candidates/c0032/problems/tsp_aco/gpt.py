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

    k = max(1, min(n - 1, int(np.sqrt(n))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)

    valid_scale = np.isfinite(local_scale) & (local_scale > 0.0)
    scale_fallback = (
        float(np.median(local_scale[valid_scale]))
        if np.any(valid_scale)
        else fallback
    )
    local_scale = np.where(
        np.isfinite(local_scale),
        np.maximum(local_scale, eps),
        max(scale_fallback, eps),
    )

    normalized = work / (local_scale[:, None] + eps)
    normalized_sym = normalized + normalized.T

    locality = 1.0 / (1.0 + normalized_sym)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)
    ranks /= max(1.0, n - 1)

    mutual_rank = 1.0 / (1.0 + 1.5 * (ranks + ranks.T))

    global_scale = float(np.median(local_scale))
    isolation = np.clip(
        local_scale / (global_scale + eps),
        0.25,
        4.0,
    )
    isolation /= np.max(isolation)

    endpoint_isolation = np.maximum(isolation[:, None], isolation[None, :])
    relative_edge = np.minimum(
        work / (local_scale[:, None] + eps),
        work / (local_scale[None, :] + eps),
    )
    bridge = 1.0 + 0.30 * endpoint_isolation / (1.0 + relative_edge)

    score = locality * (0.55 + 0.45 * mutual_rank) * bridge
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
