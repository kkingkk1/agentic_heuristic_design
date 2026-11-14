import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized stochastic-sampling prior for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    global_scale = max(global_scale, eps)

    # Estimate a local scale from each node's nearest valid positive edges.
    local_scales = np.empty(n, dtype=float)
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    for i in range(n):
        candidates = d[i, valid[i] & (d[i] > 0.0)]
        if candidates.size:
            candidates = np.sort(candidates)
            local_scales[i] = float(np.median(candidates[:k]))
        else:
            local_scales[i] = global_scale

    local_scales = np.maximum(local_scales, eps)
    pair_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        global_signal = np.exp(
            -np.clip(d / global_scale, 0.0, 60.0)
        )
        local_signal = np.exp(
            -np.clip(d / pair_scale, 0.0, 60.0)
        )

    global_signal = np.where(valid, global_signal, 0.0)
    local_signal = np.where(valid, local_signal, 0.0)

    # Rank agreement from both endpoints.
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks = np.where(valid, ranks, np.inf)

    endpoint_rank = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(
        endpoint_rank * endpoint_rank.T
    )
    mutual_rank = np.where(valid, mutual_rank, 0.0)

    # Explicit reinforcement for reciprocal nearest neighbors.
    nearest = np.argmin(safe, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if np.isfinite(safe[i, j]) and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    # Prefer edges connecting nodes with comparable neighborhood scales.
    scale_balance = np.exp(
        -0.75
        * np.abs(
            np.log(
                (local_scales[:, None] + eps)
                / (local_scales[None, :] + eps)
            )
        )
    )

    score = (
        0.38 * np.sqrt(global_signal * global_signal.T)
        + 0.32 * np.sqrt(local_signal * local_signal.T)
        + 0.18 * mutual_rank
        + 0.07 * mutual_nearest
        + 0.05 * scale_balance
    )

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
