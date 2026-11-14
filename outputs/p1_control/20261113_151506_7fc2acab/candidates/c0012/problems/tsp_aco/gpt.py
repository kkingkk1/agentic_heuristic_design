import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    finite_values = d[valid]
    fallback = float(np.max(finite_values)) if finite_values.size else 1.0
    fallback = max(fallback, 1.0)

    work = np.where(valid, d, fallback * 10.0)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    eps = max(float(np.median(positive)) * 1e-12, 1e-12) if positive.size else 1e-12

    k = min(5, n - 1)
    nearest = np.sort(work, axis=1)[:, :k]
    local_scale = np.mean(np.where(np.isfinite(nearest), nearest, fallback), axis=1)
    global_scale = float(np.median(positive)) if positive.size else fallback
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )

    row_affinity = np.exp(-work / (local_scale[:, None] + eps))
    col_affinity = np.exp(-work / (local_scale[None, :] + eps))
    distance_affinity = np.sqrt(row_affinity * col_affinity)

    isolation = 1.0 / (local_scale + eps)
    isolation /= np.max(isolation) + eps
    endpoint_need = np.sqrt(isolation[:, None] * isolation[None, :])

    best_two_hop = np.full((n, n), np.inf, dtype=float)
    for middle in range(n):
        candidate = work[:, middle, None] + work[None, middle, :]
        candidate[middle, :] = np.inf
        candidate[:, middle] = np.inf
        best_two_hop = np.minimum(best_two_hop, candidate)

    bridge_advantage = (
        best_two_hop - work
    ) / (best_two_hop + work + eps)
    bridge_advantage = np.nan_to_num(
        bridge_advantage,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    bridge_score = 0.5 * (bridge_advantage + 1.0)

    score = (
        0.55 * distance_affinity
        + 0.25 * bridge_score
        + 0.20 * distance_affinity * endpoint_need
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
