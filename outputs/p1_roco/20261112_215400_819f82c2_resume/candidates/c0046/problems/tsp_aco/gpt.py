import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    # Require a valid distance in both directions before symmetrization.
    pair_valid = valid & valid.T
    if not np.any(pair_valid):
        return scores

    symmetric_d = 0.5 * (d + d.T)
    safe_d = np.where(pair_valid, symmetric_d, np.inf)
    np.fill_diagonal(safe_d, np.inf)

    positive = safe_d[np.isfinite(safe_d) & (safe_d > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite = safe_d[np.isfinite(safe_d)]
        global_scale = float(np.median(finite)) if finite.size else 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    # Robust global shortness.
    distance_score = np.exp(
        -np.clip(safe_d / global_scale, 0.0, 50.0)
    )

    # Local scales from the closest few neighbors.
    usable = min(3, n - 1)
    sorted_d = np.sort(safe_d, axis=1)
    local_values = sorted_d[:, :usable]
    local_scale = np.mean(
        np.where(np.isfinite(local_values), local_values, global_scale),
        axis=1,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Bidirectional endpoint compatibility.
    row_gap = np.maximum(safe_d - local_scale[:, None], 0.0)
    col_gap = np.maximum(safe_d - local_scale[None, :], 0.0)
    compatibility = np.exp(
        -np.clip(row_gap / local_scale[:, None], 0.0, 50.0)
        -np.clip(col_gap / local_scale[None, :], 0.0, 50.0)
    )

    # Reciprocal rank from both endpoints.
    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    ranks[~pair_valid] = np.inf

    row_rank_score = 1.0 / (1.0 + ranks)
    reciprocal_rank = np.sqrt(row_rank_score * row_rank_score.T)
    reciprocal_rank[~pair_valid] = 0.0

    # Mutual-nearest-neighbor reinforcement.
    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if np.isfinite(safe_d[i, j]) and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    scores = (
        distance_score
        * (0.55 + 0.45 * reciprocal_rank)
        * (0.70 + 0.30 * compatibility)
        * (1.0 + 0.20 * mutual)
    )

    scores[~pair_valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
