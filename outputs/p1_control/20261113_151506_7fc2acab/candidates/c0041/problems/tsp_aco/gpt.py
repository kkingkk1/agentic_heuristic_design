import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative values")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    # TSP instances are treated as undirected for the shared prior.
    d = 0.5 * (d + d.T)
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    positive = work[np.isfinite(work) & (work > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    # Robust local neighborhood scale for each endpoint.
    k = max(1, min(n - 1, int(np.sqrt(n))))
    sorted_rows = np.sort(work, axis=1)
    local_scale = np.mean(sorted_rows[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    valid = np.isfinite(work)

    # Direct shortness relative to both endpoint neighborhoods.
    shortness = np.exp(
        -work / (local_scale[:, None] + local_scale[None, :] + global_scale * eps)
    )
    shortness[~valid] = 0.0

    # Reciprocal endpoint preference based on stable local ranks.
    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = rank_values
    ranks[~valid] = np.inf

    rank_scale = max(1.0, np.sqrt(n))
    mutual_preference = np.exp(
        -0.5
        * (
            ranks / rank_scale
            + ranks.T / rank_scale
        )
    )
    mutual_preference[~valid] = 0.0

    # Structural support: preserve edges whose endpoints both have
    # compatible local neighborhoods.
    neighborhood_compatibility = np.exp(
        -np.abs(
            np.log(
                (local_scale[:, None] + eps)
                / (local_scale[None, :] + eps)
            )
        )
    )
    neighborhood_compatibility[~valid] = 0.0

    # Two-hop resistance: penalize edges that are easily replaced by
    # an indirect route through another node.
    via = work[:, :, None] + work[None, :, :]
    candidate_mask = (
        valid[:, :, None]
        & valid[None, :, :]
        & (np.arange(n)[None, :, None] != np.arange(n)[None, None, :])
        & (np.arange(n)[:, None, None] != np.arange(n)[None, :, None])
    )

    direct = work[:, None, :]
    detour_ratio = (via - direct) / (via + direct + global_scale * eps)
    detour_ratio = np.where(candidate_mask, detour_ratio, np.nan)

    best_detour = np.nanmin(detour_ratio, axis=1)
    best_detour = np.nan_to_num(best_detour, nan=0.0, posinf=0.0, neginf=0.0)

    detour_resistance = 0.5 + 0.5 * np.tanh(3.0 * best_detour)
    detour_resistance[~valid] = 0.0

    # Geometric-style combination keeps all three signals influential.
    score = (
        np.power(shortness + eps, 0.55)
        * np.power(0.5 + 0.5 * mutual_preference + eps, 0.25)
        * np.power(0.5 + 0.5 * detour_resistance + eps, 0.12)
        * np.power(0.5 + 0.5 * neighborhood_compatibility + eps, 0.08)
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
